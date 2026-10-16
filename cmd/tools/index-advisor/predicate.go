// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// opClass is the kind of comparison a predicate applies to a column. The rules
// table keys on it.
type opClass string

const (
	opEq           opClass = "eq"
	opNe           opClass = "ne"
	opRange        opClass = "range"
	opIn           opClass = "in"
	opLikePrefix   opClass = "like_prefix"
	opLikeSuffix   opClass = "like_suffix"
	opLikeInfix    opClass = "like_infix"
	opLikePattern  opClass = "like_pattern"
	opRegex        opClass = "regex"
	opTextMatch    opClass = "text_match"
	opContains     opClass = "contains"
	opIsNull       opClass = "is_null"
	opExists       opClass = "exists"
	opGeo          opClass = "geo"
	opNotIndexable opClass = "not_indexable" // field-vs-field and arithmetic comparisons
	opOther        opClass = "other"
)

// literalLenUnknown marks a LIKE predicate whose pattern is a template variable.
const literalLenUnknown = -1

// column identifies what a predicate reads: a field, or a JSON path inside a
// JSON field (dynamic fields are paths inside the hidden $meta field).
type column struct {
	field          string
	jsonPath       []string
	dataType       schemapb.DataType // type of the field itself; JSON for paths
	elementType    schemapb.DataType // element type of an ARRAY field
	isPrimaryKey   bool
	isPartitionKey bool
	isDynamic      bool
}

// String renders the column as a user would write it: price, meta["brand"],
// or $meta["color"] for a dynamic field.
func (c column) String() string {
	var b strings.Builder
	b.WriteString(c.field)
	for _, p := range c.jsonPath {
		b.WriteString("[")
		b.WriteString(strconv.Quote(p))
		b.WriteString("]")
	}
	return b.String()
}

// predicate is one indexable condition found in an expression.
type predicate struct {
	column column
	op     opClass
	// valueKind is the literal's type ("bool", "int", "float", "string",
	// "array"), or "" for a template variable. It picks json_cast_type.
	valueKind string
	// literalLen is the length in runes of the longest fixed run in a LIKE
	// pattern, which decides whether NGRAM can serve it; 0 for other ops.
	literalLen int
	negated    bool
}

func (p predicate) key() string {
	return fmt.Sprintf("%s|%s|%s|%t|%d", p.column, p.op, p.valueKind, p.negated, p.literalLen)
}

// parsedExpr is the result of analyzing one logged filter expression.
type parsedExpr struct {
	// fingerprint is the expression with every constant and template variable
	// replaced by "?" and AND/OR operands sorted, so the same query shape maps
	// to one string. It is the only form of the expression the report carries.
	fingerprint string
	predicates  []predicate // deduplicated, sorted by key
}

// analyzeExpr parses expr with Milvus's own parser and extracts its shape.
// ParseExprTemplate is used rather than ParseExpr: logged expressions keep
// template placeholders such as {p} without their values, and it returns the
// tree before Milvus's rewriter merges or reorders predicates.
func analyzeExpr(helper *typeutil.SchemaHelper, expr string) (parsedExpr, error) {
	tree, err := planparserv2.ParseExprTemplate(helper, expr, nil)
	if err != nil {
		return parsedExpr{}, err
	}
	w := &exprWalker{helper: helper, seen: map[string]bool{}}
	fp := w.walk(tree, false)
	sort.Slice(w.predicates, func(i, j int) bool { return w.predicates[i].key() < w.predicates[j].key() })
	return parsedExpr{fingerprint: fp, predicates: w.predicates}, nil
}

type exprWalker struct {
	helper     *typeutil.SchemaHelper
	predicates []predicate
	seen       map[string]bool
}

func (w *exprWalker) add(p predicate) {
	if k := p.key(); !w.seen[k] {
		w.seen[k] = true
		w.predicates = append(w.predicates, p)
	}
}

// walk records the predicates under e and returns e's fingerprint.
func (w *exprWalker) walk(e *planpb.Expr, negated bool) string {
	switch x := e.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		return w.walkLogical(x.BinaryExpr, negated)

	case *planpb.Expr_UnaryExpr:
		return "not (" + w.walk(x.UnaryExpr.GetChild(), !negated) + ")"

	case *planpb.Expr_UnaryRangeExpr:
		r := x.UnaryRangeExpr
		col := w.column(r.GetColumnInfo())
		p := predicate{column: col, valueKind: valueKind(r.GetValue(), r.GetTemplateVariableName()), negated: negated}
		var fp string
		switch r.GetOp() {
		case planpb.OpType_Equal:
			p.op, fp = opEq, col.String()+" == ?"
		case planpb.OpType_NotEqual:
			p.op, fp = opNe, col.String()+" != ?"
		case planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual:
			p.op, fp = opRange, col.String()+" "+opSymbol(r.GetOp())+" ?"
		case planpb.OpType_PrefixMatch:
			p.op, fp = opLikePrefix, col.String()+" like ?%"
			p.literalLen = literalLen(r.GetValue(), r.GetTemplateVariableName())
		case planpb.OpType_PostfixMatch:
			p.op, fp = opLikeSuffix, col.String()+" like %?"
			p.literalLen = literalLen(r.GetValue(), r.GetTemplateVariableName())
		case planpb.OpType_InnerMatch:
			p.op, fp = opLikeInfix, col.String()+" like %?%"
			p.literalLen = literalLen(r.GetValue(), r.GetTemplateVariableName())
		case planpb.OpType_Match:
			p.op, fp = opLikePattern, col.String()+" like ?"
			p.literalLen = patternLiteralLen(r.GetValue(), r.GetTemplateVariableName())
		case planpb.OpType_RegexMatch:
			p.op, fp = opRegex, col.String()+" =~ ?"
		case planpb.OpType_TextMatch, planpb.OpType_TextMatchFuzzy:
			p.op, fp = opTextMatch, "text_match("+col.String()+", ?)"
		case planpb.OpType_PhraseMatch:
			p.op, fp = opTextMatch, "phrase_match("+col.String()+", ?)"
		default:
			p.op, fp = opOther, col.String()+" "+r.GetOp().String()+" ?"
		}
		w.add(p)
		return fp

	case *planpb.Expr_BinaryRangeExpr:
		r := x.BinaryRangeExpr
		col := w.column(r.GetColumnInfo())
		w.add(predicate{column: col, op: opRange, valueKind: valueKind(r.GetLowerValue(), r.GetLowerTemplateVariableName()), negated: negated})
		lower, upper := "<", "<"
		if r.GetLowerInclusive() {
			lower = "<="
		}
		if r.GetUpperInclusive() {
			upper = "<="
		}
		return fmt.Sprintf("? %s %s %s ?", lower, col, upper)

	case *planpb.Expr_TermExpr:
		t := x.TermExpr
		col := w.column(t.GetColumnInfo())
		kind := ""
		if len(t.GetValues()) > 0 {
			kind = valueKind(t.GetValues()[0], "")
		}
		w.add(predicate{column: col, op: opIn, valueKind: kind, negated: negated})
		return col.String() + " in [?]"

	case *planpb.Expr_JsonContainsExpr:
		c := x.JsonContainsExpr
		col := w.column(c.GetColumnInfo())
		kind := ""
		if len(c.GetElements()) > 0 {
			kind = valueKind(c.GetElements()[0], "")
		}
		w.add(predicate{column: col, op: opContains, valueKind: kind, negated: negated})
		fn := "json_contains"
		if col.dataType == schemapb.DataType_Array {
			fn = "array_contains"
		}
		switch c.GetOp() {
		case planpb.JSONContainsExpr_ContainsAll:
			fn += "_all"
		case planpb.JSONContainsExpr_ContainsAny:
			fn += "_any"
		}
		return fn + "(" + col.String() + ", ?)"

	case *planpb.Expr_NullExpr:
		col := w.column(x.NullExpr.GetColumnInfo())
		w.add(predicate{column: col, op: opIsNull, negated: negated})
		if x.NullExpr.GetOp() == planpb.NullExpr_IsNotNull {
			return col.String() + " is not null"
		}
		return col.String() + " is null"

	case *planpb.Expr_ExistsExpr:
		col := w.column(x.ExistsExpr.GetInfo())
		w.add(predicate{column: col, op: opExists, negated: negated})
		return "exists " + col.String()

	case *planpb.Expr_GisfunctionFilterExpr:
		col := w.column(x.GisfunctionFilterExpr.GetColumnInfo())
		w.add(predicate{column: col, op: opGeo, negated: negated})
		return "st_" + strings.ToLower(x.GisfunctionFilterExpr.GetOp().String()) + "(" + col.String() + ", ?)"

	case *planpb.Expr_CompareExpr:
		c := x.CompareExpr
		left, right := w.column(c.GetLeftColumnInfo()), w.column(c.GetRightColumnInfo())
		w.add(predicate{column: left, op: opNotIndexable, negated: negated})
		w.add(predicate{column: right, op: opNotIndexable, negated: negated})
		return left.String() + " " + opSymbol(c.GetOp()) + " " + right.String()

	case *planpb.Expr_BinaryArithOpEvalRangeExpr:
		a := x.BinaryArithOpEvalRangeExpr
		col := w.column(a.GetColumnInfo())
		w.add(predicate{column: col, op: opNotIndexable, negated: negated})
		if a.GetArithOp() == planpb.ArithOpType_ArrayLength {
			return fmt.Sprintf("array_length(%s) %s ?", col, opSymbol(a.GetOp()))
		}
		return fmt.Sprintf("%s %s ? %s ?", col, arithSymbol(a.GetArithOp()), opSymbol(a.GetOp()))

	case *planpb.Expr_AlwaysTrueExpr:
		return "true"

	case *planpb.Expr_ValueExpr:
		return "?"

	case *planpb.Expr_CallExpr:
		args := make([]string, 0, len(x.CallExpr.GetFunctionParameters()))
		for _, arg := range x.CallExpr.GetFunctionParameters() {
			args = append(args, w.walk(arg, negated))
		}
		return x.CallExpr.GetFunctionName() + "(" + strings.Join(args, ", ") + ")"

	case *planpb.Expr_RandomSampleExpr:
		// random_sample(f) after a filter wraps that filter as its predicate.
		if pred := x.RandomSampleExpr.GetPredicate(); pred != nil {
			return w.walk(pred, negated) + " and random_sample(?)"
		}
		return "random_sample(?)"

	case *planpb.Expr_ColumnExpr:
		return w.column(x.ColumnExpr.GetInfo()).String()

	default:
		// Sampling, element filters, membership filters and other newer nodes:
		// keep the node kind in the fingerprint without guessing at columns.
		return "<" + strings.TrimPrefix(fmt.Sprintf("%T", x), "*planpb.Expr_") + ">"
	}
}

// walkLogical flattens a chain of the same logical operator and sorts its
// operands, so "a and b" and "b and a" share a fingerprint.
func (w *exprWalker) walkLogical(b *planpb.BinaryExpr, negated bool) string {
	op := b.GetOp()
	var operands []string
	var collect func(e *planpb.Expr)
	collect = func(e *planpb.Expr) {
		if child := e.GetBinaryExpr(); child != nil && child.GetOp() == op {
			collect(child.GetLeft())
			collect(child.GetRight())
			return
		}
		s := w.walk(e, negated)
		if e.GetBinaryExpr() != nil {
			s = "(" + s + ")"
		}
		operands = append(operands, s)
	}
	collect(b.GetLeft())
	collect(b.GetRight())
	sort.Strings(operands)

	sep := " and "
	if op == planpb.BinaryExpr_LogicalOr {
		sep = " or "
	}
	return strings.Join(operands, sep)
}

func (w *exprWalker) column(info *planpb.ColumnInfo) column {
	col := column{
		field:          strconv.FormatInt(info.GetFieldId(), 10),
		jsonPath:       info.GetNestedPath(),
		dataType:       info.GetDataType(),
		elementType:    info.GetElementType(),
		isPrimaryKey:   info.GetIsPrimaryKey(),
		isPartitionKey: info.GetIsPartitionKey(),
	}
	if f, err := w.helper.GetFieldFromID(info.GetFieldId()); err == nil {
		col.field = f.GetName()
		col.isDynamic = f.GetIsDynamic()
		if col.dataType == schemapb.DataType_None {
			col.dataType = f.GetDataType()
		}
	}
	return col
}

func opSymbol(op planpb.OpType) string {
	switch op {
	case planpb.OpType_GreaterThan:
		return ">"
	case planpb.OpType_GreaterEqual:
		return ">="
	case planpb.OpType_LessThan:
		return "<"
	case planpb.OpType_LessEqual:
		return "<="
	case planpb.OpType_Equal:
		return "=="
	case planpb.OpType_NotEqual:
		return "!="
	default:
		return op.String()
	}
}

func arithSymbol(op planpb.ArithOpType) string {
	switch op {
	case planpb.ArithOpType_Add:
		return "+"
	case planpb.ArithOpType_Sub:
		return "-"
	case planpb.ArithOpType_Mul:
		return "*"
	case planpb.ArithOpType_Div:
		return "/"
	case planpb.ArithOpType_Mod:
		return "%"
	default:
		return strings.ToLower(op.String())
	}
}

func valueKind(v *planpb.GenericValue, templateVar string) string {
	if templateVar != "" {
		return ""
	}
	switch v.GetVal().(type) {
	case *planpb.GenericValue_BoolVal:
		return "bool"
	case *planpb.GenericValue_Int64Val:
		return "int"
	case *planpb.GenericValue_FloatVal:
		return "float"
	case *planpb.GenericValue_StringVal:
		return "string"
	case *planpb.GenericValue_ArrayVal:
		return "array"
	default:
		return ""
	}
}

// literalLen is the rune length of a prefix, suffix or infix LIKE operand. The
// parser has already stripped the wildcards and unescaped the operand, so
// "50\\%off%" arrives here as PrefixMatch "50%off".
func literalLen(v *planpb.GenericValue, templateVar string) int {
	if templateVar != "" {
		return literalLenUnknown
	}
	return utf8.RuneCountInString(v.GetStringVal())
}

// patternLiteralLen returns the longest run of characters that are not LIKE
// wildcards in a general pattern (OpType_Match), which the parser keeps raw:
// % and _ are wildcards unless escaped with a backslash.
func patternLiteralLen(v *planpb.GenericValue, templateVar string) int {
	if templateVar != "" {
		return literalLenUnknown
	}
	s := v.GetStringVal()
	longest, current := 0, 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '\\' && i+size < len(s):
			_, next := utf8.DecodeRuneInString(s[i+size:])
			size += next
			current++
		case r == '%' || r == '_':
			current = 0
		default:
			current++
		}
		if current > longest {
			longest = current
		}
		i += size
	}
	return longest
}
