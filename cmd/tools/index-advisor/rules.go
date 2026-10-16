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
	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	indexInverted = "INVERTED"
	indexBitmap   = "BITMAP"
	indexSTLSort  = "STL_SORT"
	indexTrie     = "Trie"
	indexNgram    = "NGRAM"
	indexRTree    = "RTREE"
)

// bitmapMaxDistinct is the cardinality below which BITMAP is suggested:
// "bitmap indexes are most effective when the cardinality of a field is less
// than 500" (https://milvus.io/docs/bitmap.md).
const bitmapMaxDistinct = 500

// Suggested NGRAM parameters. A LIKE literal shorter than min_gram cannot use
// the index and falls back to a full scan (https://milvus.io/docs/ngram.md).
const (
	ngramMinGram = 2
	ngramMaxGram = 3
)

// columnKind is the type class rules are written against. A JSON path is
// classified by the literal types compared against it.
type columnKind string

const (
	kindBool        columnKind = "bool"
	kindInt         columnKind = "int"
	kindFloat       columnKind = "float"
	kindString      columnKind = "string"
	kindTimestamptz columnKind = "timestamptz"
	kindArray       columnKind = "array"
	kindJSONPath    columnKind = "json_path"
	kindGeometry    columnKind = "geometry"
	kindOther       columnKind = "other"
)

func kindOf(col column) columnKind {
	t := col.dataType
	switch {
	case typeutil.IsJSONType(t):
		if len(col.jsonPath) == 0 {
			return kindOther // a whole JSON document, not a path
		}
		return kindJSONPath
	case typeutil.IsBoolType(t):
		return kindBool
	case typeutil.IsIntegerType(t):
		return kindInt
	case typeutil.IsFloatingType(t):
		return kindFloat
	case typeutil.IsStringType(t):
		return kindString
	case typeutil.IsTimestamptzType(t):
		return kindTimestamptz
	case typeutil.IsArrayType(t):
		return kindArray
	case typeutil.IsGeometryType(t):
		return kindGeometry
	default:
		return kindOther
	}
}

// jsonCastType maps the literal type compared against a JSON path to the
// json_cast_type an index on that path needs; "" when it cannot be inferred.
func jsonCastType(valueKind string) string {
	switch valueKind {
	case "bool":
		return "BOOL"
	case "int", "float":
		return "DOUBLE"
	case "string":
		return "VARCHAR"
	default:
		return ""
	}
}

// indexAccepts reports whether Milvus accepts an index of type indexType on col
// with the given json_cast_type (for JSON paths). It mirrors CheckValidDataType
// and the JSON cast checks in internal/util/indexparamcheck (that package is not
// imported because it links the C++ core through internal/util/cgo):
//   - inverted_checker.go: bool, numeric, string, array, JSON; JSON casts
//     BOOL, DOUBLE, VARCHAR, ARRAY_BOOL, ARRAY_DOUBLE, ARRAY_VARCHAR, JSON
//   - bitmap_index_checker.go: not a primary key; bool, integer, string, and
//     arrays of those; JSON casts BOOL, VARCHAR
//   - stl_sort_checker.go: numeric, string, timestamptz; JSON casts DOUBLE, VARCHAR
//   - trie_checker.go: string
//   - ngram_index_checker.go: string; JSON cast VARCHAR
//   - rtree_checker.go: geometry
func indexAccepts(indexType string, col column, castType string) bool {
	t, elem := col.dataType, col.elementType
	isJSON := typeutil.IsJSONType(t)
	switch indexType {
	case indexInverted:
		if isJSON {
			return lo.Contains([]string{"BOOL", "DOUBLE", "VARCHAR", "ARRAY_BOOL", "ARRAY_DOUBLE", "ARRAY_VARCHAR", "JSON"}, castType)
		}
		return typeutil.IsBoolType(t) || typeutil.IsArithmetic(t) || typeutil.IsStringType(t) || typeutil.IsArrayType(t)
	case indexBitmap:
		if col.isPrimaryKey {
			return false
		}
		if isJSON {
			return lo.Contains([]string{"BOOL", "VARCHAR"}, castType)
		}
		scalar := func(dt schemapb.DataType) bool {
			return typeutil.IsBoolType(dt) || typeutil.IsIntegerType(dt) || typeutil.IsStringType(dt)
		}
		return scalar(t) || (typeutil.IsArrayType(t) && scalar(elem))
	case indexSTLSort:
		if isJSON {
			return lo.Contains([]string{"DOUBLE", "VARCHAR"}, castType)
		}
		return typeutil.IsArithmetic(t) || typeutil.IsStringType(t) || typeutil.IsTimestamptzType(t)
	case indexTrie:
		return typeutil.IsStringType(t)
	case indexNgram:
		if isJSON {
			return castType == "VARCHAR"
		}
		return typeutil.IsStringType(t)
	case indexRTree:
		return typeutil.IsGeometryType(t)
	default:
		return false
	}
}

// ruleCondition is an extra requirement a suggestion depends on.
type ruleCondition int

const (
	condNone           ruleCondition = iota
	condLowCardinality               // distinct_count < bitmapMaxDistinct
	condNgramLiteral                 // LIKE literals at least ngramMinGram long
)

// suitabilityRule suggests indexType for predicates of the given op classes on
// columns of the given kinds. These are seed rules from the Milvus scalar index
// docs, cited per rule; they pick candidates and never estimate cost.
type suitabilityRule struct {
	indexType string
	ops       []opClass
	kinds     []columnKind
	condition ruleCondition
	reason    string
	source    string
}

var suitabilityRules = []suitabilityRule{
	{
		indexType: indexBitmap, ops: []opClass{opEq, opIn}, kinds: []columnKind{kindBool, kindInt, kindString}, condition: condLowCardinality,
		reason: "equality and IN filters on a low-cardinality field",
		source: "https://milvus.io/docs/bitmap.md",
	},
	{
		indexType: indexBitmap, ops: []opClass{opContains}, kinds: []columnKind{kindArray}, condition: condLowCardinality,
		reason: "array_contains filters on an array with few distinct elements",
		source: "https://milvus.io/docs/bitmap.md",
	},
	{
		indexType: indexInverted, ops: []opClass{opEq, opIn, opRange, opLikePrefix}, kinds: []columnKind{kindBool, kindInt, kindFloat, kindString},
		reason: "point, range and prefix-match filters at any cardinality",
		source: "https://milvus.io/docs/scalar_index.md",
	},
	{
		indexType: indexInverted, ops: []opClass{opContains}, kinds: []columnKind{kindArray},
		reason: "array_contains filters",
		source: "https://milvus.io/docs/scalar_index.md",
	},
	{
		indexType: indexInverted, ops: []opClass{opEq, opIn, opRange}, kinds: []columnKind{kindJSONPath},
		reason: "filters on one JSON path, indexed with json_path and json_cast_type",
		source: "https://milvus.io/docs/scalar_index.md",
	},
	{
		indexType: indexSTLSort, ops: []opClass{opEq, opIn, opRange}, kinds: []columnKind{kindInt, kindFloat, kindString, kindTimestamptz},
		reason: "comparison, range and IN filters on a sorted column",
		source: "https://milvus.io/docs/stl-sort.md",
	},
	{
		indexType: indexTrie, ops: []opClass{opLikePrefix}, kinds: []columnKind{kindString},
		reason: "prefix-match filters",
		source: "https://milvus.io/docs/index-scalar-fields.md",
	},
	{
		indexType: indexNgram, ops: []opClass{opLikePrefix, opLikeSuffix, opLikeInfix, opLikePattern, opRegex}, kinds: []columnKind{kindString, kindJSONPath}, condition: condNgramLiteral,
		reason: "LIKE and regex filters with literal substrings",
		source: "https://milvus.io/docs/ngram.md",
	},
	{
		indexType: indexRTree, ops: []opClass{opGeo}, kinds: []columnKind{kindGeometry},
		reason: "spatial filters on a geometry field",
		source: "https://milvus.io/docs/rtree.md",
	},
}
