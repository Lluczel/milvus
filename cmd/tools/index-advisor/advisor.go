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
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus/pkg/v3/common"
)

const reportFormatVersion = 1

// Reasons a parsed record is not analyzed, in addition to the parse-level
// reasons in logparse.go.
const (
	skipNoFilterField        = "not_a_filter_request"      // formatter logs no expression (Insert, Connect, ...)
	skipHybridSearch         = "hybrid_search_unsupported" // list-encoded sub-request filters cannot be split reliably
	skipMethodOutOfScope     = "method_out_of_scope"       // logs an expression but is not a query or search (Delete, ...)
	skipRequestFailed        = "request_failed"
	skipNoFilter             = "no_filter" // a query or search without a filter
	skipUnknownCollection    = "unknown_collection"
	skipExpressionParseError = "expression_parse_error"
)

// Methods whose filter is analyzed. Each logs a single expression.
var analyzedMethods = map[string]bool{
	"Query": true, "Search": true,
	"/query": true, "/search": true,
	"/v2/vectordb/entities/query": true, "/v2/vectordb/entities/search": true,
}

// Methods whose $method_expr is a list of sub-request filters (grpc_info.go
// Expression), written without escaping inner quotes.
var hybridMethods = map[string]bool{
	"HybridSearch": true, "/v2/vectordb/entities/advanced_search": true, "/v2/vectordb/entities/hybrid_search": true,
}

// Recommendation actions.
const (
	actionCreateIndex  = "create_index"
	actionKeepExisting = "keep_existing"
	actionNoAction     = "no_action"
)

var fixedNotes = []string{
	"Advisory only: this tool never creates, alters or drops an index.",
	"Existing indexes are listed from metadata; this report does not show whether any query used them.",
	`Statistics shown as "unknown" were not provided with --stats; candidates that depend on them are marked conditional.`,
	"Fingerprints replace every literal with ?; raw expressions, user names, client addresses and trace IDs are not reported.",
}

// report is the advise output.
type report struct {
	FormatVersion   int                  `json:"format_version"`
	Inputs          inputsSummary        `json:"inputs"`
	Fingerprints    []fingerprintSummary `json:"fingerprints"`
	Recommendations []recommendation     `json:"recommendations"`
	Notes           []string             `json:"notes"`
}

type inputsSummary struct {
	LogFiles        []string    `json:"log_files"` // base names only
	RecordsTotal    int         `json:"records_total"`
	RecordsAnalyzed int         `json:"records_analyzed"`
	Skipped         []skipEntry `json:"skipped"`
}

type skipEntry struct {
	Reason string `json:"reason"`
	Count  int    `json:"count"`
}

type fingerprintSummary struct {
	DB          string         `json:"db"`
	Collection  string         `json:"collection"`
	Fingerprint string         `json:"fingerprint"`
	Count       int            `json:"count"`
	Methods     map[string]int `json:"methods"`
	TimeCostMs  *latency       `json:"time_cost_ms"`
}

// latency summarizes logged $time_cost values; nil when none was logged.
type latency struct {
	P50     float64 `json:"p50"`
	P95     float64 `json:"p95"`
	Max     float64 `json:"max"`
	Samples int     `json:"samples"`
}

type recommendation struct {
	Rank            int             `json:"rank"`
	DB              string          `json:"db"`
	Collection      string          `json:"collection"`
	Column          string          `json:"column"`
	DataType        string          `json:"data_type"`
	Action          string          `json:"action"`
	Observed        observed        `json:"observed"`
	ExistingIndexes []existingIndex `json:"existing_indexes"`
	Candidates      []candidate     `json:"candidates"`
	Statistics      statistics      `json:"statistics"`
	Notes           []string        `json:"notes,omitempty"`
	ValidationPlan  []string        `json:"validation_plan,omitempty"`
}

type observed struct {
	// Predicates is the number of analyzed requests that filter on the column.
	Predicates      int            `json:"predicates"`
	OpClasses       map[string]int `json:"op_classes"` // requests per op class
	TopFingerprints []string       `json:"top_fingerprints"`
	TimeCostMs      *latency       `json:"time_cost_ms"` // of those requests: the baseline to validate against
}

type existingIndex struct {
	IndexName string `json:"index_name"`
	IndexType string `json:"index_type"`
}

type candidate struct {
	IndexType string `json:"index_type"`
	// FieldName is the field_name to create the index on; for a dynamic key it
	// is the key itself, as DescribeIndex reports it.
	FieldName   string            `json:"field_name"`
	Params      map[string]string `json:"params,omitempty"`
	Serves      []string          `json:"serves"`     // op classes it serves on this column
	Predicates  int               `json:"predicates"` // requests with at least one served predicate
	Reasons     []string          `json:"reasons"`
	Conditional bool              `json:"conditional"` // depends on an unknown statistic or literal
	Source      string            `json:"source"`
}

// statistics values are numbers, or "unknown" when not provided.
type statistics struct {
	RowCount      statValue `json:"row_count"`
	DistinctCount statValue `json:"distinct_count"`
	NullFraction  statValue `json:"null_fraction"`
}

type statValue struct{ v *float64 }

func (s statValue) MarshalJSON() ([]byte, error) {
	if s.v == nil {
		return []byte(`"unknown"`), nil
	}
	return json.Marshal(*s.v)
}

func intStat(v *int64) statValue {
	if v == nil {
		return statValue{}
	}
	f := float64(*v)
	return statValue{&f}
}

// fingerprintAgg and columnAgg accumulate analyzed requests.
type fingerprintAgg struct {
	db, collection, fingerprint string
	count                       int
	methods                     map[string]int
	costs                       []time.Duration
}

type columnAgg struct {
	db, collection string
	info           *collectionInfo
	col            column
	requests       int
	opRequests     map[opClass]int
	opSets         map[string]int // sorted op classes of one request -> requests
	valueKinds     map[string]bool
	minLiteral     int  // shortest known LIKE literal; -1 when none seen
	unknownLiteral bool // a LIKE operand was a template variable
	fingerprints   map[string]int
	costs          []time.Duration
}

// advise turns access-log records into a report. It is deterministic: every
// map is iterated in sorted order and the output carries no wall-clock time.
func advise(logFiles []string, records []accessRecord, parseSkips skipCounts, meta *metadataSet, stats *statsSet) report {
	skipped := skipCounts{}
	for reason, n := range parseSkips {
		skipped[reason] += n
	}
	fingerprints := map[string]*fingerprintAgg{}
	columns := map[string]*columnAgg{}
	analyzed := 0

	for _, r := range records {
		info, parsed, reason := classify(r, meta)
		if reason != "" {
			skipped[reason]++
			continue
		}
		analyzed++
		cost, hasCost := r.timeCost()

		fpKey := collectionKey(info.db, info.name) + "|" + parsed.fingerprint
		fp := fingerprints[fpKey]
		if fp == nil {
			fp = &fingerprintAgg{db: info.db, collection: info.name, fingerprint: parsed.fingerprint, methods: map[string]int{}}
			fingerprints[fpKey] = fp
		}
		fp.count++
		fp.methods[r.method()]++
		if hasCost {
			fp.costs = append(fp.costs, cost)
		}

		// Group this request's predicates by column, so each column counts the
		// request once however many predicates it has on that column.
		byColumn := map[string][]predicate{}
		for _, p := range parsed.predicates {
			byColumn[p.column.String()] = append(byColumn[p.column.String()], p)
		}
		for colKey, preds := range byColumn {
			key := collectionKey(info.db, info.name) + "|" + colKey
			agg := columns[key]
			if agg == nil {
				agg = &columnAgg{
					db: info.db, collection: info.name, info: info, col: preds[0].column,
					opRequests: map[opClass]int{}, opSets: map[string]int{}, valueKinds: map[string]bool{},
					minLiteral: -1, fingerprints: map[string]int{},
				}
				columns[key] = agg
			}
			agg.add(preds, parsed.fingerprint, cost, hasCost)
		}
	}

	rep := report{
		FormatVersion: reportFormatVersion,
		Inputs: inputsSummary{
			LogFiles:        lo.Map(logFiles, func(f string, _ int) string { return filepath.Base(f) }),
			RecordsTotal:    len(records) + parseSkips[skipUnmatchedFormat],
			RecordsAnalyzed: analyzed,
			Skipped:         []skipEntry{},
		},
		Fingerprints:    []fingerprintSummary{},
		Recommendations: []recommendation{},
		Notes:           fixedNotes,
	}
	for _, reason := range lo.Keys(skipped) {
		rep.Inputs.Skipped = append(rep.Inputs.Skipped, skipEntry{Reason: reason, Count: skipped[reason]})
	}
	sort.Slice(rep.Inputs.Skipped, func(i, j int) bool { return rep.Inputs.Skipped[i].Reason < rep.Inputs.Skipped[j].Reason })

	for _, fp := range fingerprints {
		rep.Fingerprints = append(rep.Fingerprints, fingerprintSummary{
			DB: fp.db, Collection: fp.collection, Fingerprint: fp.fingerprint,
			Count: fp.count, Methods: fp.methods, TimeCostMs: summarize(fp.costs),
		})
	}
	sort.Slice(rep.Fingerprints, func(i, j int) bool {
		a, b := rep.Fingerprints[i], rep.Fingerprints[j]
		if a.Count != b.Count {
			return a.Count > b.Count
		}
		return a.DB+"."+a.Collection+"|"+a.Fingerprint < b.DB+"."+b.Collection+"|"+b.Fingerprint
	})

	for _, agg := range columns {
		rep.Recommendations = append(rep.Recommendations, recommend(agg, stats))
	}
	actionOrder := map[string]int{actionCreateIndex: 0, actionKeepExisting: 1, actionNoAction: 2}
	sort.Slice(rep.Recommendations, func(i, j int) bool {
		a, b := rep.Recommendations[i], rep.Recommendations[j]
		if actionOrder[a.Action] != actionOrder[b.Action] {
			return actionOrder[a.Action] < actionOrder[b.Action]
		}
		if a.Observed.Predicates != b.Observed.Predicates {
			return a.Observed.Predicates > b.Observed.Predicates
		}
		return a.DB+"."+a.Collection+"|"+a.Column < b.DB+"."+b.Collection+"|"+b.Column
	})
	for i := range rep.Recommendations {
		rep.Recommendations[i].Rank = i + 1
	}
	return rep
}

// classify decides whether a record is analyzed. It returns the record's
// collection and parsed filter, or the reason it is skipped.
func classify(r accessRecord, meta *metadataSet) (*collectionInfo, parsedExpr, string) {
	expr, hasExpr := r.expr()
	switch {
	case !hasExpr:
		return nil, parsedExpr{}, skipNoFilterField
	case hybridMethods[r.method()]:
		return nil, parsedExpr{}, skipHybridSearch
	case !analyzedMethods[r.method()]:
		return nil, parsedExpr{}, skipMethodOutOfScope
	case r.status() != "Successful":
		return nil, parsedExpr{}, skipRequestFailed
	case strings.TrimSpace(expr) == "":
		return nil, parsedExpr{}, skipNoFilter
	}
	info, ok := meta.lookup(r.database(), r.collection())
	if !ok {
		return nil, parsedExpr{}, skipUnknownCollection
	}
	parsed, err := analyzeExpr(info.helper, expr)
	if err != nil {
		return nil, parsedExpr{}, skipExpressionParseError
	}
	return info, parsed, ""
}

func (a *columnAgg) add(preds []predicate, fingerprint string, cost time.Duration, hasCost bool) {
	a.requests++
	a.fingerprints[fingerprint]++
	if hasCost {
		a.costs = append(a.costs, cost)
	}
	ops := map[opClass]bool{}
	for _, p := range preds {
		ops[p.op] = true
		if p.valueKind != "" {
			a.valueKinds[p.valueKind] = true
		}
		switch p.op {
		case opLikePrefix, opLikeSuffix, opLikeInfix, opLikePattern:
			if p.literalLen == literalLenUnknown {
				a.unknownLiteral = true
			} else if a.minLiteral < 0 || p.literalLen < a.minLiteral {
				a.minLiteral = p.literalLen
			}
		case opRegex:
			a.unknownLiteral = true // literal substrings of a regex are not extracted
		}
	}
	names := make([]string, 0, len(ops))
	for op := range ops {
		a.opRequests[op]++
		names = append(names, string(op))
	}
	sort.Strings(names)
	a.opSets[strings.Join(names, ",")]++
}

// requestsServed counts requests with at least one predicate in ops.
func (a *columnAgg) requestsServed(ops []opClass) int {
	n := 0
	for set, count := range a.opSets {
		for _, op := range strings.Split(set, ",") {
			if lo.Contains(ops, opClass(op)) {
				n += count
				break
			}
		}
	}
	return n
}

func recommend(a *columnAgg, stats *statsSet) recommendation {
	col := a.col
	rowCount, fs := stats.lookup(a.db, a.collection, col)
	rec := recommendation{
		DB: a.db, Collection: a.collection, Column: col.String(), DataType: dataTypeName(col),
		Observed: observed{
			Predicates: a.requests,
			OpClasses:  lo.MapKeys(a.opRequests, func(_ int, op opClass) string { return string(op) }),
			TimeCostMs: summarize(a.costs),
		},
		ExistingIndexes: []existingIndex{},
		Candidates:      []candidate{},
		Statistics:      statistics{RowCount: intStat(rowCount), DistinctCount: intStat(fs.DistinctCount)},
	}
	if fs.NullFraction != nil {
		rec.Statistics.NullFraction = statValue{fs.NullFraction}
	}
	rec.Observed.TopFingerprints = topKeys(a.fingerprints, 3)
	for _, idx := range a.info.indexesOn(col) {
		rec.ExistingIndexes = append(rec.ExistingIndexes, existingIndex{IndexName: idx.IndexName, IndexType: idx.IndexType})
	}
	if col.isPartitionKey {
		rec.Notes = append(rec.Notes, "This is the partition key: filters on it already prune partitions.")
	}

	switch {
	case col.isPrimaryKey:
		rec.Action = actionNoAction
		rec.Notes = append(rec.Notes, "Primary key: Milvus already maintains a primary-key index.")
		return rec
	case col.isDynamic && len(col.jsonPath) != 1:
		rec.Action = actionNoAction
		rec.Notes = append(rec.Notes, "Nested dynamic-field paths are not covered by this tool's rules.")
		return rec
	}

	var note string
	rec.Candidates, note = candidatesFor(a, fs.DistinctCount)
	switch {
	case len(rec.ExistingIndexes) > 0:
		rec.Action = actionKeepExisting
		rec.ValidationPlan = []string{
			fmt.Sprintf("On a non-production copy, compare the latency of %s with and without the existing index; its presence alone does not show it is used.", quoteList(rec.Observed.TopFingerprints)),
			"Compare the existing index type with the candidates below before changing anything; this tool does not alter indexes.",
		}
	case len(rec.Candidates) == 0:
		rec.Action = actionNoAction
		if note == "" {
			note = "No scalar index rule serves these predicates."
		}
		rec.Notes = append(rec.Notes, note)
	default:
		rec.Action = actionCreateIndex
		rec.ValidationPlan = validationPlan(rec)
	}
	return rec
}

// candidatesFor applies the suitability rules to a column, keeping only index
// types Milvus accepts. Unconditional candidates come first, then those that
// serve more requests, then rule order. When nothing can be suggested for a
// reason worth reporting, it is returned as a note.
func candidatesFor(a *columnAgg, distinct *int64) ([]candidate, string) {
	col, kind := a.col, kindOf(a.col)
	castType := ""
	if kind == kindJSONPath {
		if len(a.valueKinds) == 1 {
			castType = jsonCastType(lo.Keys(a.valueKinds)[0])
		}
		if castType == "" {
			return nil, "The json_cast_type cannot be inferred: this path is compared with literals of several types, or only with template variables."
		}
	}

	type ranked struct {
		candidate
		order int
	}
	var out []ranked
	for order, rule := range suitabilityRules {
		if !lo.Contains(rule.kinds, kind) || !indexAccepts(rule.indexType, col, castType) {
			continue
		}
		var serves []string
		for _, op := range rule.ops {
			if a.opRequests[op] > 0 {
				serves = append(serves, string(op))
			}
		}
		if len(serves) == 0 {
			continue
		}
		c := candidate{
			IndexType:  rule.indexType,
			FieldName:  col.field,
			Serves:     serves,
			Predicates: a.requestsServed(rule.ops),
			Reasons:    []string{rule.reason},
			Source:     rule.source,
		}
		if !applyCondition(&c, rule.condition, a, distinct) {
			continue
		}
		if kind == kindJSONPath {
			c.Params = map[string]string{common.JSONCastTypeKey: castType}
			if col.isDynamic {
				// DescribeIndex reports a dynamic-key index under the key itself.
				c.FieldName = col.jsonPath[0]
			} else {
				c.Params[common.JSONPathKey] = col.String()
			}
		}
		if rule.indexType == indexNgram {
			if c.Params == nil {
				c.Params = map[string]string{}
			}
			c.Params["min_gram"] = fmt.Sprint(ngramMinGram)
			c.Params["max_gram"] = fmt.Sprint(ngramMaxGram)
		}
		out = append(out, ranked{c, order})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Conditional != out[j].Conditional {
			return !out[i].Conditional
		}
		if out[i].Predicates != out[j].Predicates {
			return out[i].Predicates > out[j].Predicates
		}
		return out[i].order < out[j].order
	})
	return lo.Map(out, func(r ranked, _ int) candidate { return r.candidate }), ""
}

// applyCondition checks a rule's condition, adding the evidence to the
// candidate's reasons. It returns false when the evidence rules it out; unknown
// evidence keeps the candidate and marks it conditional.
func applyCondition(c *candidate, cond ruleCondition, a *columnAgg, distinct *int64) bool {
	switch cond {
	case condLowCardinality:
		switch {
		case kindOf(a.col) == kindBool:
			c.Reasons = append(c.Reasons, "a BOOL field has at most 2 distinct values")
		case distinct == nil:
			c.Conditional = true
			c.Reasons = append(c.Reasons, fmt.Sprintf("suits fewer than %d distinct values; distinct_count is unknown", bitmapMaxDistinct))
		case *distinct < bitmapMaxDistinct:
			c.Reasons = append(c.Reasons, fmt.Sprintf("distinct_count %d is below %d", *distinct, bitmapMaxDistinct))
		default:
			return false
		}
	case condNgramLiteral:
		switch {
		case a.minLiteral >= ngramMinGram && !a.unknownLiteral:
			c.Reasons = append(c.Reasons, fmt.Sprintf("every observed pattern literal has at least %d characters (shortest: %d)", ngramMinGram, a.minLiteral))
		case a.minLiteral >= 0 && a.minLiteral < ngramMinGram:
			c.Conditional = true
			c.Reasons = append(c.Reasons, fmt.Sprintf("a pattern literal of %d characters is shorter than min_gram %d and would fall back to a scan", a.minLiteral, ngramMinGram))
		default:
			c.Conditional = true
			c.Reasons = append(c.Reasons, "pattern literal lengths are unknown (template variables or regex)")
		}
	}
	return true
}

func validationPlan(rec recommendation) []string {
	best := rec.Candidates[0]
	params := ""
	if len(best.Params) > 0 {
		keys := lo.Keys(best.Params)
		sort.Strings(keys)
		params = " with params " + strings.Join(lo.Map(keys, func(k string, _ int) string { return k + "=" + best.Params[k] }), ", ")
	}
	baseline := "no logged latency to compare against; collect a baseline first"
	if l := rec.Observed.TimeCostMs; l != nil {
		baseline = fmt.Sprintf("baseline from %d logged requests: p50 %.3f ms, p95 %.3f ms", l.Samples, l.P50, l.P95)
	}
	return []string{
		fmt.Sprintf("Create the %s index on field %q%s in a non-production copy of %s.%s; this tool does not create it.", best.IndexType, best.FieldName, params, rec.DB, rec.Collection),
		fmt.Sprintf("Replay %s with representative values and compare p50/p95 latency (%s).", quoteList(rec.Observed.TopFingerprints), baseline),
		"Check index build time, memory and disk use, and insert latency before and after.",
		"Keep the index only if the replay shows a measurable improvement.",
	}
}

func dataTypeName(col column) string {
	switch {
	case len(col.jsonPath) > 0:
		return "JSON path"
	case kindOf(col) == kindArray:
		return fmt.Sprintf("Array<%s>", col.elementType)
	default:
		return col.dataType.String()
	}
}

// summarize returns nearest-rank percentiles in milliseconds, rounded to µs.
func summarize(costs []time.Duration) *latency {
	if len(costs) == 0 {
		return nil
	}
	sorted := append([]time.Duration(nil), costs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	pct := func(q float64) float64 {
		i := int(math.Ceil(q*float64(len(sorted)))) - 1
		return ms(sorted[max(i, 0)])
	}
	return &latency{P50: pct(0.5), P95: pct(0.95), Max: ms(sorted[len(sorted)-1]), Samples: len(sorted)}
}

func ms(d time.Duration) float64 {
	return math.Round(float64(d)/float64(time.Microsecond)) / 1000
}

// topKeys returns up to n keys with the highest counts, ties broken by key.
func topKeys(counts map[string]int, n int) []string {
	keys := lo.Keys(counts)
	sort.Slice(keys, func(i, j int) bool {
		if counts[keys[i]] != counts[keys[j]] {
			return counts[keys[i]] > counts[keys[j]]
		}
		return keys[i] < keys[j]
	})
	if len(keys) > n {
		keys = keys[:n]
	}
	return keys
}

func quoteList(items []string) string {
	return strings.Join(lo.Map(items, func(s string, _ int) string { return "`" + s + "`" }), ", ")
}
