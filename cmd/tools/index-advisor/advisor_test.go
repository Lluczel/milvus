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
	"bytes"
	"encoding/json"
	"flag"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var update = flag.Bool("update", false, "rewrite the golden reports in testdata/")

// TestAdviseGolden runs the CLI on the fixture log, with and without --stats,
// and compares the output byte for byte. Regenerate with -update after an
// intended change, then review the diff.
func TestAdviseGolden(t *testing.T) {
	for golden, extra := range map[string][]string{
		"testdata/expected_report.json":       nil,
		"testdata/expected_report_stats.json": {"--stats", "testdata/stats.json"},
	} {
		args := append([]string{"advise", "--log", "testdata/access_log.txt", "--metadata", "testdata/metadata.json"}, extra...)
		stdout, stderr, err := runCLI(args...)
		require.NoError(t, err, stderr)

		again, _, err := runCLI(args...)
		require.NoError(t, err)
		require.Equal(t, stdout, again, "two runs on the same input must be byte-identical")

		if *update {
			require.NoError(t, os.WriteFile(golden, []byte(stdout), 0o600))
		}
		want, err := os.ReadFile(golden)
		require.NoError(t, err)
		assert.Equal(t, string(want), stdout, "%s is stale; rerun with -update and review the diff", golden)
	}
}

// TestAdviseGoldenInvariants pins the acceptance criteria of issue #53828 on the
// golden output, so a regenerated golden cannot silently drop them.
func TestAdviseGoldenInvariants(t *testing.T) {
	stdout, _, err := runCLI("advise", "--log", "testdata/access_log.txt", "--metadata", "testdata/metadata.json")
	require.NoError(t, err)

	byColumn := map[string]map[string]any{}
	var rep map[string]any
	require.NoError(t, jsonUnmarshal(stdout, &rep))
	for _, r := range rep["recommendations"].([]any) {
		rec := r.(map[string]any)
		byColumn[rec["column"].(string)] = rec
	}

	price := byColumn["price"]
	assert.EqualValues(t, 1, price["rank"], "the most filtered column ranks first")
	assert.Equal(t, "create_index", price["action"])
	assert.NotEmpty(t, price["validation_plan"], "every create_index carries a validation plan")

	city := byColumn["city"]
	assert.Equal(t, "keep_existing", city["action"])
	assert.Contains(t, rep["notes"], "Existing indexes are listed from metadata; this report does not show whether any query used them.")

	stats := city["statistics"].(map[string]any)
	assert.Equal(t, "unknown", stats["distinct_count"], "missing statistics are reported as unknown")
	for _, c := range city["candidates"].([]any) {
		if c.(map[string]any)["index_type"] == "BITMAP" {
			assert.Equal(t, true, c.(map[string]any)["conditional"], "a cardinality-dependent candidate is conditional without stats")
		}
	}

	assert.Equal(t, "no_action", byColumn["id"]["action"], "the primary key is never recommended")

	// Redaction: no literal from the fixture's filters, and no client address
	// or trace ID, reaches the report.
	for _, leaked := range []string{`"SF"`, `"LA"`, `"red"`, "SKU-1", "%123%", "[1, 2, 3]", "4.5", "127.0.0.1", "tcp-", "Python-3.0.2", "00000000000000000000000000000001"} {
		assert.NotContains(t, stdout, leaked)
	}
}

// adviseTestMeta is a metadata set with one collection, t, over testSchemaHelper.
func adviseTestMeta(t *testing.T, indexes map[string][]indexMetadata) *metadataSet {
	if indexes == nil {
		indexes = map[string][]indexMetadata{}
	}
	return &metadataSet{collections: map[string]*collectionInfo{
		"default.t": {db: "default", name: "t", helper: testSchemaHelper(t), indexes: indexes},
	}}
}

func queryRecord(method, status, collection, expr string, cost time.Duration) accessRecord {
	values := map[string]string{
		fieldMethodName: method, fieldStatus: status, fieldDatabase: "default",
		fieldCollection: collection, fieldExpr: expr, fieldTimeCost: cost.String(),
	}
	if cost == 0 {
		values[fieldTimeCost] = "Unknown"
	}
	return accessRecord{formatter: "query", values: values}
}

func recFor(t *testing.T, rep report, col string) recommendation {
	for _, r := range rep.Recommendations {
		if r.Column == col {
			return r
		}
	}
	require.Failf(t, "no recommendation", "column %s", col)
	return recommendation{}
}

func candidateTypes(r recommendation) []string {
	var out []string
	for _, c := range r.Candidates {
		s := c.IndexType
		if c.Conditional {
			s += "?"
		}
		out = append(out, s)
	}
	return out
}

func TestAdviseSkipReasons(t *testing.T) {
	records := []accessRecord{
		queryRecord("Query", "Successful", "missing", "price > 1", time.Millisecond),
		queryRecord("Query", "Successful", "t", "price >", time.Millisecond),
		queryRecord("Query", "Successful", "t", "  ", time.Millisecond),
		queryRecord("Query", "Failed", "t", "price > 1", time.Millisecond),
		queryRecord("Delete", "Successful", "t", "price > 1", time.Millisecond),
		queryRecord("HybridSearch", "Successful", "t", `["price > 1"]`, time.Millisecond),
		{formatter: "base", values: map[string]string{fieldMethodName: "Insert"}},
		queryRecord("Query", "Successful", "t", "price > 1", 0),
	}
	rep := advise([]string{"/home/someone/logs/access.log"}, records, skipCounts{skipOrphanLine: 2, skipUnmatchedFormat: 1}, adviseTestMeta(t, nil), nil)

	assert.Equal(t, []string{"access.log"}, rep.Inputs.LogFiles, "only base names, never local paths")
	assert.Equal(t, 9, rep.Inputs.RecordsTotal)
	assert.Equal(t, 1, rep.Inputs.RecordsAnalyzed)
	assert.Equal(t, []skipEntry{
		{skipExpressionParseError, 1},
		{skipHybridSearch, 1},
		{skipMethodOutOfScope, 1},
		{skipNoFilter, 1},
		{skipNoFilterField, 1},
		{skipOrphanLine, 2},
		{skipRequestFailed, 1},
		{skipUnknownCollection, 1},
		{skipUnmatchedFormat, 1},
	}, rep.Inputs.Skipped)

	require.Len(t, rep.Fingerprints, 1)
	assert.Nil(t, rep.Fingerprints[0].TimeCostMs, "an Unknown time_cost is missing, not zero")
	assert.Contains(t, recFor(t, rep, "price").ValidationPlan[1], "no logged latency")
}

func TestAdviseCandidates(t *testing.T) {
	meta := adviseTestMeta(t, nil)
	run := func(stats *statsSet, exprs ...string) report {
		var records []accessRecord
		for _, e := range exprs {
			records = append(records, queryRecord("Query", "Successful", "t", e, time.Millisecond))
		}
		return advise(nil, records, skipCounts{}, meta, stats)
	}
	ndv := func(n int64) *int64 { return &n }
	stats := &statsSet{
		rowCount: map[string]*int64{"default.t": ndv(1000)},
		fields: map[string]map[string]fieldStats{"default.t": {
			"city": {DistinctCount: ndv(4)}, "sku": {DistinctCount: ndv(900)},
		}},
	}

	rep := run(stats, `city == "a"`, `sku in ["x"]`, `city == "b"`)
	assert.Equal(t, []string{"BITMAP", "INVERTED", "STL_SORT"}, candidateTypes(recFor(t, rep, "city")), "low cardinality puts BITMAP first")
	assert.Equal(t, []string{"INVERTED", "STL_SORT"}, candidateTypes(recFor(t, rep, "sku")), "high cardinality rules BITMAP out")
	assert.Equal(t, "city", rep.Recommendations[0].Column, "ranked by number of filtering requests")

	rep = run(nil, `sku like "%ab%"`, `sku like "%a%"`)
	ngram := recFor(t, rep, "sku")
	assert.Equal(t, []string{"NGRAM?"}, candidateTypes(ngram), "a literal shorter than min_gram makes NGRAM conditional")
	assert.Contains(t, ngram.Candidates[0].Reasons[1], "shorter than min_gram")
	assert.Equal(t, map[string]string{"min_gram": "2", "max_gram": "3"}, ngram.Candidates[0].Params)

	rep = run(nil, "sku like {p}", `sku =~ "a.*"`)
	assert.Equal(t, []string{"NGRAM?"}, candidateTypes(recFor(t, rep, "sku")), "template and regex literals are unknown")

	rep = run(nil, `meta["k"] == 1`, `meta["k"] == "x"`, `meta["j"] == {v}`, `meta["b"] == true`)
	assert.Equal(t, "no_action", recFor(t, rep, `meta["k"]`).Action, "mixed literal types give no cast type")
	assert.Contains(t, recFor(t, rep, `meta["k"]`).Notes[0], "json_cast_type cannot be inferred")
	assert.Equal(t, "no_action", recFor(t, rep, `meta["j"]`).Action, "template-only paths give no cast type")
	b := recFor(t, rep, `meta["b"]`)
	assert.Equal(t, map[string]string{"json_cast_type": "BOOL", "json_path": `meta["b"]`}, b.Candidates[0].Params)
	assert.Equal(t, "meta", b.Candidates[0].FieldName)

	rep = run(nil, `color == "red"`, `dyn["a"] == 1`)
	color := recFor(t, rep, `$meta["color"]`)
	assert.Equal(t, "color", color.Candidates[0].FieldName, "a dynamic key is indexed under the key itself")
	assert.Equal(t, map[string]string{"json_cast_type": "VARCHAR"}, color.Candidates[0].Params)
	assert.Equal(t, "no_action", recFor(t, rep, `$meta["dyn"]["a"]`).Action)

	rep = run(nil, "store == 3", `city != "x"`, `st_contains(loc, "POINT (1 1)")`, "in_stock == true", "price > rating")
	store := recFor(t, rep, "store")
	assert.Contains(t, store.Notes[0], "partition key")
	assert.Equal(t, "create_index", store.Action)
	assert.Equal(t, []string{"No scalar index rule serves these predicates."}, recFor(t, rep, "city").Notes)
	assert.Equal(t, []string{"RTREE"}, candidateTypes(recFor(t, rep, "loc")))
	assert.Equal(t, []string{"BITMAP", "INVERTED"}, candidateTypes(recFor(t, rep, "in_stock")), "BOOL needs no statistics for BITMAP")
	assert.Equal(t, "no_action", recFor(t, rep, "rating").Action)
}

func TestAdviseKeepExisting(t *testing.T) {
	meta := adviseTestMeta(t, map[string][]indexMetadata{"price": {{IndexName: "p", IndexType: "STL_SORT"}}})
	rep := advise(nil, []accessRecord{queryRecord("Query", "Successful", "t", "price > 1", time.Millisecond)}, skipCounts{}, meta, nil)
	price := recFor(t, rep, "price")
	assert.Equal(t, "keep_existing", price.Action)
	assert.Equal(t, []existingIndex{{IndexName: "p", IndexType: "STL_SORT"}}, price.ExistingIndexes)
	assert.Contains(t, price.ValidationPlan[0], "does not show it is used")
}

// The report must not depend on record order.
func TestAdviseOrderIndependent(t *testing.T) {
	records, err := readRecordsFile("testdata/access_log.txt", defaultFormatterSet(t), skipCounts{})
	require.NoError(t, err)
	meta := loadFixtureMetadata(t)

	var want bytes.Buffer
	require.NoError(t, writeJSON(&want, advise(nil, records, skipCounts{}, meta, nil)))
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 5; i++ {
		rng.Shuffle(len(records), func(a, b int) { records[a], records[b] = records[b], records[a] })
		var got bytes.Buffer
		require.NoError(t, writeJSON(&got, advise(nil, records, skipCounts{}, meta, nil)))
		require.Equal(t, want.String(), got.String())
	}
}

func TestSummarize(t *testing.T) {
	assert.Nil(t, summarize(nil))
	var costs []time.Duration
	for i := 1; i <= 20; i++ {
		costs = append(costs, time.Duration(i)*time.Millisecond)
	}
	assert.Equal(t, &latency{P50: 10, P95: 19, Max: 20, Samples: 20}, summarize(costs))
	assert.Equal(t, &latency{P50: 0.001, P95: 0.001, Max: 0.001, Samples: 1}, summarize([]time.Duration{1499 * time.Nanosecond}))
}

func TestRunAdviseFlags(t *testing.T) {
	_, _, err := runCLI("advise", "--log", "testdata/access_log.txt")
	assert.ErrorContains(t, err, "--metadata")
	_, _, err = runCLI("advise", "-h")
	assert.NoError(t, err)
	for _, args := range [][]string{
		{"--metadata", "testdata/missing.json"},
		{"--metadata", "testdata/metadata.json", "--stats", "testdata/missing.json"},
		{"--metadata", "testdata/metadata.json", "--milvus-config", "testdata/missing.yaml"},
		{"--metadata", "testdata/metadata.json", "--log", "testdata/missing.txt"},
	} {
		_, _, err := runCLI(append([]string{"advise", "--log", "testdata/access_log.txt"}, args...)...)
		assert.Error(t, err, strings.Join(args, " "))
	}
}

func jsonUnmarshal(s string, v any) error { return json.Unmarshal([]byte(s), v) }
