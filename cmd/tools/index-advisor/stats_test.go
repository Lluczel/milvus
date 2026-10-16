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
	"os"
	"path/filepath"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestLoadStatsFixture(t *testing.T) {
	m := loadFixtureMetadata(t)
	s, err := loadStatsFile("testdata/stats.json", m)
	require.NoError(t, err)
	products, _ := m.lookup("default", "products")

	rows, city := s.lookup("default", "products", mustColumn(t, products, "city"))
	require.NotNil(t, rows)
	assert.Equal(t, int64(5000), *rows)
	require.NotNil(t, city.DistinctCount)
	assert.Equal(t, int64(5), *city.DistinctCount)

	// Keys are normalized: the file says "color", the predicate column is $meta["color"].
	_, color := s.lookup("default", "products", mustColumn(t, products, `$meta["color"]`))
	require.NotNil(t, color.DistinctCount)
	assert.Equal(t, int64(2), *color.DistinctCount)

	_, price := s.lookup("default", "products", mustColumn(t, products, "price"))
	assert.Nil(t, price.DistinctCount, "fields absent from the file are unknown")

	rows, _ = s.lookup("default", "idx_probe", mustColumn(t, products, "city"))
	assert.Nil(t, rows, "collections absent from the file are unknown")

	var none *statsSet
	rows, fs := none.lookup("default", "products", mustColumn(t, products, "city"))
	assert.Nil(t, rows)
	assert.Equal(t, fieldStats{}, fs)
}

func TestParseStatsErrors(t *testing.T) {
	m := loadFixtureMetadata(t)
	cases := map[string]string{
		"not json":       `[`,
		"version":        `{"format_version": 0}`,
		"unknown coll":   `{"format_version": 1, "collections": [{"name": "nope"}]}`,
		"duplicate coll": `{"format_version": 1, "collections": [{"name": "products"}, {"name": "products"}]}`,
		"negative rows":  `{"format_version": 1, "collections": [{"name": "products", "row_count": -1}]}`,
		"negative ndv":   `{"format_version": 1, "collections": [{"name": "products", "fields": {"city": {"distinct_count": -1}}}]}`,
		"null fraction":  `{"format_version": 1, "collections": [{"name": "products", "fields": {"city": {"null_fraction": 1.5}}}]}`,
		"bad field":      `{"format_version": 1, "collections": [{"name": "products", "fields": {"price >": {}}}]}`,
		"same column":    `{"format_version": 1, "collections": [{"name": "products", "fields": {"color": {}, "$meta[\"color\"]": {}}}]}`,
	}
	for name, data := range cases {
		_, err := parseStats([]byte(data), m)
		assert.Error(t, err, name)
	}
	_, err := parseStats([]byte(cases["same column"]), m)
	assert.ErrorContains(t, err, "more than once")

	_, err = loadStatsFile(filepath.Join(t.TempDir(), "missing.json"), m)
	assert.True(t, errors.Is(err, merr.ErrIoFailed))

	bad := filepath.Join(t.TempDir(), "bad.json")
	require.NoError(t, os.WriteFile(bad, []byte(cases["negative rows"]), 0o600))
	_, err = loadStatsFile(bad, m)
	assert.True(t, errors.Is(err, merr.ErrParameterInvalid))
}
