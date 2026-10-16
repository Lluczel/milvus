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

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func loadFixtureMetadata(t *testing.T) *metadataSet {
	m, err := loadMetadataFile("testdata/metadata.json")
	require.NoError(t, err)
	return m
}

func mustColumn(t *testing.T, info *collectionInfo, identifier string) column {
	col, err := resolveIdentifier(info.helper, identifier)
	require.NoError(t, err, identifier)
	return col
}

func indexNames(list []indexMetadata) []string {
	var names []string
	for _, idx := range list {
		names = append(names, idx.IndexName+":"+idx.IndexType)
	}
	return names
}

// The fixture is a real export-metadata run against Milvus v3.0.3.
func TestLoadMetadataFixture(t *testing.T) {
	m := loadFixtureMetadata(t)

	products, ok := m.lookup("", "products")
	require.True(t, ok, "an empty database name means default")
	assert.Equal(t, "default", products.db)
	assert.Equal(t, []string{"city:INVERTED"}, indexNames(products.indexesOn(mustColumn(t, products, "city"))))
	assert.Empty(t, products.indexesOn(mustColumn(t, products, "price")))

	// DescribeCollection hides $meta; loading restores it so dynamic keys resolve.
	color := mustColumn(t, products, "color")
	assert.True(t, color.isDynamic)
	assert.Equal(t, `$meta["color"]`, color.String())

	// Each way DescribeIndex reports an index resolves to the column it covers.
	probe, ok := m.lookup("default", "idx_probe")
	require.True(t, ok)
	assert.Equal(t, []string{"dyn_color:INVERTED"}, indexNames(probe.indexesOn(mustColumn(t, probe, "color"))))
	assert.Equal(t, []string{"meta_brand:INVERTED"}, indexNames(probe.indexesOn(mustColumn(t, probe, `meta["brand"]`))))
	assert.Empty(t, probe.indexesOn(mustColumn(t, probe, `meta["other"]`)))
	assert.Equal(t, []string{"name_ngram:NGRAM"}, indexNames(probe.indexesOn(mustColumn(t, probe, "name"))))
	assert.Equal(t, map[string]string{"min_gram": "2", "max_gram": "3"}, probe.indexesOn(mustColumn(t, probe, "name"))[0].Params)
	assert.Equal(t, []string{"price_sort:STL_SORT"}, indexNames(probe.indexesOn(mustColumn(t, probe, "price"))))

	_, ok = m.lookup("other_db", "products")
	assert.False(t, ok)
}

func TestRestoreDynamicField(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		EnableDynamicField: true,
		Fields:             []*schemapb.FieldSchema{{FieldID: 100, Name: "id"}, {FieldID: 104, Name: "v"}},
		StructArrayFields:  []*schemapb.StructArrayFieldSchema{{FieldID: 105, Fields: []*schemapb.FieldSchema{{FieldID: 107}}}},
	}
	restoreDynamicField(schema)
	last := schema.Fields[len(schema.Fields)-1]
	assert.Equal(t, common.MetaFieldName, last.GetName())
	assert.True(t, last.GetIsDynamic())
	assert.Equal(t, int64(108), last.GetFieldID(), "the new ID must not collide with struct sub-fields")

	restoreDynamicField(schema)
	assert.Len(t, schema.Fields, 3, "an existing dynamic field is kept as is")

	static := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "id"}}}
	restoreDynamicField(static)
	assert.Len(t, static.Fields, 1)
}

func TestParseMetadataErrors(t *testing.T) {
	const schema = `{"name":"c","fields":[{"fieldID":"100","name":"id","is_primary_key":true,"data_type":"Int64"}]}`
	cases := map[string]string{
		"not json":        `{`,
		"version":         `{"format_version": 2, "collections": []}`,
		"duplicate":       `{"format_version": 1, "collections": [{"name": "c", "schema": ` + schema + `}, {"db": "default", "name": "c", "schema": ` + schema + `}]}`,
		"bad schema":      `{"format_version": 1, "collections": [{"name": "c", "schema": {"fields": 3}}]}`,
		"duplicate field": `{"format_version": 1, "collections": [{"name": "c", "schema": {"fields": [{"fieldID": "100", "name": "id"}, {"fieldID": "101", "name": "id"}]}}]}`,
		"unknown index":   `{"format_version": 1, "collections": [{"name": "c", "schema": ` + schema + `, "indexes": [{"index_name": "i", "field_name": "nope"}]}]}`,
	}
	for name, data := range cases {
		_, err := parseMetadata([]byte(data))
		assert.Error(t, err, name)
	}
	_, err := parseMetadata([]byte(cases["version"]))
	assert.True(t, errors.Is(err, merr.ErrParameterInvalid))

	_, err = loadMetadataFile(filepath.Join(t.TempDir(), "missing.json"))
	assert.True(t, errors.Is(err, merr.ErrIoFailed))

	bad := filepath.Join(t.TempDir(), "bad.json")
	require.NoError(t, os.WriteFile(bad, []byte(cases["duplicate"]), 0o600))
	_, err = loadMetadataFile(bad)
	assert.ErrorContains(t, err, "listed twice")
}
