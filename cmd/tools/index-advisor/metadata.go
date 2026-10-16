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
	"os"
	"sort"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	metadataFormatVersion = 1
	defaultDatabase       = "default"
)

// metadataFile is the --metadata input, written by export-metadata.
type metadataFile struct {
	FormatVersion int                  `json:"format_version"`
	Collections   []collectionMetadata `json:"collections"`
}

type collectionMetadata struct {
	DB      string          `json:"db"`
	Name    string          `json:"name"`
	Schema  json.RawMessage `json:"schema"` // protojson of schemapb.CollectionSchema
	Indexes []indexMetadata `json:"indexes"`
}

// indexMetadata is one index as DescribeIndex reports it.
type indexMetadata struct {
	IndexName string            `json:"index_name"`
	FieldName string            `json:"field_name"`
	IndexType string            `json:"index_type"`
	Params    map[string]string `json:"params,omitempty"`
}

// collectionInfo is a loaded collection: its schema, ready for the expression
// parser, and its existing indexes keyed by the column they cover.
type collectionInfo struct {
	db, name string
	helper   *typeutil.SchemaHelper
	indexes  map[string][]indexMetadata // column.String() -> indexes on it
}

// indexesOn returns the existing indexes on col, sorted by name.
func (c *collectionInfo) indexesOn(col column) []indexMetadata {
	return c.indexes[col.String()]
}

type metadataSet struct {
	collections map[string]*collectionInfo // collectionKey -> info
}

func collectionKey(db, name string) string {
	if db == "" {
		db = defaultDatabase
	}
	return db + "." + name
}

func (m *metadataSet) lookup(db, name string) (*collectionInfo, bool) {
	c, ok := m.collections[collectionKey(db, name)]
	return c, ok
}

func loadMetadataFile(path string) (*metadataSet, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- path is a CLI argument
	if err != nil {
		return nil, merr.WrapErrIoFailed(path, err)
	}
	m, err := parseMetadata(data)
	if err != nil {
		return nil, merr.Wrapf(err, "metadata file %s", path)
	}
	return m, nil
}

func parseMetadata(data []byte) (*metadataSet, error) {
	var file metadataFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, merr.WrapErrParameterInvalidMsg("invalid metadata JSON: %s", err.Error())
	}
	if file.FormatVersion != metadataFormatVersion {
		return nil, merr.WrapErrParameterInvalidMsg("unsupported metadata format_version %d, want %d", file.FormatVersion, metadataFormatVersion)
	}

	set := &metadataSet{collections: make(map[string]*collectionInfo, len(file.Collections))}
	for _, c := range file.Collections {
		key := collectionKey(c.DB, c.Name)
		if _, dup := set.collections[key]; dup {
			return nil, merr.WrapErrParameterInvalidMsg("collection %s listed twice", key)
		}
		info, err := loadCollection(c)
		if err != nil {
			return nil, merr.Wrapf(err, "collection %s", key)
		}
		set.collections[key] = info
	}
	return set, nil
}

func loadCollection(c collectionMetadata) (*collectionInfo, error) {
	schema := &schemapb.CollectionSchema{}
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(c.Schema, schema); err != nil {
		return nil, merr.WrapErrParameterInvalidMsg("invalid schema: %s", err.Error())
	}
	restoreDynamicField(schema)
	helper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		return nil, merr.Wrap(err, "invalid schema")
	}

	info := &collectionInfo{db: c.DB, name: c.Name, helper: helper, indexes: map[string][]indexMetadata{}}
	if info.db == "" {
		info.db = defaultDatabase
	}
	for _, idx := range c.Indexes {
		col, err := indexColumn(helper, idx)
		if err != nil {
			return nil, merr.Wrapf(err, "index %s", idx.IndexName)
		}
		info.indexes[col.String()] = append(info.indexes[col.String()], idx)
	}
	for _, list := range info.indexes {
		sort.Slice(list, func(i, j int) bool { return list[i].IndexName < list[j].IndexName })
	}
	return info, nil
}

// restoreDynamicField adds the $meta field back to a schema with dynamic fields
// enabled. DescribeCollection hides it (internal/proxy/ddl/describe_util.go),
// but the expression parser needs it to resolve dynamic keys such as color.
// The FieldID only has to be unique within this tool.
func restoreDynamicField(schema *schemapb.CollectionSchema) {
	if !schema.GetEnableDynamicField() {
		return
	}
	maxID := int64(common.StartOfUserFieldID)
	for _, f := range schema.GetFields() {
		if f.GetIsDynamic() {
			return
		}
		maxID = max(maxID, f.GetFieldID())
	}
	for _, s := range schema.GetStructArrayFields() {
		maxID = max(maxID, s.GetFieldID())
		for _, f := range s.GetFields() {
			maxID = max(maxID, f.GetFieldID())
		}
	}
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
		FieldID:   maxID + 1,
		Name:      common.MetaFieldName,
		DataType:  schemapb.DataType_JSON,
		IsDynamic: true,
	})
}

// resolveIdentifier parses a field name or JSON path (price, meta["brand"],
// color) with Milvus's identifier parser into the column it names.
func resolveIdentifier(helper *typeutil.SchemaHelper, identifier string) (column, error) {
	var info *planpb.ColumnInfo
	err := planparserv2.ParseIdentifier(helper, identifier, func(e *planpb.Expr) error {
		info = e.GetColumnExpr().GetInfo()
		return nil
	})
	if err != nil {
		return column{}, err
	}
	return (&exprWalker{helper: helper}).column(info), nil
}

// indexColumn resolves the column an index covers, so it can be compared with
// predicate columns. DescribeIndex reports a JSON-path index as field "meta"
// with json_path `meta["brand"]`, and a dynamic-field index as field "color"
// (not a schema field) with no json_path. Parsing json_path, else the field
// name, handles both and plain fields alike.
func indexColumn(helper *typeutil.SchemaHelper, idx indexMetadata) (column, error) {
	identifier := idx.FieldName
	if path := idx.Params[common.JSONPathKey]; path != "" {
		identifier = path
	}
	return resolveIdentifier(helper, identifier)
}
