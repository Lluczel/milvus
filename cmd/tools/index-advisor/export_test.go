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
	"context"
	"os"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// fakeService serves canned read-only responses keyed by collection name.
type fakeService struct {
	collections []string
	schemas     map[string]*schemapb.CollectionSchema
	indexes     map[string][]*milvuspb.IndexDescription
	showErr     error
	describeErr error
	indexErr    error
	calls       []string
}

func (f *fakeService) ShowCollections(_ context.Context, in *milvuspb.ShowCollectionsRequest, _ ...grpc.CallOption) (*milvuspb.ShowCollectionsResponse, error) {
	f.calls = append(f.calls, "ShowCollections "+in.GetDbName())
	if f.showErr != nil {
		return &milvuspb.ShowCollectionsResponse{Status: merr.Status(f.showErr)}, nil
	}
	return &milvuspb.ShowCollectionsResponse{Status: merr.Success(), CollectionNames: f.collections}, nil
}

func (f *fakeService) DescribeCollection(_ context.Context, in *milvuspb.DescribeCollectionRequest, _ ...grpc.CallOption) (*milvuspb.DescribeCollectionResponse, error) {
	f.calls = append(f.calls, "DescribeCollection "+in.GetDbName()+"."+in.GetCollectionName())
	if f.describeErr != nil {
		return nil, f.describeErr
	}
	return &milvuspb.DescribeCollectionResponse{Status: merr.Success(), Schema: f.schemas[in.GetCollectionName()]}, nil
}

func (f *fakeService) DescribeIndex(_ context.Context, in *milvuspb.DescribeIndexRequest, _ ...grpc.CallOption) (*milvuspb.DescribeIndexResponse, error) {
	f.calls = append(f.calls, "DescribeIndex "+in.GetDbName()+"."+in.GetCollectionName())
	if f.indexErr != nil {
		return &milvuspb.DescribeIndexResponse{Status: merr.Status(f.indexErr)}, nil
	}
	descs, ok := f.indexes[in.GetCollectionName()]
	if !ok {
		return &milvuspb.DescribeIndexResponse{Status: merr.Status(merr.WrapErrIndexNotFound(""))}, nil
	}
	return &milvuspb.DescribeIndexResponse{Status: merr.Success(), IndexDescriptions: descs}, nil
}

func newFakeService() *fakeService {
	schema := func(name string) *schemapb.CollectionSchema {
		return &schemapb.CollectionSchema{Name: name, EnableDynamicField: true, Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "meta", DataType: schemapb.DataType_JSON},
		}}
	}
	return &fakeService{
		collections: []string{"b", "a"},
		schemas:     map[string]*schemapb.CollectionSchema{"a": schema("a"), "b": schema("b")},
		indexes: map[string][]*milvuspb.IndexDescription{
			"a": {
				{IndexName: "z_dyn", FieldName: "color", Params: []*commonpb.KeyValuePair{{Key: "index_type", Value: "INVERTED"}, {Key: "json_cast_type", Value: "varchar"}}},
				{IndexName: "a_pk", FieldName: "id", Params: []*commonpb.KeyValuePair{{Key: "index_type", Value: "STL_SORT"}}},
			},
		},
	}
}

func TestExportMetadata(t *testing.T) {
	svc := newFakeService()
	file, err := exportMetadata(context.Background(), svc, "db1", nil)
	require.NoError(t, err)

	require.Len(t, file.Collections, 2)
	assert.Equal(t, "a", file.Collections[0].Name, "collections are sorted")
	assert.Equal(t, []indexMetadata{
		{IndexName: "a_pk", FieldName: "id", IndexType: "STL_SORT"},
		{IndexName: "z_dyn", FieldName: "color", IndexType: "INVERTED", Params: map[string]string{"json_cast_type": "varchar"}},
	}, file.Collections[0].Indexes)
	assert.Empty(t, file.Collections[1].Indexes, "IndexNotFound means no index yet")
	assert.Equal(t, []string{
		"ShowCollections db1",
		"DescribeCollection db1.a", "DescribeIndex db1.a",
		"DescribeCollection db1.b", "DescribeIndex db1.b",
	}, svc.calls, "only read-only RPCs are issued")

	// The export round-trips through the advise input format.
	var buf bytes.Buffer
	require.NoError(t, writeJSON(&buf, file))
	m, err := parseMetadata(buf.Bytes())
	require.NoError(t, err)
	a, ok := m.lookup("db1", "a")
	require.True(t, ok)
	assert.Len(t, a.indexesOn(mustColumn(t, a, "color")), 1)

	// Explicit collections skip ShowCollections.
	svc = newFakeService()
	file, err = exportMetadata(context.Background(), svc, "db1", []string{"b"})
	require.NoError(t, err)
	require.Len(t, file.Collections, 1)
	assert.NotContains(t, svc.calls, "ShowCollections db1")
}

func TestExportMetadataErrors(t *testing.T) {
	for name, svc := range map[string]*fakeService{
		"show":     {showErr: merr.WrapErrServiceNotReady("proxy", 0, "starting")},
		"describe": {collections: []string{"a"}, describeErr: errors.New("connection reset")},
		"index":    {collections: []string{"a"}, schemas: map[string]*schemapb.CollectionSchema{"a": {Name: "a"}}, indexErr: merr.WrapErrServiceNotReady("proxy", 0, "starting")},
	} {
		_, err := exportMetadata(context.Background(), svc, "default", nil)
		assert.Error(t, err, name)
	}
}

// TestExportMetadataLive runs against a real Milvus when INDEX_ADVISOR_MILVUS_URI
// is set (for example localhost:19530 with the README's sample workload loaded).
func TestExportMetadataLive(t *testing.T) {
	uri := os.Getenv("INDEX_ADVISOR_MILVUS_URI")
	if uri == "" {
		t.Skip("INDEX_ADVISOR_MILVUS_URI not set")
	}
	var out, errOut bytes.Buffer
	require.NoError(t, run(context.Background(), []string{"export-metadata", "--uri", uri}, &out, &errOut), errOut.String())
	_, err := parseMetadata(out.Bytes())
	require.NoError(t, err)
}
