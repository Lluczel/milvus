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
	"encoding/json"
	"sort"

	"github.com/cockroachdb/errors"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// metadataService is the subset of milvuspb.MilvusServiceClient that
// export-metadata calls. All three are read-only: the tool never creates,
// alters or drops a collection or index.
type metadataService interface {
	ShowCollections(ctx context.Context, in *milvuspb.ShowCollectionsRequest, opts ...grpc.CallOption) (*milvuspb.ShowCollectionsResponse, error)
	DescribeCollection(ctx context.Context, in *milvuspb.DescribeCollectionRequest, opts ...grpc.CallOption) (*milvuspb.DescribeCollectionResponse, error)
	DescribeIndex(ctx context.Context, in *milvuspb.DescribeIndexRequest, opts ...grpc.CallOption) (*milvuspb.DescribeIndexResponse, error)
}

// exportMetadata reads the schema and indexes of the named collections in db,
// or of every collection in db when names is empty. It uses the raw service
// responses rather than client/entity types so the schema is exported exactly
// as Milvus describes it, and DescribeIndex reports each index's field name.
func exportMetadata(ctx context.Context, svc metadataService, db string, names []string) (*metadataFile, error) {
	if len(names) == 0 {
		resp, err := svc.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{DbName: db})
		if err := merr.CheckRPCCall(resp, err); err != nil {
			return nil, merr.Wrapf(err, "list collections in database %s", db)
		}
		names = resp.GetCollectionNames()
	}
	names = append([]string(nil), names...)
	sort.Strings(names)

	file := &metadataFile{FormatVersion: metadataFormatVersion, Collections: []collectionMetadata{}}
	for _, name := range names {
		c, err := exportCollection(ctx, svc, db, name)
		if err != nil {
			return nil, merr.Wrapf(err, "collection %s.%s", db, name)
		}
		file.Collections = append(file.Collections, c)
	}
	return file, nil
}

func exportCollection(ctx context.Context, svc metadataService, db, name string) (collectionMetadata, error) {
	desc, err := svc.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{DbName: db, CollectionName: name})
	if err := merr.CheckRPCCall(desc, err); err != nil {
		return collectionMetadata{}, merr.Wrap(err, "describe collection")
	}
	schema, err := protojson.MarshalOptions{UseProtoNames: true}.Marshal(desc.GetSchema())
	if err != nil {
		return collectionMetadata{}, merr.WrapErrServiceInternalMsg("marshal schema: %s", err.Error())
	}
	// protojson randomizes whitespace between runs; compact it so exports diff cleanly.
	var compact bytes.Buffer
	if err := json.Compact(&compact, schema); err != nil {
		return collectionMetadata{}, merr.WrapErrServiceInternalMsg("compact schema: %s", err.Error())
	}

	c := collectionMetadata{DB: db, Name: name, Schema: compact.Bytes(), Indexes: []indexMetadata{}}
	indexes, err := svc.DescribeIndex(ctx, &milvuspb.DescribeIndexRequest{DbName: db, CollectionName: name})
	err = merr.CheckRPCCall(indexes, err)
	switch {
	case errors.Is(err, merr.ErrIndexNotFound):
		return c, nil // a collection with no index yet
	case err != nil:
		return collectionMetadata{}, merr.Wrap(err, "describe indexes")
	}
	for _, d := range indexes.GetIndexDescriptions() {
		params := make(map[string]string, len(d.GetParams()))
		for _, kv := range d.GetParams() {
			params[kv.GetKey()] = kv.GetValue()
		}
		idx := indexMetadata{IndexName: d.GetIndexName(), FieldName: d.GetFieldName(), IndexType: params[common.IndexTypeKey]}
		delete(params, common.IndexTypeKey)
		if len(params) > 0 {
			idx.Params = params
		}
		c.Indexes = append(c.Indexes, idx)
	}
	sort.Slice(c.Indexes, func(i, j int) bool { return c.Indexes[i].IndexName < c.Indexes[j].IndexName })
	return c, nil
}
