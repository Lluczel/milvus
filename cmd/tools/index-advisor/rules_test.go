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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

func TestKindOf(t *testing.T) {
	for _, c := range []struct {
		col  column
		want columnKind
	}{
		{column{dataType: schemapb.DataType_Bool}, kindBool},
		{column{dataType: schemapb.DataType_Int8}, kindInt},
		{column{dataType: schemapb.DataType_Float}, kindFloat},
		{column{dataType: schemapb.DataType_VarChar}, kindString},
		{column{dataType: schemapb.DataType_Timestamptz}, kindTimestamptz},
		{column{dataType: schemapb.DataType_Array, elementType: schemapb.DataType_Int64}, kindArray},
		{column{dataType: schemapb.DataType_JSON, jsonPath: []string{"a"}}, kindJSONPath},
		{column{dataType: schemapb.DataType_JSON}, kindOther},
		{column{dataType: schemapb.DataType_Geometry}, kindGeometry},
		{column{dataType: schemapb.DataType_FloatVector}, kindOther},
	} {
		assert.Equal(t, c.want, kindOf(c.col), c.col.dataType.String())
	}
}

func TestJSONCastType(t *testing.T) {
	for kind, want := range map[string]string{"bool": "BOOL", "int": "DOUBLE", "float": "DOUBLE", "string": "VARCHAR", "array": "", "": ""} {
		assert.Equal(t, want, jsonCastType(kind), kind)
	}
}

// TestIndexAccepts pins indexAccepts to the CheckValidDataType and JSON cast
// rules in internal/util/indexparamcheck, which this tool cannot import.
func TestIndexAccepts(t *testing.T) {
	var (
		boolCol   = column{dataType: schemapb.DataType_Bool}
		intCol    = column{dataType: schemapb.DataType_Int64}
		pkCol     = column{dataType: schemapb.DataType_Int64, isPrimaryKey: true}
		floatCol  = column{dataType: schemapb.DataType_Double}
		strCol    = column{dataType: schemapb.DataType_VarChar}
		tsCol     = column{dataType: schemapb.DataType_Timestamptz}
		strArray  = column{dataType: schemapb.DataType_Array, elementType: schemapb.DataType_VarChar}
		dblArray  = column{dataType: schemapb.DataType_Array, elementType: schemapb.DataType_Double}
		jsonPath  = column{dataType: schemapb.DataType_JSON, jsonPath: []string{"a"}}
		geomCol   = column{dataType: schemapb.DataType_Geometry}
		vectorCol = column{dataType: schemapb.DataType_FloatVector}
	)
	cases := []struct {
		index string
		col   column
		cast  string
		want  bool
	}{
		{indexInverted, boolCol, "", true},
		{indexInverted, floatCol, "", true},
		{indexInverted, strArray, "", true},
		{indexInverted, tsCol, "", false},
		{indexInverted, vectorCol, "", false},
		{indexInverted, jsonPath, "DOUBLE", true},
		{indexInverted, jsonPath, "ARRAY_VARCHAR", true},
		{indexInverted, jsonPath, "", false},

		{indexBitmap, intCol, "", true},
		{indexBitmap, pkCol, "", false},
		{indexBitmap, floatCol, "", false},
		{indexBitmap, strArray, "", true},
		{indexBitmap, dblArray, "", false},
		{indexBitmap, jsonPath, "VARCHAR", true},
		{indexBitmap, jsonPath, "DOUBLE", false},

		{indexSTLSort, floatCol, "", true},
		{indexSTLSort, strCol, "", true},
		{indexSTLSort, tsCol, "", true},
		{indexSTLSort, boolCol, "", false},
		{indexSTLSort, jsonPath, "DOUBLE", true},
		{indexSTLSort, jsonPath, "BOOL", false},

		{indexTrie, strCol, "", true},
		{indexTrie, intCol, "", false},

		{indexNgram, strCol, "", true},
		{indexNgram, jsonPath, "VARCHAR", true},
		{indexNgram, jsonPath, "DOUBLE", false},
		{indexNgram, intCol, "", false},

		{indexRTree, geomCol, "", true},
		{indexRTree, strCol, "", false},

		{"HYBRID", intCol, "", false},
	}
	for _, c := range cases {
		assert.Equal(t, c.want, indexAccepts(c.index, c.col, c.cast), "%s on %s cast=%q", c.index, c.col.dataType, c.cast)
	}
}

// TestSuitabilityRulesAreAccepted guards the rule table: a rule may only
// suggest an index type Milvus accepts on every column kind it lists.
func TestSuitabilityRulesAreAccepted(t *testing.T) {
	sample := map[columnKind]column{
		kindBool:        {dataType: schemapb.DataType_Bool},
		kindInt:         {dataType: schemapb.DataType_Int64},
		kindFloat:       {dataType: schemapb.DataType_Double},
		kindString:      {dataType: schemapb.DataType_VarChar},
		kindTimestamptz: {dataType: schemapb.DataType_Timestamptz},
		kindArray:       {dataType: schemapb.DataType_Array, elementType: schemapb.DataType_VarChar},
		kindJSONPath:    {dataType: schemapb.DataType_JSON, jsonPath: []string{"a"}},
		kindGeometry:    {dataType: schemapb.DataType_Geometry},
	}
	for i, rule := range suitabilityRules {
		assert.NotEmpty(t, rule.source, "rule %d needs a source", i)
		for _, kind := range rule.kinds {
			col, ok := sample[kind]
			if !assert.True(t, ok, "no sample column for %s", kind) {
				continue
			}
			accepted := indexAccepts(rule.indexType, col, "")
			if kind == kindJSONPath {
				accepted = indexAccepts(rule.indexType, col, "VARCHAR") || indexAccepts(rule.indexType, col, "DOUBLE")
			}
			assert.True(t, accepted, fmt.Sprintf("rule %d suggests %s on %s, which Milvus rejects", i, rule.indexType, kind))
		}
	}
}
