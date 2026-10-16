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

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const statsFormatVersion = 1

// statsFile is the optional --stats input: field statistics supplied by an
// operator, since Milvus does not expose distinct counts. Anything absent is
// reported as "unknown" rather than guessed.
type statsFile struct {
	FormatVersion int               `json:"format_version"`
	Collections   []collectionStats `json:"collections"`
}

type collectionStats struct {
	DB       string                `json:"db"`
	Name     string                `json:"name"`
	RowCount *int64                `json:"row_count,omitempty"`
	Fields   map[string]fieldStats `json:"fields"` // field name or JSON path -> stats
}

type fieldStats struct {
	DistinctCount *int64   `json:"distinct_count,omitempty"`
	NullFraction  *float64 `json:"null_fraction,omitempty"`
}

// statsSet holds statistics keyed by collection and normalized column, so a
// dynamic key written as color or $meta["color"] matches the same predicate.
type statsSet struct {
	rowCount map[string]*int64
	fields   map[string]map[string]fieldStats
}

// lookup returns the row count and field statistics for col; nil pointers mean
// unknown. A nil set has no statistics at all.
func (s *statsSet) lookup(db, name string, col column) (*int64, fieldStats) {
	if s == nil {
		return nil, fieldStats{}
	}
	key := collectionKey(db, name)
	return s.rowCount[key], s.fields[key][col.String()]
}

func loadStatsFile(path string, meta *metadataSet) (*statsSet, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- path is a CLI argument
	if err != nil {
		return nil, merr.WrapErrIoFailed(path, err)
	}
	s, err := parseStats(data, meta)
	if err != nil {
		return nil, merr.Wrapf(err, "stats file %s", path)
	}
	return s, nil
}

func parseStats(data []byte, meta *metadataSet) (*statsSet, error) {
	var file statsFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, merr.WrapErrParameterInvalidMsg("invalid stats JSON: %s", err.Error())
	}
	if file.FormatVersion != statsFormatVersion {
		return nil, merr.WrapErrParameterInvalidMsg("unsupported stats format_version %d, want %d", file.FormatVersion, statsFormatVersion)
	}

	set := &statsSet{rowCount: map[string]*int64{}, fields: map[string]map[string]fieldStats{}}
	for _, c := range file.Collections {
		key := collectionKey(c.DB, c.Name)
		info, ok := meta.lookup(c.DB, c.Name)
		if !ok {
			return nil, merr.WrapErrParameterInvalidMsg("stats for collection %s, which is not in the metadata", key)
		}
		if _, dup := set.fields[key]; dup {
			return nil, merr.WrapErrParameterInvalidMsg("collection %s listed twice", key)
		}
		if c.RowCount != nil && *c.RowCount < 0 {
			return nil, merr.WrapErrParameterInvalidMsg("collection %s: row_count must not be negative", key)
		}
		set.rowCount[key] = c.RowCount
		set.fields[key] = map[string]fieldStats{}
		for name, fs := range c.Fields {
			if fs.DistinctCount != nil && *fs.DistinctCount < 0 {
				return nil, merr.WrapErrParameterInvalidMsg("collection %s field %s: distinct_count must not be negative", key, name)
			}
			if fs.NullFraction != nil && (*fs.NullFraction < 0 || *fs.NullFraction > 1) {
				return nil, merr.WrapErrParameterInvalidMsg("collection %s field %s: null_fraction must be within [0, 1]", key, name)
			}
			col, err := resolveIdentifier(info.helper, name)
			if err != nil {
				return nil, merr.Wrapf(err, "collection %s field %s", key, name)
			}
			if _, dup := set.fields[key][col.String()]; dup {
				return nil, merr.WrapErrParameterInvalidMsg("collection %s: field %s is listed more than once", key, col)
			}
			set.fields[key][col.String()] = fs
		}
	}
	return set, nil
}
