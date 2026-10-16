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
	"bufio"
	"io"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/milvus-io/milvus/internal/proxy/accesslog/info"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Reasons a line or record is left out of the analysis. Every dropped input is
// counted under one of these, so the report accounts for the whole log.
const (
	skipOrphanLine      = "orphan_line"      // text before the first record header
	skipUnmatchedFormat = "unmatched_format" // record matches no configured formatter
)

// recordStart matches the "[$time_now] [ACCESS] " prefix every default formatter
// begins with ($time_now uses info's "2006/01/02 15:04:05.000 -07:00" layout). An
// expression may contain newlines, which Milvus writes unescaped, so any other
// line continues the current record.
var recordStart = regexp.MustCompile(`^\[\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3} [+-]\d{2}:\d{2}\] \[ACCESS\] `)

// accessRecord is one parsed access-log record.
type accessRecord struct {
	file      string
	line      int // 1-based line number of the record header
	formatter string
	values    map[string]string // placeholder -> written value
}

func (r accessRecord) method() string     { return r.values[fieldMethodName] }
func (r accessRecord) status() string     { return r.values[fieldStatus] }
func (r accessRecord) database() string   { return r.values[fieldDatabase] }
func (r accessRecord) collection() string { return r.values[fieldCollection] }

// expr returns the logged filter expression and whether the record's formatter
// logs one at all. An empty expression with ok=true means "no filter".
func (r accessRecord) expr() (string, bool) {
	v, ok := r.values[fieldExpr]
	return v, ok
}

// timeCost returns the request latency, or false when it was not logged or
// Milvus wrote a sentinel because the request never finished.
func (r accessRecord) timeCost() (time.Duration, bool) {
	return parseTimeCost(r.values[fieldTimeCost])
}

// parseTimeCost parses a $time_cost value, which Milvus writes with
// fmt.Sprint(time.Duration) (e.g. "1.2ms", "850µs") or info.Unknown.
func parseTimeCost(v string) (time.Duration, bool) {
	if v == "" || v == info.Unknown || v == info.NotAny {
		return 0, false
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, false
	}
	return d, true
}

// skipCounts counts dropped inputs by reason.
type skipCounts map[string]int

// readRecordsFile parses one access-log file. See readRecords.
func readRecordsFile(path string, formatters *formatterSet, skipped skipCounts) ([]accessRecord, error) {
	f, err := os.Open(path) // #nosec G304 -- path is a CLI argument
	if err != nil {
		return nil, merr.WrapErrIoFailed(path, err)
	}
	defer f.Close()
	return readRecords(f, path, formatters, skipped)
}

// readRecords splits r into records and parses each with its formatter.
// Unparseable input is counted in skipped rather than returned as an error, so
// one odd line does not hide the rest of the log.
func readRecords(r io.Reader, file string, formatters *formatterSet, skipped skipCounts) ([]accessRecord, error) {
	var (
		records []accessRecord
		current strings.Builder
		start   int // header line of current; 0 when no record is open
	)
	flush := func() {
		if start == 0 {
			return
		}
		name, values, ok := formatters.match(current.String())
		if ok {
			records = append(records, accessRecord{file: file, line: start, formatter: name, values: values})
		} else {
			skipped[skipUnmatchedFormat]++
		}
		current.Reset()
		start = 0
	}

	// bufio.Reader rather than bufio.Scanner: an IN list can make one line
	// longer than any fixed token limit.
	reader := bufio.NewReader(r)
	for lineNo := 1; ; lineNo++ {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, merr.WrapErrIoFailed(file, err)
		}
		if line == "" && err == io.EOF {
			break
		}
		line = strings.TrimSuffix(line, "\n")

		switch {
		case recordStart.MatchString(line):
			flush()
			start = lineNo
			current.WriteString(line)
		case start == 0:
			skipped[skipOrphanLine]++
		default:
			current.WriteByte('\n')
			current.WriteString(line)
		}
		if err == io.EOF {
			break
		}
	}
	flush()
	return records, nil
}
