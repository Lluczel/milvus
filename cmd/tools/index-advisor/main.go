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

// Command index-advisor recommends scalar indexes from the filter expressions
// in Milvus proxy access logs. It is advisory only: it reads logs and metadata
// and never creates, alters or drops an index. See README.md.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const usage = `Usage:
  index-advisor advise --log FILE [--log FILE ...] --metadata FILE [--stats FILE] [--milvus-config FILE]
  index-advisor export-metadata [--uri HOST:PORT] [--db NAME] [--collection NAME ...] [--token TOKEN]

Run "index-advisor <command> -h" for the flags of a command.
`

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "index-advisor: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return merr.WrapErrParameterMissingMsg("a command is required")
	}
	switch args[0] {
	case "advise":
		return runAdvise(args[1:], stdout, stderr)
	case "export-metadata":
		return runExportMetadata(ctx, args[1:], stdout, stderr)
	case "-h", "-help", "--help", "help":
		fmt.Fprint(stdout, usage)
		return nil
	default:
		fmt.Fprint(stderr, usage)
		return merr.WrapErrParameterInvalidMsg("unknown command %q", args[0])
	}
}

// stringList is a repeatable string flag.
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// parseFlags parses args and reports whether the command should run; -h
// prints the flags and stops without an error.
func parseFlags(fs *flag.FlagSet, args []string) (bool, error) {
	err := fs.Parse(args)
	switch {
	case errors.Is(err, flag.ErrHelp):
		return false, nil
	case err != nil:
		return false, merr.WrapErrParameterInvalidMsg("%s: %s", fs.Name(), err.Error())
	case fs.NArg() > 0:
		return false, merr.WrapErrParameterInvalidMsg("%s: unexpected argument %q", fs.Name(), fs.Arg(0))
	}
	return true, nil
}

func runAdvise(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("advise", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var logs stringList
	fs.Var(&logs, "log", "access log file; repeatable, e.g. for rotated files (required)")
	metadataPath := fs.String("metadata", "", "metadata JSON written by export-metadata (required)")
	statsPath := fs.String("stats", "", "optional field statistics JSON; missing statistics are reported as unknown")
	configPath := fs.String("milvus-config", "", "optional milvus.yaml or user.yaml whose proxy.accessLog.formatters override the defaults")
	if ok, err := parseFlags(fs, args); !ok {
		return err
	}
	if len(logs) == 0 || *metadataPath == "" {
		fs.Usage()
		return merr.WrapErrParameterMissingMsg("advise needs --log and --metadata")
	}

	cfgs, err := loadFormatterConfigs(*configPath)
	if err != nil {
		return err
	}
	formatters, err := newFormatterSet(cfgs)
	if err != nil {
		return err
	}
	meta, err := loadMetadataFile(*metadataPath)
	if err != nil {
		return err
	}
	var stats *statsSet
	if *statsPath != "" {
		if stats, err = loadStatsFile(*statsPath, meta); err != nil {
			return err
		}
	}

	skipped := skipCounts{}
	var records []accessRecord
	for _, path := range logs {
		got, err := readRecordsFile(path, formatters, skipped)
		if err != nil {
			return err
		}
		records = append(records, got...)
	}
	return writeJSON(stdout, advise(logs, records, skipped, meta, stats))
}

func runExportMetadata(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("export-metadata", flag.ContinueOnError)
	fs.SetOutput(stderr)
	uri := fs.String("uri", "localhost:19530", "Milvus proxy address")
	db := fs.String("db", defaultDatabase, "database to export")
	token := fs.String("token", "", `"user:password" or API key, when authentication is enabled`)
	timeout := fs.Duration("timeout", 30*time.Second, "timeout for the whole export")
	var collections stringList
	fs.Var(&collections, "collection", "collection to export; repeatable (default: every collection in --db)")
	if ok, err := parseFlags(fs, args); !ok {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: *uri, DBName: *db, APIKey: *token})
	if err != nil {
		return merr.Wrapf(err, "connect to %s", *uri)
	}
	defer cli.Close(ctx)

	file, err := exportMetadata(ctx, cli.GetService(), *db, collections)
	if err != nil {
		return err
	}
	return writeJSON(stdout, file)
}

// writeJSON writes v as indented JSON. Field order follows the struct
// definitions and every slice is sorted by its producer, so output is stable.
func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return merr.WrapErrIoFailedReason(err.Error(), "write output")
	}
	return nil
}
