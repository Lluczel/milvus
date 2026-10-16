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
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func runCLI(args ...string) (string, string, error) {
	var out, errOut bytes.Buffer
	err := run(context.Background(), args, &out, &errOut)
	return out.String(), errOut.String(), err
}

func TestRunCommands(t *testing.T) {
	_, stderr, err := runCLI()
	assert.True(t, errors.Is(err, merr.ErrParameterMissing))
	assert.Contains(t, stderr, "Usage:")

	stdout, _, err := runCLI("help")
	assert.NoError(t, err)
	assert.Contains(t, stdout, "export-metadata")

	_, _, err = runCLI("frobnicate")
	assert.True(t, errors.Is(err, merr.ErrParameterInvalid))
}

func TestRunExportMetadataFlags(t *testing.T) {
	_, stderr, err := runCLI("export-metadata", "-h")
	assert.NoError(t, err, "-h prints flags without failing")
	assert.Contains(t, stderr, "-collection")

	_, _, err = runCLI("export-metadata", "--no-such-flag")
	assert.True(t, errors.Is(err, merr.ErrParameterInvalid))

	_, _, err = runCLI("export-metadata", "stray")
	assert.True(t, errors.Is(err, merr.ErrParameterInvalid))

	// Nothing listens on port 1, so connecting fails within the timeout.
	_, _, err = runCLI("export-metadata", "--uri", "127.0.0.1:1", "--timeout", "300ms")
	assert.Error(t, err)
}
