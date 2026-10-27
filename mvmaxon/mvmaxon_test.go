// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mvmaxon

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInterpreter checks that Goal code can use the axon package
// without importing it, and that math mode works without tensorfs.
func TestInterpreter(t *testing.T) {
	var out bytes.Buffer
	in, err := NewInterpreter(&out)
	require.NoError(t, err)
	_, err = in.Eval("fmt.Println(axon.NeuronTraceCycles == 10, axon.SuperLayer)")
	require.NoError(t, err)
	_, err = in.Eval("# x := linspace(0, 1, 3, true)\nfmt.Println(x)")
	require.NoError(t, err)
	assert.Equal(t, "true SuperLayer\n[3]   0 0.5   1 \n\n", out.String())
}
