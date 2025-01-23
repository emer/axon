// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"os"
	"testing"

	"cogentcore.org/core/paint"
	"github.com/emer/axon/v2/kinase"
)

func TestMain(m *testing.M) {
	paint.FontLibrary.InitFontPaths(paint.FontPaths...)
	os.Exit(m.Run())
}

func TestLinear(t *testing.T) {
	var ls Linear
	ls.Defaults()
	ls.SynCaBin.Envelope = kinase.Env10
	ls.Cycles = 200
	ls.PlusCycles = 50
	ls.CyclesPerBin = 10 // now always 10
	ls.Update()
	ls.Init()
	ls.Run()
	ls.Regress()
}
