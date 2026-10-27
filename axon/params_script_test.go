// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptSim is a sim for a params script, which uses it as sim.Sim.
type scriptSim struct {
	Params Params
}

// TestParamsScript checks that a params script, run by the interpreter,
// adds Layer and Path sheets whose Set functions are callable from Go.
func TestParamsScript(t *testing.T) {
	sim := &scriptSim{}
	sim.Params.Config(LayerSheets{}, PathSheets{}, "Script", "", reflect.ValueOf(sim))
	_, err := sim.Params.Interp.Eval("params", `sim.Sim.Params.Layer["Script"] = &axon.LayerSheet{
	&axon.LayerSel{Sel: "Layer", Set: func(ly *axon.LayerParams) {
		ly.Acts.Dend.GExp = 0.123
	}},
}
sim.Sim.Params.Path["Script"] = &axon.PathSheet{
	&axon.PathSel{Sel: "Path", Set: func(pt *axon.PathParams) {
		pt.Learn.LRate.Base = 0.042
	}},
}
`)
	require.NoError(t, err)

	lsheet, err := sim.Params.Layer.SheetByName("Script")
	require.NoError(t, err)
	require.Len(t, *lsheet, 1)
	ly := &LayerParams{}
	(*lsheet)[0].Set(ly)
	assert.Equal(t, float32(0.123), ly.Acts.Dend.GExp)

	psheet, err := sim.Params.Path.SheetByName("Script")
	require.NoError(t, err)
	pt := &PathParams{}
	(*psheet)[0].Set(pt)
	assert.Equal(t, float32(0.042), pt.Learn.LRate.Base)
}
