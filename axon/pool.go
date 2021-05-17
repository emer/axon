// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fffb"
	"github.com/emer/etable/minmax"
)

// Pool contains computed values for FFFB inhibition, and various other state values for layers
// and pools (unit groups) that can be subject to inhibition, including:
// * average / max stats on Ge and Act that drive inhibition
type Pool struct {
	StIdx, EdIdx int             `desc:"starting and ending (exlusive) indexes for the list of neurons in this pool"`
	Inhib        fffb.Inhib      `desc:"FFFB inhibition computed values, including Ge and Act AvgMax which drive inhibition"`
	ActM         minmax.AvgMax32 `desc:"minus phase average and max Act activation values, for ActAvg updt"`
	ActP         minmax.AvgMax32 `desc:"plus phase average and max Act activation values, for ActAvg updt"`
	GeM          minmax.AvgMax32 `desc:"stats for GeM minus phase averaged Ge values"`
	GiM          minmax.AvgMax32 `desc:"stats for GiM minus phase averaged Gi values"`
	AvgDif       minmax.AvgMax32 `desc:"absolute value of AvgDif differences from actual neuron ActPct relative to TrgAvg"`
}

func (pl *Pool) Init() {
	pl.Inhib.Init()
}
