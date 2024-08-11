// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start layervals

// ActAvgValues are long-running-average activation levels stored in the LayerValues,
// for monitoring and adapting inhibition and possibly scaling parameters.
// All of these integrate over NData within a network, so are the same across them.
type ActAvgValues struct {

	// running-average minus-phase activity integrated at Dt.LongAvgTau -- used for adapting inhibition relative to target level
	ActMAvg float32 `edit:"-"`

	// running-average plus-phase activity integrated at Dt.LongAvgTau
	ActPAvg float32 `edit:"-"`

	// running-average max of minus-phase Ge value across the layer integrated at Dt.LongAvgTau
	AvgMaxGeM float32 `edit:"-"`

	// running-average max of minus-phase Gi value across the layer integrated at Dt.LongAvgTau
	AvgMaxGiM float32 `edit:"-"`

	// multiplier on inhibition -- adapted to maintain target activity level
	GiMult float32 `edit:"-"`

	pad, pad1, pad2 float32
}

func (lv *ActAvgValues) Init() {
	lv.ActMAvg = 0 // will be set to ly.Params.InhibActAvg.Nominal in InitWeights
	lv.ActPAvg = 0 // will be set to ly.Params.InhibActAvg.Nominal in InitWeights
	lv.AvgMaxGeM = 1
	lv.AvgMaxGiM = 1
	lv.GiMult = 1
}

// CorSimStats holds correlation similarity (centered cosine aka normalized dot product)
// statistics at the layer level
type CorSimStats struct {

	// correlation (centered cosine aka normalized dot product) activation difference between ActP and ActM on this alpha-cycle for this layer -- computed by CorSimFromActs called by PlusPhase
	Cor float32 `edit:"-"`

	// running average of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase
	Avg float32 `edit:"-"`

	// running variance of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase
	Var float32 `edit:"-"`

	pad float32
}

func (cd *CorSimStats) Init() {
	cd.Cor = 0
	cd.Avg = 0
	cd.Var = 0
}

// LaySpecialValues holds special values used to communicate to other layers
// based on neural values, used for special algorithms such as RL where
// some of the computation is done algorithmically.
type LaySpecialValues struct {

	// one value
	V1 float32 `edit:"-"`

	// one value
	V2 float32 `edit:"-"`

	// one value
	V3 float32 `edit:"-"`

	// one value
	V4 float32 `edit:"-"`
}

func (lv *LaySpecialValues) Init() {
	lv.V1 = 0
	lv.V2 = 0
	lv.V3 = 0
	lv.V4 = 0
}

// LayerValues holds extra layer state that is updated per layer.
// It is sync'd down from the GPU to the CPU after every Cycle.
type LayerValues struct {

	// layer index for these vals
	LayIndex uint32 `display:"-"`

	// data index for these vals
	DataIndex uint32 `display:"-"`

	// reaction time for this layer in cycles, which is -1 until the Max CaSpkP level (after MaxCycStart) exceeds the Act.Attn.RTThr threshold
	RT  float32 `edit:"-"`
	pad uint32

	// running-average activation levels used for adaptive inhibition, and other adapting values
	ActAvg ActAvgValues `display:"inline"`

	// correlation (centered cosine aka normalized dot product) similarity between ActM, ActP states
	CorSim CorSimStats

	// special values used to communicate to other layers based on neural values computed on the GPU -- special cross-layer computations happen CPU-side and are sent back into the network via Context on the next cycle -- used for special algorithms such as RL / DA etc
	Special LaySpecialValues `display:"inline"`
}

func (lv *LayerValues) Init() {
	lv.ActAvg.Init()
	lv.CorSim.Init()
	lv.Special.Init()
}

//gosl:end layervals
