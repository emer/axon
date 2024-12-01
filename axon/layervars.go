// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start

// LayerVars are layer-level state values.
type LayerVars int32 //enums:enum

const (
	// LayerActMAvg is the running-average minus-phase activity integrated
	// at Dt.LongAvgTau, used for adapting inhibition relative to target level.
	LayerActMAvg LayerVars = iota

	// LayerActPAvg is the running-average plus-phase activity integrated at Dt.LongAvgTau.
	LayerActPAvg

	// LayerAvgMaxGeM is the running-average max of minus-phase Ge value across the layer
	// integrated at Dt.LongAvgTau.
	LayerAvgMaxGeM

	// LayerAvgMaxGiM is the running-average max of minus-phase Gi value across the layer
	// integrated at Dt.LongAvgTau.
	LayerAvgMaxGiM

	// LayerGiMult is a multiplier on layer-level inhibition, which can be adapted to
	// maintain target activity level.
	LayerGiMult

	// LayerPhaseDiff is the phase-wise difference in the activity state between the
	// minus [ActM] and plus [ActP] phases, measured using 1 minus the correlation
	// (centered cosine aka normalized dot product).  0 = no difference,
	// 2 = maximum difference. Computed by PhaseDiffFromActs in the PlusPhase.
	LayerPhaseDiff

	// LayerPhaseDiffAvg is the running average of [LayerPhaseDiff] over time,
	// integrated at Dt.LongAvgTau.
	LayerPhaseDiffAvg

	// LayerPhaseDiffVar is the running variance of [LayerPhaseDiff], integrated
	// at Dt.LongAvgTau.
	LayerPhaseDiffVar

	// LayerRT is the reaction time for this layer in cycles, which is -1 until the
	// Max CaP level (after MaxCycStart) exceeds the Act.Attn.RTThr threshold.
	LayerRT

	// LayerRewPredPos is the positive-valued Reward Prediction value, for
	// RL specific layers: [RWPredLayer], [TDPredLayer].
	// For [TDIntegLayer], this is the plus phase current integrated reward prediction.
	LayerRewPredPos

	// LayerRewPredNeg is the negative-valued Reward Prediction value, for
	// RL specific layers: [RWPredLayer], [TDPredLayer]
	// For [TDIntegLayer], this is the minus phase previous integrated reward prediction.
	LayerRewPredNeg
)

//gosl:end
