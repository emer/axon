// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsfffb

//gosl:start

// InhibVars are inhibitory variables for computing fsfffb inhibition.
type InhibVars int32 //enums:enum

const (
	// FFsRaw is the raw aggregation of all feedforward incoming spikes into neurons
	// in this pool. It is integrated using FFsRawInt in InhibIntVars.
	FFsRaw InhibVars = iota

	// FBsRaw is the raw aggregation of all feedback outgoing spikes generated
	// from neurons in this pool. It is integrated using FBsRawInt in InhibIntVars.
	FBsRaw

	// GeExtRaw is the raw aggregation of all extra GeExt conductances added to neurons.
	// It is integrated using GeExtRawInt in InhibIntVars.
	GeExtRaw

	// FFs is all feedforward incoming spikes into neurons in this pool,
	// normalized by pool size.
	FFs

	// FBs is all feedback outgoing spikes generated from neurons in this pool,
	// normalized by pool size.
	FBs

	// GeExts is all extra GeExt conductances added to neurons,
	// normalized by pool size.
	GeExts

	// FSi is the fast spiking PV+ fast integration of FFs feedforward spikes.
	FSi

	// SSi is the slow spiking SST+ integration of FBs feedback spikes.
	SSi

	// SSf is the slow spiking facilitation factor, representing facilitating
	// effects of recent activity.
	SSf

	// FSGi is the overall fast-spiking inhibitory conductance.
	FSGi

	// SSGi is the overall slow-spiking inhibitory conductance.
	SSGi

	// Gi is the overall inhibitory conductance = FSGi + SSGi.
	Gi

	// GiOrig is the original value of the inhibition (before pool or other effects).
	GiOrig

	// LayGi is the layer-level inhibition that is MAX'd with the pool-level
	// inhibition to produce the net inhibition, only for sub-pools.
	LayGi

	// FFAvg is the longer time scale running average FF drive, used for FFAvgPrv.
	FFAvg

	// FFAvgPrv is the previous theta cycle FFAvg value, for the FFPrv factor.
	// Updated in the Decay function that is called at start of new State / Trial.
	FFAvgPrv
)

// InhibIntVars are int32 inhibitory variables for computing fsfffb inhibition.
// Note: we combine these with PoolIntVars in axon.
type InhibIntVars int32 //enums:enum

const (
	// FFsRawInt is the int32 atomic add compatible integration of FFsRaw.
	FFsRawInt InhibIntVars = iota

	// FBsRawInt is the int32 atomic add compatible integration of FBsRaw.
	FBsRawInt

	// GeExtRawInt is the int32 atomic add compatible integration of GeExtRaw.
	GeExtRawInt

	// Clamped if true (!=0), this layer is hard-clamped and should
	// use GeExts exclusively for PV.
	Clamped
)

//gosl:end
