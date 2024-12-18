// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/math32"
)

// NBins is the number of spike bins
const NBins = 8

// KinaseNeuron has Neuron state
type KinaseNeuron struct {
	// Neuron spiking (0,1)
	Spike float32

	// Neuron probability of spiking
	SpikeP float32

	// CaSyn is spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning.
	CaSyn float32

	// regression variables
	StartCaSyn float32

	TotalSpikes float32

	// binned count of spikes, for regression learning
	SpikeBins [NBins]float32
}

func (kn *KinaseNeuron) Init() {
	kn.Spike = 0
	kn.SpikeP = 1
	kn.CaSyn = 0
	kn.StartTrial()
}

func (kn *KinaseNeuron) StartTrial() {
	kn.StartCaSyn = kn.CaSyn
	kn.TotalSpikes = 0
	for i := range kn.SpikeBins {
		kn.SpikeBins[i] = 0
	}
	// kn.CaSyn = 0 // note: better fits with carryover
}

// Cycle does one cycle of neuron updating, with given exponential spike interval
// based on target spiking firing rate.
func (ss *Sim) Cycle(kn *KinaseNeuron, expInt float32, cyc int) {
	kn.Spike = 0
	cycPerBin := ss.Config.Run.Cycles / NBins
	bin := cyc / cycPerBin
	if expInt > 0 {
		kn.SpikeP *= rand.Float32()
		if kn.SpikeP <= expInt {
			kn.Spike = 1
			kn.SpikeP = 1
			kn.TotalSpikes += 1
			kn.SpikeBins[bin] += 1
		}
	}
	kn.CaSyn += ss.NeurCa.SynDt * (ss.NeurCa.SpikeG*kn.Spike - kn.CaSyn)
}

func (kn *KinaseNeuron) SetInput(inputs []float32, off int) {
	inputs[off] = kn.StartCaSyn
	inputs[off+1] = kn.TotalSpikes
	for i, s := range kn.SpikeBins {
		inputs[off+2+i] = s
	}
}

// KinaseSynapse has Synapse state
type KinaseSynapse struct {
	// CaM is first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
	CaM float32

	// CaP is shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
	CaP float32

	// CaD is longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
	CaD float32

	// DWt is the CaP - CaD
	DWt float32
}

func (ks *KinaseSynapse) Init() {
	ks.CaM = 0
	ks.CaP = 0
	ks.CaD = 0
	ks.DWt = 0
}

// KinaseState is basic Kinase equation state
type KinaseState struct {

	// SSE for decoder
	SSE float32

	// Condition counter
	Condition int

	// Condition description
	Cond string

	// Trial counter
	Trial int

	// Cycle counter
	Cycle int

	// phase-based firing rates
	MinusHz, PlusHz float32

	// ErrDWt is the target error dwt: PlusHz - MinusHz
	ErrDWt float32

	// Sending neuron
	Send KinaseNeuron

	// Receiving neuron
	Recv KinaseNeuron

	// Standard synapse values
	StdSyn KinaseSynapse

	// Linear synapse values
	LinearSyn KinaseSynapse

	// binned integration of send, recv spikes
	SpikeBins [NBins]float32
}

func (ks *KinaseState) Init() {
	ks.Send.Init()
	ks.Recv.Init()
	ks.StdSyn.Init()
	ks.LinearSyn.Init()
}

func (kn *KinaseState) StartTrial() {
	kn.Send.StartTrial()
	kn.Recv.StartTrial()
}

func (ss *Sim) ConfigKinase() {
}

// Sweep runs a sweep through minus-plus ranges
func (ss *Sim) Sweep() {
	// hz := []float32{25, 50, 100}
	// nhz := len(hz)

	ss.StatsStart(Test, Condition)
	nhz := 100 / 5
	hz := make([]float32, nhz)
	i := 0
	for h := float32(5); h <= 100; h += 5 {
		hz[i] = h
		i++
	}

	cond := 0
	for mi := 0; mi < nhz; mi++ {
		minusHz := hz[mi]
		for pi := 0; pi < nhz; pi++ {
			plusHz := hz[pi]
			condStr := fmt.Sprintf("%03d -> %03d", minusHz, plusHz)
			ss.Kinase.Condition = cond
			ss.Kinase.Cond = condStr
			ss.RunImpl(minusHz, plusHz, ss.Config.Run.Trials)
			cond++
		}
	}
}

// Run runs for given parameters
func (ss *Sim) Run() {
	ss.RunImpl(ss.Config.MinusHz, ss.Config.PlusHz, ss.Config.Run.Trials)
}

// RunImpl runs NTrials, recording to RunLog and TrialLog
func (ss *Sim) RunImpl(minusHz, plusHz float32, ntrials int) {
	if ss.GUI.StopNow {
		return
	}
	ss.StatsStart(Test, Trial)
	ss.Kinase.Init()
	for trl := 0; trl < ntrials; trl++ {
		ss.Kinase.Trial = trl
		ss.TrialImpl(minusHz, plusHz)
	}
	ss.StatsStep(Test, Condition)
}

func (ss *Sim) Trial() {
	ss.Kinase.Init()
	ss.TrialImpl(ss.Config.MinusHz, ss.Config.PlusHz)
}

// TrialImpl runs one trial for given parameters
func (ss *Sim) TrialImpl(minusHz, plusHz float32) {
	if ss.GUI.StopNow {
		return
	}
	ss.StatsStart(Test, Trial)
	cfg := ss.Config
	ks := &ss.Kinase
	ks.MinusHz = minusHz
	ks.PlusHz = plusHz
	ks.Cycle = 0
	ks.ErrDWt = (plusHz - minusHz) / 100

	minusCycles := cfg.Run.Cycles - cfg.Run.PlusCycles

	ks.StartTrial()
	for phs := 0; phs < 2; phs++ {
		var maxcyc int
		var rhz float32
		switch phs {
		case 0:
			rhz = minusHz
			maxcyc = minusCycles
		case 1:
			rhz = plusHz
			maxcyc = cfg.Run.PlusCycles
		}
		shz := rhz + cfg.SendDiffHz
		if shz < 0 {
			shz = 0
		}

		var Sint, Rint float32
		if rhz > 5 {
			Rint = math32.Exp(-1000.0 / float32(rhz))
		}
		if shz > 5 {
			Sint = math32.Exp(-1000.0 / float32(shz))
		}
		for t := 0; t < maxcyc; t++ {
			ss.Cycle(&ks.Send, Sint, ks.Cycle)
			ss.Cycle(&ks.Recv, Rint, ks.Cycle)

			ca := ks.Send.CaSyn * ks.Recv.CaSyn
			ss.SynCa.FromCa(ca, &ks.StdSyn.CaM, &ks.StdSyn.CaP, &ks.StdSyn.CaD)
			ss.StatsStep(Test, Cycle)
			ks.Cycle++
		}
	}
	ks.StdSyn.DWt = ks.StdSyn.CaP - ks.StdSyn.CaD

	for i := range ks.SpikeBins {
		ks.SpikeBins[i] = 0.1 * (ks.Recv.SpikeBins[i] * ks.Send.SpikeBins[i])
	}

	ss.LinearSynCa.FinalCa(ks.SpikeBins[0], ks.SpikeBins[1], ks.SpikeBins[2], ks.SpikeBins[3], ks.SpikeBins[4], ks.SpikeBins[5], ks.SpikeBins[6], ks.SpikeBins[7], &ks.LinearSyn.CaP, &ks.LinearSyn.CaD)
	ks.LinearSyn.DWt = ks.LinearSyn.CaP - ks.LinearSyn.CaD

	ss.StatsStep(Test, Trial)
}
