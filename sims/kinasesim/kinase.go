// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/kinase"
)

// KinaseNeuron has Neuron state
type KinaseNeuron struct {
	// Neuron spiking (0,1)
	Spike float32 `edit:"-"`

	// Neuron probability of spiking
	SpikeP float32 `edit:"-"`

	// CaSyn is spike-driven calcium trace for synapse-level Ca-driven learning:
	// exponential integration of SpikeCaSyn * Spike at CaSynTau time constant (typically 30).
	// Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the
	// synaptic trace driving credit assignment in learning. Time constant reflects
	// binding time of Glu to NMDA and Ca buffering postsynaptically, and determines
	// time window where pre * post spiking must overlap to drive learning.
	CaSyn float32 `edit:"-"`

	// regression variables
	StartCaSyn float32 `edit:"-"`

	TotalSpikes float32 `edit:"-"`

	// binned count of spikes, for regression learning
	CaBins []float32
}

func (kn *KinaseNeuron) Init() {
	kn.Spike = 0
	kn.SpikeP = 1
	kn.CaSyn = 0
	kn.StartTrial()
}

func (kn *KinaseNeuron) Config(nCaBins int) {
	kn.CaBins = make([]float32, nCaBins)
}

func (kn *KinaseNeuron) StartTrial() {
	kn.StartCaSyn = kn.CaSyn
	kn.TotalSpikes = 0
	for i := range kn.CaBins {
		kn.CaBins[i] = 0
	}
	// kn.CaSyn = 0 // note: better fits with carryover
}

// Cycle does one cycle of neuron updating, with given exponential spike interval
// based on target spiking firing rate.
func (ss *Sim) Cycle(kn *KinaseNeuron, expInt float32, cyc int) {
	kn.Spike = 0
	if expInt > 0 {
		kn.SpikeP *= rand.Float32()
		if kn.SpikeP <= expInt {
			kn.Spike = 1
			kn.SpikeP = 1
			kn.TotalSpikes += 1
		}
	}
	kn.CaSyn += ss.CaSpike.CaSynDt * (ss.CaSpike.SpikeCaSyn*kn.Spike - kn.CaSyn)
	bin := cyc / ss.Config.Run.CaBinCycles
	kn.CaBins[bin] += kn.CaSyn / float32(ss.Config.Run.CaBinCycles)
}

func (kn *KinaseNeuron) SetInput(inputs []float32, off int) {
	inputs[off] = kn.StartCaSyn
	inputs[off+1] = kn.TotalSpikes
	for i, s := range kn.CaBins {
		inputs[off+2+i] = s
	}
}

// KinaseSynapse has Synapse state
type KinaseSynapse struct {
	// CaM is first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
	CaM float32 `edit:"-" width:"12"`

	// CaP is shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
	CaP float32 `edit:"-" width:"12"`

	// CaD is longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
	CaD float32 `edit:"-" width:"12"`

	// DWt is the CaP - CaD
	DWt float32 `edit:"-" width:"12"`
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

	// Current ca bin value
	CaBin float32

	// Linear synapse values
	LinearSyn KinaseSynapse

	// binned integration of send, recv spikes
	CaBins []float32
}

func (ks *KinaseState) Init() {
	ks.Send.Init()
	ks.Recv.Init()
	ks.StdSyn.Init()
	ks.LinearSyn.Init()
	ks.CaBin = 0
}

func (ks *KinaseState) Config(nCaBins int) {
	ks.Send.Config(nCaBins)
	ks.Recv.Config(nCaBins)
	ks.CaBins = make([]float32, nCaBins)
	ks.StdSyn.Init()
	ks.LinearSyn.Init()
}

func (kn *KinaseState) StartTrial() {
	kn.Send.StartTrial()
	kn.Recv.StartTrial()
	kn.LinearSyn.CaM = 0
	kn.LinearSyn.CaP = 0
	kn.LinearSyn.CaD = 0
}

func (ss *Sim) ConfigKinase() {
	ss.Config.Run.Update()
	nbins := ss.Config.Run.NCaBins
	ss.CaPWts = make([]float32, nbins)
	ss.CaDWts = make([]float32, nbins)
	nplus := ss.Config.Run.PlusCycles / ss.Config.Run.CaBinCycles
	kinase.CaBinWts(nplus, ss.Config.Run.CaBinCycles, ss.CaPWts, ss.CaDWts)
	ss.Kinase.Config(nbins)
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
			condStr := fmt.Sprintf("%03d -> %03d", int(minusHz), int(plusHz))
			ss.Kinase.Condition = cond
			ss.Kinase.Cond = condStr
			ss.StatsStart(Test, Condition)
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
	nbins := ss.Config.Run.NCaBins
	spikeBinCycles := ss.Config.Run.CaBinCycles
	lsint := 1.0 / float32(spikeBinCycles)

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

			// original synaptic-level integration into "StdSyn"
			ca := 8 * ks.Send.CaSyn * ks.Recv.CaSyn // 8 is standard CaGain Factor
			ss.CaSpike.Dt.FromCa(ca, &ks.StdSyn.CaM, &ks.StdSyn.CaP, &ks.StdSyn.CaD)

			// CaBin linear regression integration.
			bin := ks.Cycle / spikeBinCycles
			ks.CaBins[bin] = (ks.Recv.CaBins[bin] * ks.Send.CaBins[bin])
			ks.CaBin = ks.CaBins[bin]
			ks.LinearSyn.CaM = ks.CaBin
			ks.LinearSyn.CaP += lsint * ss.CaPWts[bin] * ks.CaBin // slow integ just for visualization
			ks.LinearSyn.CaD += lsint * ss.CaDWts[bin] * ks.CaBin

			ss.StatsStep(Test, Cycle)
			ks.Cycle++
		}
	}
	ks.StdSyn.DWt = ks.StdSyn.CaP - ks.StdSyn.CaD

	var cp, cd float32
	for i := range nbins {
		cp += ks.CaBins[i] * ss.CaPWts[i]
		cd += ks.CaBins[i] * ss.CaDWts[i]
	}
	ks.LinearSyn.DWt = cp - cd
	ks.LinearSyn.CaP = cp
	ks.LinearSyn.CaD = cd

	ss.StatsStep(Test, Trial)
}

// Regress runs the linear regression on the data
// func (ss *Sim) Regress() {
// 	r := glm.NewGLM()
// 	mode := Test
// 	level := Condition
// 	modeDir := ss.Stats.Dir(mode.String())
// 	levelDir := modeDir.Dir(level.String())
//
// 	dt := tensorfs.DirTable(axon.StatsNode(ss.Stats, Test, Condition))
// 	err := r.SetTable(&ls.Data, "State", "StdCa", "PredCa", "ErrCa")
// 	if err != nil {
// 		slog.Error(err.Error())
// 		return
// 	}
// 	r.DepNames = []string{"CaP", "CaD"}
// 	r.L1Cost = 0.1
// 	r.L2Cost = 0.1
// 	r.StopTolerance = 0.00001
// 	r.ZeroOffset = true
//
// 	// NBins = 4
// 	// r.Coeff.Values = []float64{
// 	// 	0.05, 0.25, 0.5, 0.6, 0, // linear progression
// 	// 	0.25, 0.5, 0.5, 0.25, 0} // hump in the middle
//
// 	// NBins = 8, 200+50 cycles
// 	// r.Coeff.Values = []float64{
// 	// 	0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 1.0, 1.0, 0, // linear progression
// 	// 	0.5, 0.65, 0.75, 0.9, 0.9, 0.9, 0.65, 0.55, .0} // hump in the middle
//
// 	// NBins = 8, 280+70 cycles
// 	r.Coeff.Values = []float64{
// 		0.0, 0.1, 0.23, 0.35, 0.45, 0.55, 0.75, 0.75, 0, // linear progression
// 		0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3, .0} // hump in the middle
//
// 	fmt.Println(r.Coeffs())
//
// 	r.Run()
//
// 	fmt.Println(r.Variance())
// 	fmt.Println(r.Coeffs())
//
// ls.Data.SaveCSV("linear_data.tsv", tensor.Tab, table.Headers)
// }
