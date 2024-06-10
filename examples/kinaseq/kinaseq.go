// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"reflect"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/etime"
)

// KinaseState is basic Kinase equation state
type KinaseState struct {

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

	// Neuron spiking
	SendSpike, RecvSpike float32

	// Neuron probability of spiking
	SendP, RecvP float32

	// CaSyn is spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning.
	SendCaSyn, RecvCaSyn float32

	// CaM is first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
	CaM float32

	// CaP is shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
	CaP float32

	// CaD is longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
	CaD float32

	// CaUpT is time in CyclesTotal of last updating of Ca values at the synapse level, for optimized synaptic-level Ca integration -- converted to / from uint32
	CaUpT float32

	// DWt is the CaP - CaD
	DWt float32

	// ErrDWt is the target error dwt
	ErrDWt float32
}

func (ks *KinaseState) Init() {
	ks.SendSpike = 0
	ks.RecvSpike = 0
	ks.SendP = 1
	ks.RecvP = 1
	ks.SendCaSyn = 0
	ks.RecvCaSyn = 0
}

// Sweep runs a sweep through minus-plus ranges
func (ss *Sim) Sweep() {
	hz := []float32{25, 50, 100}
	nhz := len(hz)

	cond := 0
	for mi := 0; mi < nhz; mi++ {
		minusHz := hz[mi]
		for pi := 0; pi < nhz; pi++ {
			plusHz := hz[pi]
			condStr := fmt.Sprintf("%03d -> %03d", minusHz, plusHz)
			ss.Kinase.Condition = cond
			ss.Kinase.Cond = condStr
			ss.RunImpl(minusHz, plusHz, ss.Config.Run.NTrials)
			cond++
		}
	}
	// ss.Plot("DWtPlot").Update()
	// ss.Plot("DWtVarPlot").Update()
}

// Run runs for given parameters
func (ss *Sim) Run() {
	cr := &ss.Config.Run
	ss.RunImpl(cr.MinusHz, cr.PlusHz, cr.NTrials)
}

// RunImpl runs NTrials, recording to RunLog and TrialLog
func (ss *Sim) RunImpl(minusHz, plusHz float32, ntrials int) {
	ss.Kinase.Init()
	for trl := 0; trl < ntrials; trl++ {
		ss.Kinase.Trial = trl
		ss.TrialImpl(minusHz, plusHz)
	}
	ss.Logs.LogRow(etime.Test, etime.Condition, ss.Kinase.Condition)
	ss.GUI.UpdatePlot(etime.Test, etime.Condition)
}

func (ss *Sim) Trial() {
	cr := &ss.Config.Run
	ss.Kinase.Init()
	ss.TrialImpl(cr.MinusHz, cr.PlusHz)
}

// TrialImpl runs one trial for given parameters
func (ss *Sim) TrialImpl(minusHz, plusHz float32) {
	cfg := &ss.Config
	ks := &ss.Kinase
	ks.MinusHz = minusHz
	ks.PlusHz = plusHz
	ks.Cycle = 0
	for phs := 0; phs < 3; phs++ {
		var maxms int
		var rhz float32
		switch phs {
		case 0:
			rhz = minusHz
			maxms = cfg.Run.MinusMSec
		case 1:
			rhz = plusHz
			maxms = cfg.Run.PlusMSec
		case 2:
			rhz = 0
			maxms = cfg.Run.ISIMSec
		}
		shz := rhz + cfg.Run.SendDiffHz
		if shz < 0 {
			shz = 0
		}

		var Sint, Rint float32
		if rhz > 0 {
			Rint = math32.Exp(-1000.0 / float32(rhz))
		}
		if shz > 0 {
			Sint = math32.Exp(-1000.0 / float32(shz))
		}
		for t := 0; t < maxms; t++ {
			ks.SendSpike = 0
			if Sint > 0 {
				ks.SendP *= rand.Float32()
				if ks.SendP <= Sint {
					ks.SendSpike = 1
					ks.SendP = 1
				}
			}
			ks.SendCaSyn += cfg.Params.SynDt * (cfg.Params.SpikeG*ks.SendSpike - ks.SendCaSyn)

			ks.RecvSpike = 0
			if Rint > 0 {
				ks.RecvP *= rand.Float32()
				if ks.RecvP <= Rint {
					ks.RecvSpike = 1
					ks.RecvP = 1
				}
			}
			ks.RecvCaSyn += cfg.Params.SynDt * (cfg.Params.SpikeG*ks.RecvSpike - ks.RecvCaSyn)

			ca := ks.SendCaSyn * ks.RecvCaSyn
			ss.CaParams.FromCa(ca, &ks.CaM, &ks.CaP, &ks.CaD)

			ss.Logs.LogRow(etime.Test, etime.Cycle, ks.Cycle)
			ks.Cycle++
		}
		if phs == 1 {
			ks.DWt = ks.CaP - ks.CaD
		}
	}

	ks.ErrDWt = (plusHz - minusHz) / 100

	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
	ss.Logs.LogRow(etime.Test, etime.Trial, ks.Trial)
	ss.GUI.UpdatePlot(etime.Test, etime.Trial)
}

func (ss *Sim) ConfigKinaseLogItems() {
	lg := &ss.Logs

	ks := &ss.Kinase
	typ := reflect.TypeOf(*ks)
	val := reflect.ValueOf(ks).Elem()
	nf := typ.NumField()
	for i := 0; i < nf; i++ {
		field := typ.Field(i)
		itm := lg.AddItem(&elog.Item{
			Name:   field.Name,
			Type:   field.Type.Kind(),
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					switch field.Type.Kind() {
					case reflect.Float32:
						ctx.SetFloat32(val.Field(i).Interface().(float32))
					case reflect.Int:
						ctx.SetFloat32(float32(val.Field(i).Interface().(int)))
					case reflect.String:
						ctx.SetString(val.Field(i).Interface().(string))
					}
				},
			}})
		times := []etime.Times{etime.Condition, etime.Trial, etime.Cycle}
		if field.Type.Kind() == reflect.Float32 {
			lg.AddStdAggs(itm, etime.Test, times...)
		} else {
			tn := len(times)
			for ti := 0; ti < tn-1; ti++ {
				itm.Write[etime.Scope(etime.Test, times[ti])] = func(ctx *elog.Context) {
					switch field.Type.Kind() {
					case reflect.Int:
						ctx.SetFloat32(float32(val.Field(i).Interface().(int)))
					case reflect.String:
						ctx.SetString(val.Field(i).Interface().(string))
					}
				}
			}
		}
	}
}
