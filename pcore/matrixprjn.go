// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// MatrixTraceParams for for trace-based learning in the MatrixPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is reset at time of reward based on ACh level from CINs.
type MatrixTraceParams struct {
	CurTrlDA bool    `def:"true" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	Decay    float32 `def:"2" min:"0" desc:"multiplier on CIN ACh level for decaying prior traces -- decay never exceeds 1.  larger values drive strong credit assignment for any US outcome."`
}

func (tp *MatrixTraceParams) Defaults() {
	tp.CurTrlDA = true
	tp.Decay = 2
}

// MatrixPrjn does dopamine-modulated, gated trace learning, for Matrix learning
// in PBWM context
type MatrixPrjn struct {
	axon.Prjn
	Trace  MatrixTraceParams `view:"inline" desc:"special parameters for matrix trace learning"`
	TrSyns []TraceSyn        `desc:"trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`
}

var KiT_MatrixPrjn = kit.Types.AddType(&MatrixPrjn{}, axon.PrjnProps)

func (pj *MatrixPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.Trace.Defaults()
	// no additional factors
	pj.SWt.Adapt.SigGain = 1
}

func (pj *MatrixPrjn) Build() error {
	err := pj.Prjn.Build()
	pj.TrSyns = make([]TraceSyn, len(pj.SConIdx))
	return err
}

func (pj *MatrixPrjn) ClearTrace() {
	for si := range pj.TrSyns {
		tsy := &pj.TrSyns[si]
		tsy.NTr = 0
		sy := &pj.Syns[si]
		sy.Tr = 0
	}
}

func (pj *MatrixPrjn) InitWts() {
	pj.Prjn.InitWts()
	pj.ClearTrace()
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *MatrixPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Enabled {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(*MatrixLayer)

	da := rlay.DA
	daLrn := rlay.DALrn // includes d2 reversal etc

	ach := rlay.ACh
	achDk := mat32.Min(1, ach*pj.Trace.Decay)

	lr := pj.Learn.Lrate.Eff

	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		trsyns := pj.TrSyns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			trsy := &trsyns[ci]
			ri := scons[ci]
			rn := &rlay.PCoreNeurs[ri]
			tr := sy.Tr

			ntr := rn.ActLrn * sn.Act // sn.ActLrn // todo
			dwt := float32(0)

			if pj.Trace.CurTrlDA {
				tr += ntr
			}

			if da != 0 {
				dwt = daLrn * tr
			}
			tr -= achDk * tr // decay trace that drove dwt

			if !pj.Trace.CurTrlDA {
				tr += ntr
			}
			sy.Tr = tr
			trsy.NTr = ntr
			sy.DWt += lr * dwt // note: missing rn.RLrate *
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// SynVals

// SynVarIdx returns the index of given variable within the synapse,
// according to *this prjn's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *MatrixPrjn) SynVarIdx(varNm string) (int, error) {
	vidx, err := pj.Prjn.SynVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nn := pj.Prjn.SynVarNum()
	switch varNm {
	case "NTr":
		return nn, nil
	}
	return -1, fmt.Errorf("MatrixPrjn SynVarIdx: variable name: %v not valid", varNm)
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *MatrixPrjn) SynVal1D(varIdx int, synIdx int) float32 {
	if varIdx < 0 || varIdx >= len(SynVarsAll) {
		return mat32.NaN()
	}
	nn := pj.Prjn.SynVarNum()
	if varIdx < nn {
		return pj.Prjn.SynVal1D(varIdx, synIdx)
	}
	if synIdx < 0 || synIdx >= len(pj.TrSyns) {
		return mat32.NaN()
	}
	varIdx -= nn
	sy := &pj.TrSyns[synIdx]
	return sy.VarByIndex(varIdx)
}

// SynVarNum returns the number of synapse-level variables
// for this prjn.  This is needed for extending indexes in derived types.
func (pj *MatrixPrjn) SynVarNum() int {
	return pj.Prjn.SynVarNum() + len(TraceSynVars)
}

func (pj *MatrixPrjn) SynVarNames() []string {
	return SynVarsAll
}
