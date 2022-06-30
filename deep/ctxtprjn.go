// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// CtxtSender is an interface for layers that implement the SendCtxtGe method
// (SuperLayer, CTLayer)
type CtxtSender interface {
	axon.AxonLayer

	// SendCtxtGe sends activation over CTCtxtPrjn projections to integrate
	// CtxtGe excitatory conductance on CT layers.
	// This must be called at the end of the Burst quarter for this layer.
	SendCtxtGe(ltime *axon.Time)
}

// CTCtxtPrjn is the "context" temporally-delayed projection into CTLayer,
// (corticothalamic deep layer 6) where the CtxtGe excitatory input
// is integrated only at end of Burst Quarter.
// Set FmSuper for the main projection from corresponding Super layer.
type CTCtxtPrjn struct {
	axon.Prjn           // access as .Prjn
	FmSuper   bool      `desc:"if true, this is the projection from corresponding Superficial layer -- should be OneToOne prjn, with Learn.Learn = false, WtInit.Var = 0, Mean = 0.8 -- these defaults are set if FmSuper = true"`
	CtxtGeInc []float32 `desc:"local per-recv unit accumulator for Ctxt excitatory conductance from sending units -- not a delta -- the full value"`
}

var KiT_CTCtxtPrjn = kit.Types.AddType(&CTCtxtPrjn{}, PrjnProps)

func (pj *CTCtxtPrjn) Defaults() {
	pj.Prjn.Defaults() // note: used to have other defaults
}

func (pj *CTCtxtPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *CTCtxtPrjn) Type() emer.PrjnType {
	return CTCtxt
}

func (pj *CTCtxtPrjn) PrjnTypeName() string {
	if pj.Typ < emer.PrjnTypeN {
		return pj.Typ.String()
	}
	ptyp := PrjnType(pj.Typ)
	ts := ptyp.String()
	sz := len(ts)
	if sz > 0 {
		return ts[:sz-1] // cut off trailing _
	}
	return ""
}

func (pj *CTCtxtPrjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	rsh := pj.Recv.Shape()
	rlen := rsh.Len()
	pj.CtxtGeInc = make([]float32, rlen)
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (pj *CTCtxtPrjn) InitGbuf() {
	pj.Prjn.InitGBufs()
	for ri := range pj.CtxtGeInc {
		pj.CtxtGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendSpike: disabled for this type
func (pj *CTCtxtPrjn) SendSpike(si int) {
}

// RecvGInc: disabled for this type
func (pj *CTCtxtPrjn) RecvGInc(ltime *axon.Time) {
}

// SendCtxtGe sends the full Burst activation from sending neuron index si,
// to integrate CtxtGe excitatory conductance on receivers
func (pj *CTCtxtPrjn) SendCtxtGe(si int, dburst float32) {
	scdb := dburst * pj.GScale.Scale
	nc := pj.SConN[si]
	st := pj.SConIdxSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIdx[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.CtxtGeInc[ri] += scdb * syns[ci].Wt
	}
}

// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the projections
func (pj *CTCtxtPrjn) RecvCtxtGeInc() {
	rlay, ok := pj.Recv.(*CTLayer)
	if !ok {
		return
	}
	for ri := range rlay.CtxtGes {
		rlay.CtxtGes[ri] += pj.CtxtGeInc[ri]
		pj.CtxtGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// SynCa does Kinase learning based on Ca -- doesn't do
func (pj *CTCtxtPrjn) SynCa(ltime *axon.Time) {
	return
}

// DWt computes the weight change (learning) for Ctxt projections
func (pj *CTCtxtPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	sslay, issuper := pj.Send.(*SuperLayer)
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	lr := pj.Learn.Lrate.Eff
	for si := range slay.Neurons {
		sact := float32(0)
		if issuper {
			sact = sslay.SuperNeurs[si].BurstPrv
		} else {
			sact = slay.Neurons[si].ActPrv
		}
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			if pj.Learn.ETrace.On {
				pj.Learn.ETrace.ETrFmCaP(&sy.ETr, sact*rn.CaP)
			}
			// following line should be ONLY diff: sact for *both* short and medium *sender*
			// activations, which are first two args:
			err := pj.Learn.CHLdWt(sact, sact, rn.CaP, rn.CaD)
			// sb immediately -- enters into zero sum
			if err > 0 {
				err *= (1 - sy.LWt)
			} else {
				err *= sy.LWt
			}
			sy.DWt += lr * err
			if pj.Learn.ETrace.On {
				sy.DWt *= sy.ETr
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnType

// PrjnType has the DeepAxon extensions to the emer.PrjnType types, for gui
type PrjnType emer.PrjnType

//go:generate stringer -type=PrjnType

var KiT_PrjnType = kit.Enums.AddEnumExt(emer.KiT_PrjnType, PrjnTypeN, kit.NotBitFlag, nil)

// The DeepAxon prjn types
const (
	// CTCtxt are projections from Superficial layers to CT layers that
	// send Burst activations drive updating of CtxtGe excitatory conductance,
	// at end of a DeepBurst quarter.  These projections also use a special learning
	// rule that takes into account the temporal delays in the activation states.
	// Can also add self context from CT for deeper temporal context.
	CTCtxt emer.PrjnType = emer.PrjnTypeN + iota
)

// gui versions
const (
	CTCtxt_ PrjnType = PrjnType(emer.PrjnTypeN) + iota
	PrjnTypeN
)

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnType,
}
