// Code generated by "goal build"; DO NOT EDIT.
//line path.goal:1
// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
)

// https://github.com/kisvegabor/abbreviations-in-code suggests Buf instead of Buff

// index naming:
// syi =  path-relative synapse index (per existing usage)
// syni = network-relative synapse index -- add SynStIndex to syi

func (pt *Path) Defaults() {
	if pt.Params == nil {
		return
	}
	ctx := pt.Recv.Network.Context()
	pt.Params.PathType = pt.Type
	pt.Params.Defaults()
	pt.Params.Learn.KinaseCa.WtsForNCycles(int(ctx.ThetaCycles))
	switch pt.Type {
	case InhibPath:
		pt.Params.SWts.Adapt.On.SetBool(false)
	case BackPath:
		pt.Params.PathScale.Rel = 0.1
	case RWPath, TDPredPath:
		pt.Params.RLPredDefaults()
	case BLAPath:
		pt.Params.BLADefaults()
	case HipPath:
		pt.Params.HipDefaults()
	case VSPatchPath:
		pt.Params.VSPatchDefaults()
	case VSMatrixPath:
		pt.Params.MatrixDefaults()
	case DSMatrixPath:
		pt.Params.MatrixDefaults()
	}
	pt.ApplyDefaultParams()
	pt.UpdateParams()
}

// Update is interface that does local update of struct vals
func (pt *Path) Update() {
	if pt.Params == nil {
		return
	}
	if pt.Params.PathType == InhibPath {
		pt.Params.Com.GType = InhibitoryG
	}
	pt.Params.Update()
}

// UpdateParams updates all params given any changes
// that might have been made to individual values
func (pt *Path) UpdateParams() {
	pt.Update()
}

// AllParams returns a listing of all parameters in the Layer
func (pt *Path) AllParams() string {
	str := "///////////////////////////////////////////////////\nPath: " + pt.Name + "\n" + pt.Params.AllParams()
	return str
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pt *Path) SetSynValue(varNm string, sidx, ridx int, val float32) error {
	vidx, err := pt.SynVarIndex(varNm)
	if err != nil {
		return err
	}
	syi := uint32(pt.SynIndex(sidx, ridx))
	if syi < 0 || syi >= pt.NSyns {
		return err
	}
	syni := pt.SynStIndex + syi
	if vidx < int(SynapseVarsN) {
		Synapses.Set(val, int(SynapseVars(vidx)), int(syni))
	} else {
		for di := uint32(0); di < pt.Recv.MaxData; di++ {
			SynapseTraces.Set(val, int(SynapseTraceVars(vidx-int(SynapseVarsN))), int(syni), int(di))
		}
	}
	if varNm == "Wt" {
		wt := Synapses.Value(int(Wt), int(syni))
		if Synapses.Value(int(SWt), int(syni)) == 0 {
			Synapses.Set(wt, int(SWt), int(syni))
		}
		Synapses.Set(pt.Params.SWts.LWtFromWts(wt, Synapses.Value(int(SWt), int(syni))), int(LWt), int(syni))
	}
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// SetSWtsRPool initializes SWt structural weight values using given tensor
// of values which has unique values for each recv neuron within a given pool.
func (pt *Path) SetSWtsRPool(ctx *Context, swts tensor.Tensor) {
	rNuY := swts.DimSize(0)
	rNuX := swts.DimSize(1)
	rNu := rNuY * rNuX
	rfsz := swts.Len() / rNu

	rsh := pt.Recv.Shape
	rNpY := rsh.DimSize(0)
	rNpX := rsh.DimSize(1)
	r2d := false
	if rsh.NumDims() != 4 {
		r2d = true
		rNpY = 1
		rNpX = 1
	}

	wsz := swts.Len()

	for rpy := 0; rpy < rNpY; rpy++ {
		for rpx := 0; rpx < rNpX; rpx++ {
			for ruy := 0; ruy < rNuY; ruy++ {
				for rux := 0; rux < rNuX; rux++ {
					ri := 0
					if r2d {
						ri = rsh.IndexTo1D(ruy, rux)
					} else {
						ri = rsh.IndexTo1D(rpy, rpx, ruy, rux)
					}
					scst := (ruy*rNuX + rux) * rfsz
					syIndexes := pt.RecvSynIxs(uint32(ri))
					for ci, syi := range syIndexes {
						syni := pt.SynStIndex + syi
						swt := float32(swts.Float1D((scst + ci) % wsz))
						Synapses.Set(float32(swt), int(SWt), int(syni))
						wt := pt.Params.SWts.ClipWt(swt + (Synapses.Value(int(Wt), int(syni)) - pt.Params.SWts.Init.Mean))
						Synapses.Set(wt, int(Wt), int(syni))
						Synapses.Set(pt.Params.SWts.LWtFromWts(wt, swt), int(LWt), int(syni))
					}
				}
			}
		}
	}
}

// SetWeightsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
// Strongly suggest calling SWtRescale after.
func (pt *Path) SetWeightsFunc(ctx *Context, wtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := &pt.Recv.Shape
	rn := rsh.Len()
	ssh := &pt.Send.Shape

	for ri := 0; ri < rn; ri++ {
		syIndexes := pt.RecvSynIxs(uint32(ri))
		for _, syi := range syIndexes {
			syni := pt.SynStIndex + syi
			si := pt.Params.SynSendLayerIndex(syni)
			wt := wtFun(int(si), ri, ssh, rsh)
			Synapses.Set(wt, int(SWt), int(syni))
			Synapses.Set(wt, int(Wt), int(syni))
			Synapses.Set(0.5, int(LWt), int(syni))
		}
	}
}

// SetSWtsFunc initializes structural SWt values using given function
// based on receiving and sending unit indexes.
func (pt *Path) SetSWtsFunc(ctx *Context, swtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := &pt.Recv.Shape
	rn := rsh.Len()
	ssh := &pt.Send.Shape

	for ri := 0; ri < rn; ri++ {
		syIndexes := pt.RecvSynIxs(uint32(ri))
		for _, syi := range syIndexes {
			syni := pt.SynStIndex + syi
			si := int(pt.Params.SynSendLayerIndex(syni))
			swt := swtFun(si, ri, ssh, rsh)
			Synapses.Set(swt, int(SWt), int(syni))
			wt := pt.Params.SWts.ClipWt(swt + (Synapses.Value(int(Wt), int(syni)) - pt.Params.SWts.Init.Mean))
			Synapses.Set(wt, int(Wt), int(syni))
			Synapses.Set(pt.Params.SWts.LWtFromWts(wt, swt), int(LWt), int(syni))
		}
	}
}

// InitWeightsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pt *Path) InitWeightsSyn(ctx *Context, syni uint32, rnd randx.Rand, mean, spct float32) {
	pt.Params.SWts.InitWeightsSyn(ctx, syni, rnd, mean, spct)
}

// InitWeights initializes weight values according to SWt params,
// enforcing current constraints.
func (pt *Path) InitWeights(ctx *Context, nt *Network) {
	pt.Params.Learn.LRate.Init()
	pt.InitGBuffs()
	rlay := pt.Recv
	spct := pt.Params.SWts.Init.SPct
	if rlay.Params.IsTarget() {
		pt.Params.SWts.Init.SPct = 0
		spct = 0
	}
	smn := pt.Params.SWts.Init.Mean
	// todo: why is this recv based?  prob important to keep for consistency
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ni := rlay.NeurStIndex + lni
		if NrnIsOff(ni) {
			continue
		}
		syIndexes := pt.RecvSynIxs(lni)
		for _, syi := range syIndexes {
			syni := pt.SynStIndex + syi
			pt.InitWeightsSyn(ctx, syni, &nt.Rand, smn, spct)
		}
	}
	if pt.Params.SWts.Adapt.On.IsTrue() && !rlay.Params.IsTarget() {
		pt.SWtRescale(ctx)
	}
}

// SWtRescale rescales the SWt values to preserve the target overall mean value,
// using subtractive normalization.
func (pt *Path) SWtRescale(ctx *Context) {
	rlay := pt.Recv
	smn := pt.Params.SWts.Init.Mean
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ni := rlay.NeurStIndex + lni
		if NrnIsOff(ni) {
			continue
		}
		var nmin, nmax int
		var sum float32
		syIndexes := pt.RecvSynIxs(lni)
		nCons := len(syIndexes)
		if nCons <= 1 {
			continue
		}
		for _, syi := range syIndexes {
			syni := pt.SynStIndex + syi
			swt := Synapses.Value(int(SWt), int(syni))
			sum += swt
			if swt <= pt.Params.SWts.Limit.Min {
				nmin++
			} else if swt >= pt.Params.SWts.Limit.Max {
				nmax++
			}
		}
		amn := sum / float32(nCons)
		mdf := smn - amn // subtractive
		if mdf == 0 {
			continue
		}
		if mdf > 0 { // need to increase
			if nmax > 0 && nmax < nCons {
				amn = sum / float32(nCons-nmax)
				mdf = smn - amn
			}
			for _, syi := range syIndexes {
				syni := pt.SynStIndex + syi
				if Synapses.Value(int(SWt), int(syni)) <= pt.Params.SWts.Limit.Max {
					swt := pt.Params.SWts.ClipSWt(Synapses.Value(int(SWt), int(syni)) + mdf)
					Synapses.Set(swt, int(SWt), int(syni))
					Synapses.Set(pt.Params.SWts.WtValue(swt, Synapses.Value(int(LWt), int(syni))), int(Wt), int(syni))
				}
			}
		} else {
			if nmin > 0 && nmin < nCons {
				amn = sum / float32(nCons-nmin)
				mdf = smn - amn
			}
			for _, syi := range syIndexes {
				syni := pt.SynStIndex + syi
				if Synapses.Value(int(SWt), int(syni)) >= pt.Params.SWts.Limit.Min {
					swt := pt.Params.SWts.ClipSWt(Synapses.Value(int(SWt), int(syni)) + mdf)
					Synapses.Set(swt, int(SWt), int(syni))
					Synapses.Set(pt.Params.SWts.WtValue(swt, Synapses.Value(int(LWt), int(syni))), int(Wt), int(syni))
				}
			}
		}
	}
}

// sender based version:
//                     only looked at recv layers > send layers -- ri is "above" si
//  for:      ri    <- this is now sender on recip path: recipSi
//           ^ \    <- look for send back to original si, now as a receiver
//          /   v
// start: si == recipRi <- look in sy.RecvIndex of recipSi's sending cons for recipRi == si
//

// InitWtSym initializes weight symmetry.
// Is given the reciprocal pathway where
// the Send and Recv layers are reversed
// (see LayerBase RecipToRecvPath)
func (pt *Path) InitWtSym(ctx *Context, rpj *Path) {
	if len(rpj.SendCon) == 0 {
		return
	}
	slay := pt.Send
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pt.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			syni := pt.SynStIndex + syi
			ri := pt.Params.SynRecvLayerIndex(syni) // <- this sends to me, ri
			recipSi := ri                           // reciprocal case is si is now receiver
			recipc := rpj.SendCon[recipSi]
			if recipc.N == 0 {
				continue
			}
			firstSyni := rpj.SynStIndex + recipc.Start
			lastSyni := rpj.SynStIndex + recipc.Start + recipc.N - 1
			firstRi := rpj.Params.SynRecvLayerIndex(firstSyni)
			lastRi := rpj.Params.SynRecvLayerIndex(lastSyni)
			if lni < firstRi || lni > lastRi { // fast reject -- paths are always in order!
				continue
			}
			// start at index proportional to ri relative to rist
			up := int32(0)
			if lastRi > firstRi {
				up = int32(float32(recipc.N) * float32(lni-firstRi) / float32(lastRi-firstRi))
			}
			dn := up - 1

			for {
				doing := false
				if up < int32(recipc.N) {
					doing = true
					recipCi := uint32(recipc.Start) + uint32(up)
					recipSyni := rpj.SynStIndex + recipCi
					recipRi := rpj.Params.SynRecvLayerIndex(recipSyni)
					if recipRi == lni {
						Synapses.Set(Synapses.Value(int(Wt), int(syni)), int(Wt), int(recipSyni))
						Synapses.Set(Synapses.Value(int(LWt), int(syni)), int(LWt), int(recipSyni))
						Synapses.Set(Synapses.Value(int(SWt), int(syni)), int(SWt), int(recipSyni))
						// note: if we support SymFromTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					recipCi := uint32(recipc.Start) + uint32(dn)
					recipSyni := rpj.SynStIndex + recipCi
					recipRi := rpj.Params.SynRecvLayerIndex(recipSyni)
					if recipRi == lni {
						Synapses.Set(Synapses.Value(int(Wt), int(syni)), int(Wt), int(recipSyni))
						Synapses.Set(Synapses.Value(int(LWt), int(syni)), int(LWt), int(recipSyni))
						Synapses.Set(Synapses.Value(int(SWt), int(syni)), int(SWt), int(recipSyni))
						// note: if we support SymFromTop then can have option to go other way
						break
					}
					dn--
				}
				if !doing {
					break
				}
			}
		}
	}
}

// InitGBuffs initializes the per-pathway synaptic conductance buffers.
// This is not typically needed (called during InitWeights, InitActs)
// but can be called when needed.  Must be called to completely initialize
// prior activity, e.g., full Glong clearing.
func (pt *Path) InitGBuffs() {
	for ri := range pt.GBuf {
		pt.GBuf[ri] = 0
	}
	for ri := range pt.GSyns {
		pt.GSyns[ri] = 0
	}
}
