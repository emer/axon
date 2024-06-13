// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"io"
	"strconv"

	"cogentcore.org/core/base/indent"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/weights"
)

// https://github.com/kisvegabor/abbreviations-in-code suggests Buf instead of Buff

// index naming:
// syi =  path-relative synapse index (per existing usage)
// syni = network-relative synapse index -- add SynStIndex to syi

// axon.Path is a basic Axon pathway with synaptic learning parameters
type Path struct {
	PathBase

	// all path-level parameters -- these must remain constant once configured
	Params *PathParams
}

// Object returns the object with parameters to be set by emer.Params
func (pj *Path) Object() any {
	return pj.Params
}

// AsAxon returns this path as a axon.Path -- all derived paths must redefine
// this to return the base Path type, so that the AxonPath interface does not
// need to include accessors to all the basic stuff.
func (pj *Path) AsAxon() *Path {
	return pj
}

// PathType returns axon specific cast of pj.Typ path type
func (pj *Path) PathType() PathTypes {
	return PathTypes(pj.Typ)
}

func (pj *Path) Class() string {
	return pj.PathType().String() + " " + pj.Cls
}

func (pj *Path) Defaults() {
	if pj.Params == nil {
		return
	}
	pj.Params.PathType = pj.PathType()
	pj.Params.Defaults()
	switch pj.PathType() {
	case InhibPath:
		pj.Params.SWts.Adapt.On.SetBool(false)
	case BackPath:
		pj.Params.PathScale.Rel = 0.1
	case RWPath, TDPredPath:
		pj.Params.RLPredDefaults()
	case BLAPath:
		pj.Params.BLADefaults()
	case HipPath:
		pj.Params.HipDefaults()
	case VSPatchPath:
		pj.Params.VSPatchDefaults()
	case VSMatrixPath:
		pj.Params.MatrixDefaults()
	case DSMatrixPath:
		pj.Params.MatrixDefaults()
	}
	pj.ApplyDefParams()
	pj.UpdateParams()
}

// Update is interface that does local update of struct vals
func (pj *Path) Update() {
	if pj.Params == nil {
		return
	}
	if pj.Params.PathType == InhibPath {
		pj.Params.Com.GType = InhibitoryG
	}
	pj.Params.Update()
}

// UpdateParams updates all params given any changes
// that might have been made to individual values
func (pj *Path) UpdateParams() {
	pj.Update()
}

// AllParams returns a listing of all parameters in the Layer
func (pj *Path) AllParams() string {
	str := "///////////////////////////////////////////////////\nPath: " + pj.Name() + "\n" + pj.Params.AllParams()
	return str
}

// SetParam sets parameter at given path to given value.
// returns error if path not found or value cannot be set.
func (pj *Path) SetParam(path, val string) error {
	return params.SetParam(pj.Params, path, val)
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pj *Path) SetSynValue(varNm string, sidx, ridx int, val float32) error {
	vidx, err := pj.SynVarIndex(varNm)
	if err != nil {
		return err
	}
	syi := uint32(pj.SynIndex(sidx, ridx))
	if syi < 0 || syi >= pj.NSyns {
		return err
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIndex + syi
	if vidx < int(SynapseVarsN) {
		SetSynV(ctx, syni, SynapseVars(vidx), val)
	} else {
		for di := uint32(0); di < pj.Recv.MaxData; di++ {
			SetSynCaV(ctx, syni, di, SynapseCaVars(vidx-int(SynapseVarsN)), val)
		}
	}
	if varNm == "Wt" {
		wt := SynV(ctx, syni, Wt)
		if SynV(ctx, syni, SWt) == 0 {
			SetSynV(ctx, syni, SWt, wt)
		}
		SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFromWts(wt, SynV(ctx, syni, SWt)))
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWtsJSON writes the weights from this pathway from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pj *Path) WriteWtsJSON(w io.Writer, depth int) {
	ctx := &pj.Recv.Network.Ctx
	slay := pj.Send
	rlay := pj.Recv
	nr := int(rlay.NNeurons)
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"From\": %q,\n", slay.Name())))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Rs\": [\n")))
	depth++
	for ri := 0; ri < nr; ri++ {
		rc := pj.RecvCon[ri]
		syIndexes := pj.RecvSynIndexes(uint32(ri))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("{\n"))
		depth++
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"Ri\": %v,\n", ri)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"N\": %v,\n", rc.N)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Si\": [ "))
		for ci, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			si := pj.Params.SynSendLayIndex(ctx, syni)
			w.Write([]byte(fmt.Sprintf("%v", si)))
			if ci == int(rc.N-1) {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt\": [ "))
		for ci, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			w.Write([]byte(strconv.FormatFloat(float64(SynV(ctx, syni, Wt)), 'g', weights.Prec, 32)))
			if ci == int(rc.N-1) {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt1\": [ ")) // Wt1 is SWt
		for ci, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			w.Write([]byte(strconv.FormatFloat(float64(SynV(ctx, syni, SWt)), 'g', weights.Prec, 32)))
			if ci == int(rc.N-1) {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("]\n"))
		depth--
		w.Write(indent.TabBytes(depth))
		if ri == nr-1 {
			w.Write([]byte("}\n"))
		} else {
			w.Write([]byte("},\n"))
		}
	}
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("]\n"))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("}")) // note: leave unterminated as outer loop needs to add , or just \n depending
}

// ReadWtsJSON reads the weights from this pathway from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one path only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (pj *Path) ReadWtsJSON(r io.Reader) error {
	pw, err := weights.PathReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	return pj.SetWts(pw)
}

// SetWts sets the weights for this pathway from weights.Path decoded values
func (pj *Path) SetWts(pw *weights.Path) error {
	var err error
	for i := range pw.Rs {
		pr := &pw.Rs[i]
		hasWt1 := len(pr.Wt1) >= len(pr.Si)
		for si := range pr.Si {
			if hasWt1 {
				er := pj.SetSynValue("SWt", pr.Si[si], pr.Ri, pr.Wt1[si])
				if er != nil {
					err = er
				}
			}
			er := pj.SetSynValue("Wt", pr.Si[si], pr.Ri, pr.Wt[si]) // updates lin wt
			if er != nil {
				err = er
			}
		}
	}
	return err
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// SetSWtsRPool initializes SWt structural weight values using given tensor
// of values which has unique values for each recv neuron within a given pool.
func (pj *Path) SetSWtsRPool(ctx *Context, swts tensor.Tensor) {
	rNuY := swts.DimSize(0)
	rNuX := swts.DimSize(1)
	rNu := rNuY * rNuX
	rfsz := swts.Len() / rNu

	rsh := pj.Recv.Shape()
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
						ri = rsh.Offset([]int{ruy, rux})
					} else {
						ri = rsh.Offset([]int{rpy, rpx, ruy, rux})
					}
					scst := (ruy*rNuX + rux) * rfsz
					syIndexes := pj.RecvSynIndexes(uint32(ri))
					for ci, syi := range syIndexes {
						syni := pj.SynStIndex + syi
						swt := float32(swts.Float1D((scst + ci) % wsz))
						SetSynV(ctx, syni, SWt, float32(swt))
						wt := pj.Params.SWts.ClipWt(swt + (SynV(ctx, syni, Wt) - pj.Params.SWts.Init.Mean))
						SetSynV(ctx, syni, Wt, wt)
						SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFromWts(wt, swt))
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
// Strongly suggest calling SWtRescale after.
func (pj *Path) SetWtsFunc(ctx *Context, wtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		syIndexes := pj.RecvSynIndexes(uint32(ri))
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			si := pj.Params.SynSendLayIndex(ctx, syni)
			wt := wtFun(int(si), ri, ssh, rsh)
			SetSynV(ctx, syni, SWt, wt)
			SetSynV(ctx, syni, Wt, wt)
			SetSynV(ctx, syni, LWt, 0.5)
		}
	}
}

// SetSWtsFunc initializes structural SWt values using given function
// based on receiving and sending unit indexes.
func (pj *Path) SetSWtsFunc(ctx *Context, swtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		syIndexes := pj.RecvSynIndexes(uint32(ri))
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			si := int(pj.Params.SynSendLayIndex(ctx, syni))
			swt := swtFun(si, ri, ssh, rsh)
			SetSynV(ctx, syni, SWt, swt)
			wt := pj.Params.SWts.ClipWt(swt + (SynV(ctx, syni, Wt) - pj.Params.SWts.Init.Mean))
			SetSynV(ctx, syni, Wt, wt)
			SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFromWts(wt, swt))
		}
	}
}

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pj *Path) InitWtsSyn(ctx *Context, syni uint32, rnd randx.Rand, mean, spct float32) {
	pj.Params.SWts.InitWtsSyn(ctx, syni, rnd, mean, spct)
}

// InitWts initializes weight values according to SWt params,
// enforcing current constraints.
func (pj *Path) InitWts(ctx *Context, nt *Network) {
	pj.Params.Learn.LRate.Init()
	pj.InitGBuffs()
	rlay := pj.Recv
	spct := pj.Params.SWts.Init.SPct
	if rlay.Params.IsTarget() {
		pj.Params.SWts.Init.SPct = 0
		spct = 0
	}
	smn := pj.Params.SWts.Init.Mean
	// todo: why is this recv based?  prob important to keep for consistency
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ni := rlay.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		syIndexes := pj.RecvSynIndexes(lni)
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			pj.InitWtsSyn(ctx, syni, &nt.Rand, smn, spct)
		}
	}
	if pj.Params.SWts.Adapt.On.IsTrue() && !rlay.Params.IsTarget() {
		pj.SWtRescale(ctx)
	}
}

// SWtRescale rescales the SWt values to preserve the target overall mean value,
// using subtractive normalization.
func (pj *Path) SWtRescale(ctx *Context) {
	rlay := pj.Recv
	smn := pj.Params.SWts.Init.Mean
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ni := rlay.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		var nmin, nmax int
		var sum float32
		syIndexes := pj.RecvSynIndexes(lni)
		nCons := len(syIndexes)
		if nCons <= 1 {
			continue
		}
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			swt := SynV(ctx, syni, SWt)
			sum += swt
			if swt <= pj.Params.SWts.Limit.Min {
				nmin++
			} else if swt >= pj.Params.SWts.Limit.Max {
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
				syni := pj.SynStIndex + syi
				if SynV(ctx, syni, SWt) <= pj.Params.SWts.Limit.Max {
					swt := pj.Params.SWts.ClipSWt(SynV(ctx, syni, SWt) + mdf)
					SetSynV(ctx, syni, SWt, swt)
					SetSynV(ctx, syni, Wt, pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt)))
				}
			}
		} else {
			if nmin > 0 && nmin < nCons {
				amn = sum / float32(nCons-nmin)
				mdf = smn - amn
			}
			for _, syi := range syIndexes {
				syni := pj.SynStIndex + syi
				if SynV(ctx, syni, SWt) >= pj.Params.SWts.Limit.Min {
					swt := pj.Params.SWts.ClipSWt(SynV(ctx, syni, SWt) + mdf)
					SetSynV(ctx, syni, SWt, swt)
					SetSynV(ctx, syni, Wt, pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt)))
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
func (pj *Path) InitWtSym(ctx *Context, rpj *Path) {
	if len(rpj.SendCon) == 0 {
		return
	}
	slay := pj.Send
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pj.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			syni := pj.SynStIndex + syi
			ri := pj.Params.SynRecvLayIndex(ctx, syni) // <- this sends to me, ri
			recipSi := ri                              // reciprocal case is si is now receiver
			recipc := rpj.SendCon[recipSi]
			if recipc.N == 0 {
				continue
			}
			firstSyni := rpj.SynStIndex + recipc.Start
			lastSyni := rpj.SynStIndex + recipc.Start + recipc.N - 1
			firstRi := rpj.Params.SynRecvLayIndex(ctx, firstSyni)
			lastRi := rpj.Params.SynRecvLayIndex(ctx, lastSyni)
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
					recipRi := rpj.Params.SynRecvLayIndex(ctx, recipSyni)
					if recipRi == lni {
						SetSynV(ctx, recipSyni, Wt, SynV(ctx, syni, Wt))
						SetSynV(ctx, recipSyni, LWt, SynV(ctx, syni, LWt))
						SetSynV(ctx, recipSyni, SWt, SynV(ctx, syni, SWt))
						// note: if we support SymFromTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					recipCi := uint32(recipc.Start) + uint32(dn)
					recipSyni := rpj.SynStIndex + recipCi
					recipRi := rpj.Params.SynRecvLayIndex(ctx, recipSyni)
					if recipRi == lni {
						SetSynV(ctx, recipSyni, Wt, SynV(ctx, syni, Wt))
						SetSynV(ctx, recipSyni, LWt, SynV(ctx, syni, LWt))
						SetSynV(ctx, recipSyni, SWt, SynV(ctx, syni, SWt))
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
// This is not typically needed (called during InitWts, InitActs)
// but can be called when needed.  Must be called to completely initialize
// prior activity, e.g., full Glong clearing.
func (pj *Path) InitGBuffs() {
	for ri := range pj.GBuf {
		pj.GBuf[ri] = 0
	}
	for ri := range pj.GSyns {
		pj.GSyns[ri] = 0
	}
}
