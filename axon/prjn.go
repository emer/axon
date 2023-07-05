// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"io"
	"strconv"

	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// https://github.com/kisvegabor/abbreviations-in-code suggests Buf instead of Buff

// index naming:
// syi =  prjn-relative synapse index (per existing usage)
// syni = network-relative synapse index -- add SynStIdx to syi

// axon.Prjn is a basic Axon projection with synaptic learning parameters
type Prjn struct {
	PrjnBase
	Params *PrjnParams `desc:"all prjn-level parameters -- these must remain constant once configured"`
}

var KiT_Prjn = kit.Types.AddType(&Prjn{}, PrjnProps)

// Object returns the object with parameters to be set by emer.Params
func (pj *Prjn) Object() any {
	return pj.Params
}

// AsAxon returns this prjn as a axon.Prjn -- all derived prjns must redefine
// this to return the base Prjn type, so that the AxonPrjn interface does not
// need to include accessors to all the basic stuff.
func (pj *Prjn) AsAxon() *Prjn {
	return pj
}

// PrjnType returns axon specific cast of pj.Typ prjn type
func (pj *Prjn) PrjnType() PrjnTypes {
	return PrjnTypes(pj.Typ)
}

func (pj *Prjn) Class() string {
	return pj.PrjnType().String() + " " + pj.Cls
}

func (pj *Prjn) Defaults() {
	if pj.Params == nil {
		return
	}
	pj.Params.PrjnType = pj.PrjnType()
	pj.Params.Defaults()
	switch pj.PrjnType() {
	case InhibPrjn:
		pj.Params.SWts.Adapt.On.SetBool(false)
	case BackPrjn:
		pj.Params.PrjnScale.Rel = 0.1
	case RWPrjn, TDPredPrjn:
		pj.Params.RLPredDefaults()
	case BLAPrjn:
		pj.Params.BLADefaults()
	case VSPatchPrjn:
		pj.Params.VSPatchDefaults()
	case MatrixPrjn:
		pj.Params.MatrixDefaults()
	}
	pj.ApplyDefParams()
	pj.UpdateParams()
}

// Update is interface that does local update of struct vals
func (pj *Prjn) Update() {
	if pj.Params == nil {
		return
	}
	pj.Params.Update()
}

// UpdateParams updates all params given any changes
// that might have been made to individual values
func (pj *Prjn) UpdateParams() {
	pj.Update()
}

// AllParams returns a listing of all parameters in the Layer
func (pj *Prjn) AllParams() string {
	str := "///////////////////////////////////////////////////\nPrjn: " + pj.Name() + "\n" + pj.Params.AllParams()
	return str
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pj *Prjn) SetSynVal(varNm string, sidx, ridx int, val float32) error {
	vidx, err := pj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	syi := uint32(pj.SynIdx(sidx, ridx))
	if syi < 0 || syi >= pj.NSyns {
		return err
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIdx + syi
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
		SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFmWts(wt, SynV(ctx, syni, SWt)))
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWtsJSON writes the weights from this projection from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pj *Prjn) WriteWtsJSON(w io.Writer, depth int) {
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
		syIdxs := pj.RecvSynIdxs(uint32(ri))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("{\n"))
		depth++
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"Ri\": %v,\n", ri)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"N\": %v,\n", rc.N)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Si\": [ "))
		for ci, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			si := pj.Params.SynSendLayIdx(ctx, syni)
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
		for ci, syi := range syIdxs {
			syni := pj.SynStIdx + syi
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
		for ci, syi := range syIdxs {
			syni := pj.SynStIdx + syi
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

// ReadWtsJSON reads the weights from this projection from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one prjn only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (pj *Prjn) ReadWtsJSON(r io.Reader) error {
	pw, err := weights.PrjnReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	return pj.SetWts(pw)
}

// SetWts sets the weights for this projection from weights.Prjn decoded values
func (pj *Prjn) SetWts(pw *weights.Prjn) error {
	var err error
	for i := range pw.Rs {
		pr := &pw.Rs[i]
		hasWt1 := len(pr.Wt1) >= len(pr.Si)
		for si := range pr.Si {
			if hasWt1 {
				er := pj.SetSynVal("SWt", pr.Si[si], pr.Ri, pr.Wt1[si])
				if er != nil {
					err = er
				}
			}
			er := pj.SetSynVal("Wt", pr.Si[si], pr.Ri, pr.Wt[si]) // updates lin wt
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
func (pj *Prjn) SetSWtsRPool(ctx *Context, swts etensor.Tensor) {
	rNuY := swts.Dim(0)
	rNuX := swts.Dim(1)
	rNu := rNuY * rNuX
	rfsz := swts.Len() / rNu

	rsh := pj.Recv.Shape()
	rNpY := rsh.Dim(0)
	rNpX := rsh.Dim(1)
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
					syIdxs := pj.RecvSynIdxs(uint32(ri))
					for ci, syi := range syIdxs {
						syni := pj.SynStIdx + syi
						swt := float32(swts.FloatVal1D((scst + ci) % wsz))
						SetSynV(ctx, syni, SWt, float32(swt))
						wt := pj.Params.SWts.ClipWt(swt + (SynV(ctx, syni, Wt) - pj.Params.SWts.Init.Mean))
						SetSynV(ctx, syni, Wt, wt)
						SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFmWts(wt, swt))
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
// Strongly suggest calling SWtRescale after.
func (pj *Prjn) SetWtsFunc(ctx *Context, wtFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		syIdxs := pj.RecvSynIdxs(uint32(ri))
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			si := pj.Params.SynSendLayIdx(ctx, syni)
			wt := wtFun(int(si), ri, ssh, rsh)
			SetSynV(ctx, syni, SWt, wt)
			SetSynV(ctx, syni, Wt, wt)
			SetSynV(ctx, syni, LWt, 0.5)
		}
	}
}

// SetSWtsFunc initializes structural SWt values using given function
// based on receiving and sending unit indexes.
func (pj *Prjn) SetSWtsFunc(ctx *Context, swtFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		syIdxs := pj.RecvSynIdxs(uint32(ri))
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			si := int(pj.Params.SynSendLayIdx(ctx, syni))
			swt := swtFun(si, ri, ssh, rsh)
			SetSynV(ctx, syni, SWt, swt)
			wt := pj.Params.SWts.ClipWt(swt + (SynV(ctx, syni, Wt) - pj.Params.SWts.Init.Mean))
			SetSynV(ctx, syni, Wt, wt)
			SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFmWts(wt, swt))
		}
	}
}

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pj *Prjn) InitWtsSyn(ctx *Context, syni uint32, rnd erand.Rand, mean, spct float32) {
	pj.Params.SWts.InitWtsSyn(ctx, syni, rnd, mean, spct)
}

// InitSynCa initializes synaptic calcium variables
func (pj *Prjn) InitSynCa(ctx *Context, syni, di uint32) {
	InitSynCa(ctx, syni, di)
}

// InitWts initializes weight values according to SWt params,
// enforcing current constraints.
func (pj *Prjn) InitWts(ctx *Context, nt *Network) {
	if pj.Typ == InhibPrjn {
		pj.Params.Com.GType = InhibitoryG
	}
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
		ni := rlay.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		syIdxs := pj.RecvSynIdxs(lni)
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			pj.InitWtsSyn(ctx, syni, &nt.Rand, smn, spct)
			for di := uint32(0); di < rlay.MaxData; di++ {
				pj.InitSynCa(ctx, syni, di)
			}
		}
	}
	if pj.Params.SWts.Adapt.On.IsTrue() && !rlay.Params.IsTarget() {
		pj.SWtRescale(ctx)
	}
}

// SWtRescale rescales the SWt values to preserve the target overall mean value,
// using subtractive normalization.
func (pj *Prjn) SWtRescale(ctx *Context) {
	rlay := pj.Recv
	smn := pj.Params.SWts.Init.Mean
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ni := rlay.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		var nmin, nmax int
		var sum float32
		syIdxs := pj.RecvSynIdxs(lni)
		nCons := len(syIdxs)
		if nCons <= 1 {
			continue
		}
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
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
			for _, syi := range syIdxs {
				syni := pj.SynStIdx + syi
				if SynV(ctx, syni, SWt) <= pj.Params.SWts.Limit.Max {
					swt := pj.Params.SWts.ClipSWt(SynV(ctx, syni, SWt) + mdf)
					SetSynV(ctx, syni, SWt, swt)
					SetSynV(ctx, syni, Wt, pj.Params.SWts.WtVal(swt, SynV(ctx, syni, LWt)))
				}
			}
		} else {
			if nmin > 0 && nmin < nCons {
				amn = sum / float32(nCons-nmin)
				mdf = smn - amn
			}
			for _, syi := range syIdxs {
				syni := pj.SynStIdx + syi
				if SynV(ctx, syni, SWt) >= pj.Params.SWts.Limit.Min {
					swt := pj.Params.SWts.ClipSWt(SynV(ctx, syni, SWt) + mdf)
					SetSynV(ctx, syni, SWt, swt)
					SetSynV(ctx, syni, Wt, pj.Params.SWts.WtVal(swt, SynV(ctx, syni, LWt)))
				}
			}
		}
	}
}

// sender based version:
//                     only looked at recv layers > send layers -- ri is "above" si
//  for:      ri    <- this is now sender on recip prjn: recipSi
//           ^ \    <- look for send back to original si, now as a receiver
//          /   v
// start: si == recipRi <- look in sy.RecvIdx of recipSi's sending cons for recipRi == si
//

// InitWtSym initializes weight symmetry.
// Is given the reciprocal projection where
// the Send and Recv layers are reversed
// (see LayerBase RecipToRecvPrjn)
func (pj *Prjn) InitWtSym(ctx *Context, rpj *Prjn) {
	if len(rpj.SendCon) == 0 {
		return
	}
	slay := pj.Send
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pj.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			syni := pj.SynStIdx + syi
			ri := pj.Params.SynRecvLayIdx(ctx, syni) // <- this sends to me, ri
			recipSi := ri                            // reciprocal case is si is now receiver
			recipc := rpj.SendCon[recipSi]
			if recipc.N == 0 {
				continue
			}
			firstSyni := rpj.SynStIdx + recipc.Start
			lastSyni := rpj.SynStIdx + recipc.Start + recipc.N - 1
			firstRi := rpj.Params.SynRecvLayIdx(ctx, firstSyni)
			lastRi := rpj.Params.SynRecvLayIdx(ctx, lastSyni)
			if lni < firstRi || lni > lastRi { // fast reject -- prjns are always in order!
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
					recipSyni := rpj.SynStIdx + recipCi
					recipRi := rpj.Params.SynRecvLayIdx(ctx, recipSyni)
					if recipRi == lni {
						SetSynV(ctx, recipSyni, Wt, SynV(ctx, syni, Wt))
						SetSynV(ctx, recipSyni, LWt, SynV(ctx, syni, LWt))
						SetSynV(ctx, recipSyni, SWt, SynV(ctx, syni, SWt))
						// note: if we support SymFmTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					recipCi := uint32(recipc.Start) + uint32(dn)
					recipSyni := rpj.SynStIdx + recipCi
					recipRi := rpj.Params.SynRecvLayIdx(ctx, recipSyni)
					if recipRi == lni {
						SetSynV(ctx, recipSyni, Wt, SynV(ctx, syni, Wt))
						SetSynV(ctx, recipSyni, LWt, SynV(ctx, syni, LWt))
						SetSynV(ctx, recipSyni, SWt, SynV(ctx, syni, SWt))
						// note: if we support SymFmTop then can have option to go other way
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

// InitGBuffs initializes the per-projection synaptic conductance buffers.
// This is not typically needed (called during InitWts, InitActs)
// but can be called when needed.  Must be called to completely initialize
// prior activity, e.g., full Glong clearing.
func (pj *Prjn) InitGBuffs() {
	for ri := range pj.GBuf {
		pj.GBuf[ri] = 0
	}
	for ri := range pj.GSyns {
		pj.GSyns[ri] = 0
	}
}

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnTypes, // uses our PrjnTypes for GUI
}
