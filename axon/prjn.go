// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"io"
	"strconv"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// https://github.com/kisvegabor/abbreviations-in-code suggests Buf instead of Buff

// axon.Prjn is a basic Axon projection with synaptic learning parameters
type Prjn struct {
	PrjnBase
	Params *PrjnParams `desc:"all prjn-level parameters -- these must remain constant once configured"`
	Vals   *PrjnVals   `view:"-" desc:"projection state values updated during computation"`
}

var KiT_Prjn = kit.Types.AddType(&Prjn{}, PrjnProps)

// Object returns the object with parameters to be set by emer.Params
func (pj *Prjn) Object() interface{} {
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
		pj.Params.SWt.Adapt.On.SetBool(false)
	case RWPrjn, TDPredPrjn:
		pj.Params.RLPredPrjnDefaults()
	}
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
	vidx, err := pj.AxonPrj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	synIdx := pj.SynIdx(sidx, ridx)
	if synIdx < 0 || synIdx >= len(pj.Syns) {
		return err
	}
	sy := &pj.Syns[synIdx]
	sy.SetVarByIndex(vidx, val)
	if varNm == "Wt" {
		if sy.SWt == 0 {
			sy.SWt = sy.Wt
		}
		sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWtsJSON writes the weights from this projection from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pj *Prjn) WriteWtsJSON(w io.Writer, depth int) {
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	nr := len(rlay.Neurons)
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"From\": %q,\n", slay.Name())))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GScale\": \"%g\"\n", pj.Params.GScale.Scale)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	// w.Write(indent.TabBytes(depth))
	// w.Write([]byte(fmt.Sprintf("\"MetaVals\": {\n")))
	// depth++
	// w.Write(indent.TabBytes(depth))
	// w.Write([]byte(fmt.Sprintf("\"SWtMeans\": [ ")))
	// nn := len(pj.Params.SWtMeans)
	// for ni := range pj.Params.SWtMeans {
	// 	w.Write([]byte(fmt.Sprintf("%g", pj.Params.SWtMeans[ni])))
	// 	if ni < nn-1 {
	// 		w.Write([]byte(", "))
	// 	}
	// }
	// w.Write([]byte(" ]\n"))
	// depth--
	// w.Write(indent.TabBytes(depth))
	// w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Rs\": [\n")))
	depth++
	for ri := 0; ri < nr; ri++ {
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("{\n"))
		depth++
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"Ri\": %v,\n", ri)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"N\": %v,\n", nc)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Si\": [ "))
		for ci := 0; ci < nc; ci++ {
			si := pj.RecvConIdx[st+ci]
			w.Write([]byte(fmt.Sprintf("%v", si)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt\": [ "))
		for ci := 0; ci < nc; ci++ {
			rsi := pj.RecvSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			w.Write([]byte(strconv.FormatFloat(float64(sy.Wt), 'g', weights.Prec, 32)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt1\": [ ")) // Wt1 is SWt
		for ci := 0; ci < nc; ci++ {
			rsi := pj.RecvSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			w.Write([]byte(strconv.FormatFloat(float64(sy.SWt), 'g', weights.Prec, 32)))
			if ci == nc-1 {
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
	if pw.MetaData != nil {
		if gs, ok := pw.MetaData["GScale"]; ok {
			pv, _ := strconv.ParseFloat(gs, 32)
			pj.Params.GScale.Scale = float32(pv)
		}
	}
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

// BuildGBuf builds GBuf with current Com Delay values, if not correct size
func (pj *Prjn) BuildGBuffs() {
	rlen := uint32(pj.Recv.Shape().Len())
	dl := pj.Params.Com.Delay + 1
	gblen := dl * rlen
	if pj.Vals.Gidx.Len == dl && uint32(len(pj.GBuf)) == gblen {
		return
	}
	pj.Vals.Gidx.Len = dl
	pj.Vals.Gidx.Zi = 0
	pj.GBuf = make([]float32, gblen)
	rlay := pj.Recv.(AxonLayer).AsAxon()
	npools := len(rlay.Pools)
	pj.PIBuf = make([]float32, int(dl)*npools)
}

// SetSWtsRPool initializes SWt structural weight values using given tensor
// of values which has unique values for each recv neuron within a given pool.
func (pj *Prjn) SetSWtsRPool(swts etensor.Tensor) {
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
					nc := int(pj.RecvConN[ri])
					st := int(pj.RecvConIdxStart[ri])
					for ci := 0; ci < nc; ci++ {
						// si := int(pj.RecvConIdx[st+ci]) // could verify coords etc
						rsi := pj.RecvSynIdx[st+ci]
						sy := &pj.Syns[rsi]
						swt := swts.FloatVal1D((scst + ci) % wsz)
						sy.SWt = float32(swt)
						sy.Wt = pj.Params.SWt.ClipWt(sy.SWt + (sy.Wt - pj.Params.SWt.Init.Mean))
						sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
// Strongly suggest calling SWtRescale after.
func (pj *Prjn) SetWtsFunc(wtFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RecvConIdx[st+ci])
			rsi := pj.RecvSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			wt := wtFun(si, ri, ssh, rsh)
			sy.SWt = wt
			sy.Wt = wt
			sy.LWt = 0.5
		}
	}
}

// SetSWtsFunc initializes structural SWt values using given function
// based on receiving and sending unit indexes.
func (pj *Prjn) SetSWtsFunc(swtFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RecvConIdx[st+ci])
			swt := swtFun(si, ri, ssh, rsh)
			rsi := pj.RecvSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			sy.SWt = swt
			sy.Wt = pj.Params.SWt.ClipWt(sy.SWt + (sy.Wt - pj.Params.SWt.Init.Mean))
			sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
		}
	}
}

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pj *Prjn) InitWtsSyn(sy *Synapse, mean, spct float32) {
	pj.Params.SWt.InitWtsSyn(sy, mean, spct)
}

// InitWts initializes weight values according to SWt params,
// enforcing current constraints.
func (pj *Prjn) InitWts() {
	pj.Params.Com.Inhib.SetBool(pj.Typ == emer.Inhib)
	pj.Params.Learn.LRate.Init()
	pj.AxonPrj.InitGBuffs()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	spct := pj.Params.SWt.Init.SPct
	if rlay.AxonLay.IsTarget() {
		pj.Params.SWt.Init.SPct = 0
		spct = 0
	}
	smn := pj.Params.SWt.Init.Mean
	for ri := range rlay.Neurons {
		nrn := &rlay.Neurons[ri]
		if nrn.IsOff() {
			continue
		}
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]
		for _, rsi := range rsidxs {
			sy := &pj.Syns[rsi]
			pj.InitWtsSyn(sy, smn, spct)
		}
	}
	if pj.Params.SWt.Adapt.On.IsTrue() && !rlay.AxonLay.IsTarget() {
		pj.SWtRescale()
	}
}

// SWtRescale rescales the SWt values to preserve the target overall mean value,
// using subtractive normalization.
func (pj *Prjn) SWtRescale() {
	rlay := pj.Recv.(AxonLayer).AsAxon()
	smn := pj.Params.SWt.Init.Mean
	for ri := range rlay.Neurons {
		nrn := &rlay.Neurons[ri]
		if nrn.IsOff() {
			continue
		}
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]

		var nmin, nmax int
		var sum float32
		for _, rsi := range rsidxs {
			swt := pj.Syns[rsi].SWt
			sum += swt
			if swt <= pj.Params.SWt.Limit.Min {
				nmin++
			} else if swt >= pj.Params.SWt.Limit.Max {
				nmax++
			}
		}
		if nc <= 1 {
			continue
		}
		amn := sum / float32(nc)
		mdf := smn - amn // subtractive
		if mdf == 0 {
			continue
		}
		if mdf > 0 { // need to increase
			if nmax > 0 && nmax < nc {
				amn = sum / float32(nc-nmax)
				mdf = smn - amn
			}
			for _, rsi := range rsidxs {
				sy := &pj.Syns[rsi]
				if sy.SWt <= pj.Params.SWt.Limit.Max {
					sy.SWt = pj.Params.SWt.ClipSWt(sy.SWt + mdf)
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
			}
		} else {
			if nmin > 0 && nmin < nc {
				amn = sum / float32(nc-nmin)
				mdf = smn - amn
			}
			for _, rsi := range rsidxs {
				sy := &pj.Syns[rsi]
				if sy.SWt >= pj.Params.SWt.Limit.Min {
					sy.SWt = pj.Params.SWt.ClipSWt(sy.SWt + mdf)
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
			}
		}
	}
}

// InitWtSym initializes weight symmetry -- is given the reciprocal projection where
// the Send and Recv layers are reversed.
func (pj *Prjn) InitWtSym(rpjp AxonPrjn) {
	rpj := rpjp.AsAxon()
	slay := pj.Send.(AxonLayer).AsAxon()
	ns := uint32(len(slay.Neurons))
	for si := uint32(0); si < ns; si++ {
		nc := pj.SendConN[si]
		st := pj.SendConIdxStart[si]
		for ci := uint32(0); ci < nc; ci++ {
			sy := &pj.Syns[st+ci]
			ri := pj.SendConIdx[st+ci]
			// now we need to find the reciprocal synapse on rpj!
			// look in ri for sending connections
			rsi := ri
			if len(rpj.SendConN) == 0 {
				continue
			}
			rsnc := rpj.SendConN[rsi]
			if rsnc == 0 {
				continue
			}
			rsst := rpj.SendConIdxStart[rsi]
			rist := rpj.SendConIdx[rsst]        // starting index in recv prjn
			ried := rpj.SendConIdx[rsst+rsnc-1] // ending index
			if si < rist || si > ried {         // fast reject -- prjns are always in order!
				continue
			}
			// start at index proportional to si relative to rist
			up := uint32(0)
			if ried > rist {
				up = uint32(float32(rsnc) * float32(si-rist) / float32(ried-rist))
			}
			dn := up - 1

			for {
				doing := false
				if up < rsnc {
					doing = true
					rrii := rsst + up
					rri := rpj.SendConIdx[rrii]
					if rri == si {
						rsy := &rpj.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.SWt = sy.SWt
						// note: if we support SymFmTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					rrii := rsst + dn
					rri := rpj.SendConIdx[rrii]
					if rri == si {
						rsy := &rpj.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.SWt = sy.SWt
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
	pj.BuildGBuffs() // make sure correct size based on Com.Delay setting
	for ri := range pj.GBuf {
		pj.GBuf[ri] = 0
	}
	for ri := range pj.GVals {
		pj.GVals[ri].Init()
	}
	for pi := range pj.PIBuf {
		pj.PIBuf[pi] = 0
	}
}

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnTypes, // uses our PrjnTypes for GUI
}
