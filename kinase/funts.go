// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"

	"github.com/c2h5oh/datasize"
	"github.com/emer/etable/etensor"
)

// Funts are function tables for CaP and CaD integration levels
// allowing integration timecourse of CaP and CaD variables
// driven by CaM, only when there is a spike.
type Funts struct {
	CaP etensor.Float32 `view:"-" desc:"CaP timescale function table"`
	CaD etensor.Float32 `view:"-" desc:"CaD timescale function table"`
}

// FuntMap is a map of rendered Funts for different parameters.
// The parameters are printed as a string for the key.
// Many cases will use the same
var FuntMap map[string]*Funts

// ParamString returns the string for given parameters, space separated
func ParamString(params []float32) string {
	pstr := ""
	for _, p := range params {
		pstr += fmt.Sprintf("%g ", p)
	}
	return pstr
}

// FuntsForParams returns function tables for given parameters.
// Returns true if already existed, false if newly created.
func FuntsForParams(paramkey string) (*Funts, bool) {
	if FuntMap == nil {
		FuntMap = make(map[string]*Funts)
	}
	ft, ok := FuntMap[paramkey]
	if ok {
		return ft, ok
	}
	ft = &Funts{}
	FuntMap[paramkey] = ft
	return ft, false
}

////////////////////////////////////////////////////////////////////
// build tables

// Funts builds the function tables, returns true if already built
// and false if not and needed to be built for current params.
func (kp *SynParams) Funts() bool {
	pars := []float32{kp.SpikeG, kp.MTau, kp.PTau, kp.DTau}
	kp.FuntKey = ParamString(pars)
	ft, ok := FuntsForParams(kp.FuntKey)
	if ok {
		return true
	}

	mn := int(kp.Mmax/kp.Yres) + 1
	pn := int(kp.PDmax/kp.Yres) + 1
	ft.CaP.SetShape([]int{pn, mn, kp.Tmax + 1}, nil, []string{"P", "M", "T"})
	fmt.Printf("PFunt has %d elements\n", ft.CaP.Len())
	for pi := 0; pi < pn; pi++ {
		for mi := 0; mi < mn; mi++ {
			pv := float32(pi) * kp.Yres
			mv := float32(mi) * kp.Yres
			m := mv
			ps := pv
			for t := 0; t <= kp.Tmax; t++ {
				ps += kp.PDt * (m - ps)
				m -= kp.MDt * m
				ft.CaP.Set([]int{pi, mi, t}, ps)
			}
		}
	}
	ft.CaD.SetShape([]int{pn, pn, mn, kp.Tmax + 1}, nil, []string{"D", "P", "M", "T"})
	fmt.Printf("DFunt has %s elements\n", datasize.ByteSize(ft.CaD.Len()).HumanReadable())
	for di := 0; di < pn; di++ {
		for pi := 0; pi < pn; pi++ {
			for mi := 0; mi < mn; mi++ {
				dv := float32(di) * kp.Yres
				pv := float32(pi) * kp.Yres
				mv := float32(mi) * kp.Yres
				m := mv
				ds := dv
				ps := pv
				for t := 0; t <= kp.Tmax; t++ {
					ps += kp.PDt * (m - ps)
					ds += kp.DDt * (ps - ds)
					m -= kp.MDt * m
					ft.CaD.Set([]int{di, pi, mi, t}, ds)
				}
			}
		}
	}
	return false
}

// TClip clips T value in valid range
func (kp *SynParams) TClip(t int) int {
	if t < 0 {
		t = 0
	}
	if t > kp.Tmax {
		t = kp.Tmax
	}
	return t
}

// MClip clips M value in valid range
func (kp *SynParams) MClip(v float32) float32 {
	if v > kp.Mmax-kp.Yres {
		v = kp.Mmax - kp.Yres
	}
	if v < 0 {
		v = 0
	}
	return v
}

// PDClip clips P or D value in valid range
func (kp *SynParams) PDClip(v float32) float32 {
	if v > kp.PDmax-kp.Yres {
		v = kp.PDmax - kp.Yres
	}
	if v < 0 {
		v = 0
	}
	return v
}

// PFmLastSpike returns P value for given P, M values at
// the last spike, and current ISI time since last spike
// using PFunt lookup table
func (kp *SynParams) PFmLastSpike(pv, mv float32, t int) float32 {
	ft, ok := FuntsForParams(kp.FuntKey)
	if !ok { // will always exist due to Update() logic
		return 0
	}
	pv = kp.PDClip(pv)
	mv = kp.MClip(mv)
	t = kp.TClip(t)
	pi := int(pv / kp.Yres)
	mi := int(mv / kp.Yres)
	y00 := ft.CaP.Value([]int{pi, mi, t})
	// interpolate relative to next points up in p and m
	pr := (pv - (float32(pi) * kp.Yres)) / kp.Yres
	y10 := ft.CaP.Value([]int{pi + 1, mi, t})
	y10d := pr * (y10 - y00)
	mr := (mv - (float32(mi) * kp.Yres)) / kp.Yres
	y01 := ft.CaP.Value([]int{pi, mi + 1, t})
	y01d := mr * (y01 - y00)
	y11 := ft.CaP.Value([]int{pi + 1, mi + 1, t})
	y11d := (0.5 * (mr + pr)) * (y11 - y00) // best would be sqrt(mr^2, pr^2)
	y := y00 + (y10d+y01d+y11d)/3
	return y
}

// DFmLastSpike returns D value for given D, P, M values at
// the last spike, and current ISI time since last spike
// using DFunt lookup table
func (kp *SynParams) DFmLastSpike(dv, pv, mv float32, t int) float32 {
	ft, ok := FuntsForParams(kp.FuntKey)
	if !ok { // will always exist due to Update() logic
		return 0
	}
	dv = kp.PDClip(dv)
	pv = kp.PDClip(pv)
	mv = kp.MClip(mv)
	t = kp.TClip(t)
	di := int(dv / kp.Yres)
	pi := int(pv / kp.Yres)
	mi := int(mv / kp.Yres)
	var y0, dsum float32
	// interpolate relative to next points up in p and m
	ix := 0
	for do := 0; do < 2; do++ {
		dr := (dv - (float32(di) * kp.Yres)) / kp.Yres
		for po := 0; po < 2; po++ {
			pr := (pv - (float32(pi) * kp.Yres)) / kp.Yres
			for mo := 0; mo < 2; mo++ {
				mr := (mv - (float32(mi) * kp.Yres)) / kp.Yres
				yv := ft.CaD.Value([]int{di + do, pi + po, mi + mo, t})
				if ix > 0 {
					dsum += ((dr + pr + mr) / float32(do+po+mo)) * (yv - y0)
				} else {
					y0 = yv
				}
				ix++
			}
		}
	}
	return y0 + dsum/7
}
