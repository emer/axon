// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/c2h5oh/datasize"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/kit"
)

// KinaseRules are different options for Kinase-based learning rules
type KinaseRules int32

//go:generate stringer -type=KinaseRules

var KiT_KinaseRules = kit.Enums.AddEnum(KinaseRulesN, kit.NotBitFlag, nil)

func (ev KinaseRules) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *KinaseRules) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The time scales
const (
	// NeurSpkCa uses neuron-level spike-driven calcium signals
	// integrated at P vs. D time scales -- this is the original
	// Leabra and Axon XCAL / CHL learning rule.
	NeurSpkCa KinaseRules = iota

	// SynSpkCaOR uses synapse-level spike-driven calcium signals
	// with an OR rule for pre OR post spiking driving the CaM up,
	// which is then integrated at P vs. D time scales.
	// Basically a synapse version of original learning rule.
	SynSpkCaOR

	// SynSpkNMDAOR uses synapse-level spike-driven calcium signals
	// with an OR rule for pre OR post spiking driving the CaM up,
	// with NMDAo multiplying the spike drive to fit Bio Ca better
	// including the Bonus factor.
	// which is then integrated at P vs. D time scales.
	SynSpkNMDAOR

	// SynNMDACa uses synapse-level NMDA-driven calcium signals
	// (which can be either Urakubo allosteric or Kinase abstract)
	// integrated at P vs. D time scales -- abstract version
	// of the KinaseB biophysical learniung rule
	SynNMDACa

	KinaseRulesN
)

// KinaseSynParams has rate constants for averaging over activations
// at different time scales, to produce the running average activation
// values that then drive learning.
type KinaseSynParams struct {
	Rule     KinaseRules     `desc:"which learning rule to use"`
	SpikeG   float32         `def:"20,8,200" desc:"spiking gain for Spk rules"`
	MTau     float32         `def:"10,40" min:"1" desc:"CaM mean running-average time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life). This provides a pre-integration step before integrating into the CaP short time scale"`
	PTau     float32         `def:"40,10" min:"1" desc:"LTP Ca-driven factor time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life). Continuously updates based on current CaI value, resulting in faster tracking of plus-phase signals."`
	DTau     float32         `def:"40" min:"1" desc:"LTD Ca-driven factor time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life).  Continuously updates based on current CaP value, resulting in slower integration that still reflects earlier minus-phase signals."`
	DScale   float32         `def:"0.93,1.05" desc:"scaling factor on CaD as it enters into the learning rule, to compensate for systematic decrease in activity over the course of a theta cycle"`
	PFunt    etensor.Float32 `view:"-" desc:"P function table"`
	DFunt    etensor.Float32 `view:"-" desc:"D function table"`
	Tmax     int             `view:"-" desc:"maximum time in lookup tables"`
	Mmax     float32         `view:"-" desc:"maximum CaM value in lookup tables"`
	PDmax    float32         `view:"-" desc:"maximum CaP, CaD value in lookup tables"`
	Yres     float32         `view:"-" desc:"resolution of Y value in lookup tables"`
	FuntPars []float32       `view:"-" desc:"params the function tables were built for -- to prevent rebuilds"`

	MDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
}

func (kp *KinaseSynParams) Defaults() {
	kp.Rule = SynSpkCaOR
	kp.SpikeG = 8 // 200 // 8
	kp.MTau = 10  // 40
	kp.PTau = 40  // 10
	kp.DTau = 40
	kp.DScale = 1.05
	kp.Tmax = 100
	kp.Mmax = 2.8
	kp.PDmax = 1.8
	kp.Yres = 0.02
	kp.Update()
}

func (kp *KinaseSynParams) Update() {
	kp.MDt = 1 / kp.MTau
	kp.PDt = 1 / kp.PTau
	kp.DDt = 1 / kp.DTau
	kp.Funts()
}

// FmSpike computes updates from current spike value
func (kp *KinaseSynParams) FmSpike(spike float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (kp.SpikeG*spike - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// DWt computes the weight change from CaP, CaD values
func (kp *KinaseSynParams) DWt(caP, caD float32) float32 {
	return caP - kp.DScale*caD
}

// Funts builds the function tables
func (kp *KinaseSynParams) Funts() {
	pars := []float32{kp.SpikeG, kp.MTau, kp.PTau, kp.DTau}
	if len(kp.FuntPars) > 0 {
		diff := false
		for i := range pars {
			if pars[i] != kp.FuntPars[i] {
				diff = true
				break
			}
		}
		if !diff {
			return
		}
	}
	kp.FuntPars = pars

	mn := int(kp.Mmax/kp.Yres) + 1
	pn := int(kp.PDmax/kp.Yres) + 1
	kp.PFunt.SetShape([]int{pn, mn, kp.Tmax + 1}, nil, []string{"P", "M", "T"})
	fmt.Printf("PFunt has %d elements\n", kp.PFunt.Len())
	for pi := 0; pi < pn; pi++ {
		for mi := 0; mi < mn; mi++ {
			pv := float32(pi) * kp.Yres
			mv := float32(mi) * kp.Yres
			m := mv
			ps := pv
			for t := 0; t <= kp.Tmax; t++ {
				ps += kp.PDt * (m - ps)
				m -= kp.MDt * m
				kp.PFunt.Set([]int{pi, mi, t}, ps)
			}
		}
	}
	kp.DFunt.SetShape([]int{pn, pn, mn, kp.Tmax + 1}, nil, []string{"D", "P", "M", "T"})
	fmt.Printf("DFunt has %s elements\n", datasize.ByteSize(kp.DFunt.Len()).HumanReadable())
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
					kp.DFunt.Set([]int{di, pi, mi, t}, ds)
				}
			}
		}
	}
}

// TClip clips T value in valid range
func (kp *KinaseSynParams) TClip(t int) int {
	if t < 0 {
		t = 0
	}
	if t > kp.Tmax {
		t = kp.Tmax
	}
	return t
}

// MClip clips M value in valid range
func (kp *KinaseSynParams) MClip(v float32) float32 {
	if v > kp.Mmax-kp.Yres {
		v = kp.Mmax - kp.Yres
	}
	if v < 0 {
		v = 0
	}
	return v
}

// PDClip clips P or D value in valid range
func (kp *KinaseSynParams) PDClip(v float32) float32 {
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
func (kp *KinaseSynParams) PFmLastSpike(pv, mv float32, t int) float32 {
	pv = kp.PDClip(pv)
	mv = kp.MClip(mv)
	t = kp.TClip(t)
	pi := int(pv / kp.Yres)
	mi := int(mv / kp.Yres)
	y00 := kp.PFunt.Value([]int{pi, mi, t})
	// interpolate relative to next points up in p and m
	pr := (pv - (float32(pi) * kp.Yres)) / kp.Yres
	y10 := kp.PFunt.Value([]int{pi + 1, mi, t})
	y10d := pr * (y10 - y00)
	mr := (mv - (float32(mi) * kp.Yres)) / kp.Yres
	y01 := kp.PFunt.Value([]int{pi, mi + 1, t})
	y01d := mr * (y01 - y00)
	y11 := kp.PFunt.Value([]int{pi + 1, mi + 1, t})
	y11d := (0.5 * (mr + pr)) * (y11 - y00) // best would be sqrt(mr^2, pr^2)
	y := y00 + (y10d+y01d+y11d)/3
	return y
}

// DFmLastSpike returns D value for given D, P, M values at
// the last spike, and current ISI time since last spike
// using DFunt lookup table
func (kp *KinaseSynParams) DFmLastSpike(dv, pv, mv float32, t int) float32 {
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
				yv := kp.DFunt.Value([]int{di + do, pi + po, mi + mo, t})
				if ix > 0 {
					dsum += ((dr + pr + mr) / float32(do+po+mo)) * (yv - y0)
				} else {
					y0 = yv
				}
				ix++
			}
		}
	}
	// return y0
	return y0 + dsum/7
}
