// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/goki/ki/kit"
)

type Stims int32

//go:generate stringer -type=Stims

var KiT_Stims = kit.Enums.AddEnum(StimsN, kit.NotBitFlag, nil)

func (ev Stims) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Stims) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The different stimulus functions
const (
	Baseline Stims = iota

	CaTarg

	ClampCa1

	STDP

	StimsN
)

// StimFuncs are the stimulus functions
var StimFuncs = map[Stims]func(){
	Baseline: BaselineFun,
	CaTarg:   CaTargFun,
	ClampCa1: ClampCa1Fun,
	STDP:     STDPFun,
}

// ClampCa1Ca is direct copy of Ca values from test_stdp.g genesis func
var ClampCa1Ca = []float64{
	509.987, 0.05731354654,
	509.990, 1.800978422,
	509.994, 3.778658628,
	509.999, 3.385097265,
	510.000, 3.192493439,
	510.001, 3.484202623,
	510.002, 9.131223679,
	510.003, 7.309978008,
	510.006, 3.479086161,
	510.010, 2.207912683,
	510.015, 1.591691375,
	510.021, 1.139062405,
	510.029, 0.8029100895,
	510.040, 0.5597535968,
	510.052, 0.3869290054,
	510.066, 0.2666117251,
	510.083, 0.1854287386,
	510.103, 0.1331911981,
	510.126, 0.1012685224,
	510.151, 0.08303561062,
	510.180, 0.07339032739,
	510.213, 0.06863268465,
	510.249, 0.06637191772,
	510.289, 0.06523273885,
	510.333, 0.06451437622,
	510.382, 0.06390291452,
	510.435, 0.06327681988,
	510.492, 0.062598221,
	510.555, 0.06186179072,
	510.623, 0.06107418239,
	510.696, 0.06024680287,
	510.774, 0.05939326808,
}

var ClampVm = []float64{
	16.990, -64.62194824,
	16.994, -63.22241211,
	16.999, -62.93297958,
	17.001, -62.99502563,
	17.002, -58.25670624,
	17.003, -4.396664619,
	17.004, -40.63706589,
	17.007, -55.61388779,
	17.010, -60.13838196,
	17.015, -61.69696426,
	17.022, -62.90755844,
	17.030, -63.82915497,
	17.040, -64.40870667,
}

// PerMsec returns per msec for given input data
func PerMsec(orig []float64) []float64 {
	ost := orig[0]
	nca := len(orig) / 2
	oet := orig[(nca-1)*2]
	dur := oet - ost
	dms := int(dur / 0.001)
	rdt := make([]float64, dms)
	si := 0
	mxi := 0
	for i := 0; i < dms; i++ {
		ct := ost + float64(i)*0.001
		st := orig[si*2]
		et := orig[(si+1)*2]
		sca := orig[si*2+1]
		eca := orig[(si+1)*2+1]
		if ct > et {
			si++
			if si >= nca-1 {
				break
			}
			st = orig[si*2]
			et = orig[(si+1)*2]
			sca = orig[si*2+1]
			eca = orig[(si+1)*2+1]
		}
		mxi = i
		pt := (ct - st) / (et - st)
		ca := sca + pt*(eca-sca)
		rdt[i] = ca
		// fmt.Printf("%d \tct:  %g  \tca:  %g  \tst:  %g  \tet:  %g  \tsca:  %g \teca:  %g\n", i, ct, ca, st, et, sca, eca)
	}
	return rdt[:mxi+1]
}

func BaselineFun() {
	ss := &TheSim
	for msec := 0; msec < 500000; msec++ { // 500000 = 500 sec for full baseline
		ss.NeuronUpdt(msec)
		ss.LogDefault()
		if ss.StopNow {
			break
		}
	}
	ss.Spine.InitCode()
	ss.Stopped()
}

func CaTargFun() {
	ss := &TheSim
	ss.Spine.Ca.SetBuffTarg(ss.CaTarg.Cyt, ss.CaTarg.PSD)
	for msec := 0; msec < 20000; msec++ {
		ss.NeuronUpdt(msec)
		ss.LogDefault()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func ClampCa1Fun() {
	ss := &TheSim
	cas := PerMsec(ClampCa1Ca)
	nca := len(cas)
	bca := 0.05
	for msec := 0; msec < 20000; msec++ {
		tms := (msec + 500) % 1000
		ca := bca
		if tms < nca {
			ca = cas[tms]
		}
		cca := bca + ((ca - bca) / 3)
		ss.Spine.Ca.SetClamp(cca, ca)
		ss.NeuronUpdt(msec)
		ss.LogDefault()
		if ss.StopNow {
			break
		}
	}
	ss.GraphRun(2)
	ss.Stopped()
}

func STDPFun() {
	ss := &TheSim
	vms := PerMsec(ClampVm)
	nvm := len(vms)
	bvm := -65.0
	peakT := 13 // offset in ClampVm for peak
	toff := 500
	vmoff := toff - peakT // peak hits at toff exactly
	psms := toff - ss.DeltaT
	tott := 1 * 1000
	for msec := 0; msec < tott; msec++ {
		ims := msec % 1000
		vmms := ims - vmoff
		vm := bvm
		if vmms >= 0 && vmms < nvm {
			vm = vms[vmms]
		}
		if ims == psms {
			ss.Spine.States.PreSpike = 1
		} else {
			ss.Spine.States.PreSpike = 0
		}
		ss.Spine.States.VmS = vm
		ss.NeuronUpdt(msec)
		ss.LogDefault()
		if ss.StopNow {
			break
		}
	}
	// ss.GraphRun(2)
	ss.Stopped()
}
