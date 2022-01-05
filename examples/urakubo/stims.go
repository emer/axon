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
	CaTarg Stims = iota

	ClampCa1

	StimsN
)

// StimFuncs are the stimulus functions
var StimFuncs = map[Stims]func(){
	CaTarg:   CaTargFun,
	ClampCa1: ClampCa1Fun,
}

// ClampCa1Ca is direct copy of Ca values from test_stdp.g genesis func
var ClampCa1Ca = []float64{
	509.98792, 0.05731354654,
	509.99052, 1.800978422,
	509.99422, 3.778658628,
	509.99922, 3.385097265,
	510.00052, 3.192493439,
	510.00072, 3.175482512,
	510.00122, 3.484202623,
	510.00222, 9.131223679,
	510.00392, 7.309978008,
	510.00652, 3.479086161,
	510.01022, 2.207912683,
	510.01522, 1.591691375,
	510.02172, 1.139062405,
	510.02992, 0.8029100895,
	510.04002, 0.5597535968,
	510.05222, 0.3869290054,
	510.06672, 0.2666117251,
	510.08372, 0.1854287386,
	510.1034, 0.1331911981,
	510.126, 0.1012685224,
	510.1517, 0.08303561062,
	510.1807, 0.07339032739,
	510.2132, 0.06863268465,
	510.2494, 0.06637191772,
	510.2895, 0.06523273885,
	510.3337, 0.06451437622,
	510.3822, 0.06390291452,
	510.4352, 0.06327681988,
	510.4929, 0.062598221,
	510.5555, 0.06186179072,
	510.6232, 0.06107418239,
	510.6962, 0.06024680287,
	510.7747, 0.05939326808,
}

// InitSettle settles out the spine for 200 secs
func InitSettle() {
	ss := &TheSim
	ss.StopNow = false
	ss.Spine.StepTime(100)
}

// CaPerMsec returns calcium per msec for given input data
func CaPerMsec(orig []float64) []float64 {
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

func CaTargFun() {
	ss := &TheSim
	InitSettle()
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
	cas := CaPerMsec(ClampCa1Ca)
	nca := len(cas)
	InitSettle()
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
	ss.Stopped()
}
