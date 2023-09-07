// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

import "sort"

var RunNames []string

func init() {
	RunNames = make([]string, len(AllRuns))
	idx := 0
	for nm := range AllRuns {
		RunNames[idx] = nm
		idx++
	}
	sort.Strings(RunNames)
}

var AllRuns = map[string]*Run{
	"PosAcq_A100": {
		Name:  "PosAcq_A100",
		Desc:  "Standard positive valence acquisition: A = 100%",
		Cond1: "PosAcq_A100",
	},
	"PosExt_A0": {
		Name:  "PosExt_A0",
		Desc:  "extinguish positive valence: A_NR_Pos -- typically use after some amount of PosAcq_A100",
		Cond1: "PosExt_A0",
	},
	"PosAcqExt_A100Wts": {
		Name:    "PosAcqExt_A100Wts",
		Desc:    "Load weights of acquisition A 100%, go directly to extinguish -- must save weights from PosAcq_A100 first",
		Weights: "PosAcq_A100",
		Cond1:   "PosExt_A0",
	},
	"PosAcqExt_A100_A0": {
		Name:  "PosAcq",
		Desc:  "Standard positive valence acquisition: A = 100%, then extinction A0",
		Cond1: "PosAcq_A100",
		Cond2: "PosExt_A0",
	},
	"PosAcq_A100B50": {
		Name:  "PosAcq_A100B50",
		Desc:  "Standard positive valence acquisition: A = 100%, B = 50%",
		Cond1: "PosAcq_A100B50",
	},
	"PosAcq_A100B0": {
		Name:  "PosAcq_A100B0",
		Desc:  "Standard positive valence acquisition: A = 100%, B = 0%",
		Cond1: "PosAcq_A100B0",
	},
	"PosExt_A0B0": {
		Name:  "PosExt_A0B0",
		Desc:  "extinguish positive valence: A_NR_Pos, B_NR_Pos",
		Cond1: "PosExt_A0B0",
	},
	"PosAcq_A50": {
		Name:  "PosAcq_A50",
		Desc:  "A = 50%, B = 50%",
		Cond1: "PosAcq_A50",
	},
	"PosAcqExt_A100B50_A0B0": {
		Name:  "PosAcqExt_A00B50_A0B0",
		Desc:  "positive valence acquisition A=100%, B=50%, then extinguish A, B = 0%",
		Cond1: "PosAcq_A100B50",
		Cond2: "PosExt_A0B0",
	},
	"PosAcqExt_A100B50_Wts": {
		Name:    "PosAcqExt_A100B50_A0B0_Wts",
		Desc:    "Load weights of acquisition A = 100%, B = 50%, go directly to extinguish -- must save weights from PosAcq_A100B50",
		Weights: "PosAcq_A100B50",
		Cond1:   "PosExt_A0B0",
	},
	"PosAcqExtAcq_A100B50_A0B0_A100B50": {
		Name:  "PosAcqExtAcq_A100B50_A0B0_A100B50",
		Desc:  "Full cycle: acq, ext, acq, A=100%, B=50%, then extinguish, then acq again, marked as ReAcq",
		Cond1: "PosAcq_A100B50",
		Cond2: "PosExt_A0B0",
		Cond3: "PosReAcq_A100B50",
	},
	"PosAcqExt_A100B100": {
		Name:  "PosAcqExt_A100B100",
		Desc:  "",
		Cond1: "PosAcq_A100B100",
		Cond2: "PosExt_A0B0",
	},
	"PosAcq_A100B25": {
		Name:  "PosAcq_A100B25",
		Desc:  "",
		Cond1: "PosAcq_A100B25",
	},
	"NegAcq_D100": {
		Name:  "NegAcq_D100",
		Desc:  "",
		Cond1: "NegAcq_D100",
	},
	"NegAcq_D100E25": {
		Name:  "NegAcq_D100E25",
		Desc:  "",
		Cond1: "NegAcq_D100E25",
	},
	"NegAcqMag": {
		Name:  "NegAcqMag",
		Desc:  "",
		Cond1: "NegAcqMag",
	},
	"PosAcqMag": {
		Name:  "PosAcqMag",
		Desc:  "",
		Cond1: "PosAcqMag",
	},
	"NegAcqExt_D100": {
		Name:  "NegAcqExt_D100",
		Desc:  "",
		Cond1: "NegAcq_D100",
		Cond2: "NegExt_D0",
	},
	"NegExt_D0": {
		Name:  "NegExt_D0",
		Desc:  "",
		Cond1: "NegExt_D0",
	},
	"NegExt_D100Wts": {
		Name:    "NegAcq_D100Wts",
		Desc:    "Load weights of negative acquisition D 100%, go directly to extinguish -- must save weights from NegAcq_D100 first",
		Weights: "NegAcq_D100",
		Cond1:   "NegExt_D0",
	},
	"NegAcqExt_D100E25": {
		Name:  "NegAcqExt_D100E25",
		Desc:  "",
		Cond1: "NegAcq_D100E25",
		Cond2: "NegExt_D0E0",
	},
	"NegExt_D0E0": {
		Name:  "NegExt_D0E0",
		Desc:  "",
		Cond1: "NegExt_D0E0",
	},
	"PosCondInhib": {
		Name:  "PosCondInhib",
		Desc:  "",
		Cond1: "PosAcq_cxA",
		Cond2: "PosCondInhib",
		Cond3: "PosCondInhib_test",
	},
	"PosSecondOrderCond": {
		Name:  "PosSecondOrderCond",
		Desc:  "",
		Cond1: "PosAcqPreSecondOrder",
		Cond2: "PosSecondOrderCond",
	},
	"PosBlocking": {
		Name:  "PosBlocking",
		Desc:  "",
		Cond1: "PosBlocking_A_train",
		Cond2: "PosBlocking",
		Cond3: "PosBlocking_test",
	},
	"PosBlocking2": {
		Name:  "PosBlocking2",
		Desc:  "",
		Cond1: "PosBlocking_A_train",
		Cond2: "PosBlocking",
		Cond3: "PosBlocking2_test",
	},
	"NegCondInhib": {
		Name:  "NegCondInhib",
		Desc:  "",
		Cond1: "NegAcq_D100E25",
		Cond2: "NegCondInh",
		Cond3: "NegCondInh_test",
	},
	"AbaRenewal": {
		Name:  "AbaRenewal",
		Desc:  "",
		Cond1: "PosAcq_cxA",
		Cond2: "PosExtinct_cxB",
		Cond3: "PosRenewal_cxA",
	},
	"NegBlocking": {
		Name:  "NegBlocking",
		Desc:  "",
		Cond1: "NegBlocking_E_train",
		Cond2: "NegBlocking",
		Cond3: "NegBlocking_test",
	},
	"PosSum_test": {
		Name:  "PosSum_test",
		Desc:  "",
		Cond1: "PosSumAcq",
		Cond2: "PosSumCondInhib",
		Cond3: "PosSum_test",
	},
	"NegSum_test": {
		Name:  "NegSum_test",
		Desc:  "",
		Cond1: "NegSumAcq",
		Cond2: "NegSumCondInhib",
		Cond3: "NegSum_test",
	},
	"UnblockingValue": {
		Name:  "UnblockingValue",
		Desc:  "",
		Cond1: "Unblocking_train",
		Cond2: "UnblockingValue",
		Cond3: "UnblockingValue_test",
	},
	"UnblockingIdentity": {
		Name:  "UnblockingIdentity",
		Desc:  "",
		Cond1: "Unblocking_trainUS",
		Cond2: "UnblockingIdentity",
		Cond3: "UnblockingIdentity_test",
	},
	"Overexpect": {
		Name:  "Overexpect",
		Desc:  "",
		Cond1: "Overexpect_train",
		Cond2: "OverexpectCompound",
		Cond3: "Overexpect_test",
	},
	"PosMagChange": {
		Name:  "PosMagChange",
		Desc:  "",
		Cond1: "PosAcqMag",
		Cond2: "PosAcqMagChange",
		Cond3: "Overexpect_test",
	},
	"NegMagChange": {
		Name:  "NegMagChange",
		Desc:  "",
		Cond1: "NegAcqMag",
		Cond2: "NegAcqMagChange",
	},
	"PosNeg": {
		Name:  "PosNeg",
		Desc:  "",
		Cond1: "PosOrNegAcq",
	},
	"PosAcqEarlyUSTest": {
		Name:  "PosAcqEarlyUSTest",
		Desc:  "",
		Cond1: "PosAcq_A100B50",
		Cond2: "PosAcqEarlyUS_test",
	},
	"PosOrNegAcq": {
		Name:  "PosOrNegAcq",
		Desc:  "",
		Cond1: "PosOrNegAcq",
	},
	"PosCondInhib_test": {
		Name:  "PosCondInhib_test",
		Desc:  "For debugging",
		Cond1: "PosCondInhib_test",
	},
	"US0": {
		Name:  "US0",
		Desc:  "",
		Cond1: "US0",
	},
}
