// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

const (
	// DefBlocks is the default number of training blocks for standard cases
	DefBlocks = 25

	// Default number of trials per condition type
	TrialsPerCond = 4
)

var AllConditions = map[string]*Condition{
	"PosAcq_A100": {
		Desc:       "Standard positive valence acquisition: A = 100%",
		Block:      "PosAcq_A100",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A100_Blk10": {
		Desc:       "Standard positive valence acquisition: A = 100% -- 10 blocks",
		Block:      "PosAcq_A100",
		FixedProb:  true,
		NBlocks:    10,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A100_Blk20": {
		Desc:       "Standard positive valence acquisition: A = 100% -- 20 blocks",
		Block:      "PosAcq_A100",
		FixedProb:  true,
		NBlocks:    20,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A100B50": {
		Desc:       "Standard positive valence acquisition: A = 100%, B = 50%",
		Block:      "PosAcq_A100B50",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A50": {
		Desc:       "Positive valence acquisition: A = 50%",
		Block:      "PosAcq_A50",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A50_Blk10": {
		Desc:       "Positive valence acquisition: A = 50% -- 10 blocks",
		Block:      "PosAcq_A50",
		FixedProb:  true,
		NBlocks:    10,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A50_Blk20": {
		Desc:       "Positive valence acquisition: A = 50% -- 20 blocks",
		Block:      "PosAcq_A50",
		FixedProb:  true,
		NBlocks:    20,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"US0": {
		Desc:       "No US at all",
		Block:      "US0",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A100B100": {
		Desc:       "Positive valence acquisition: A_R_Pos, B at 100%",
		Block:      "PosAcq_A100B100",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcqEarlyUS_test": {
		Desc:       "Testing session: after pos_acq trng, deliver US early or late",
		Block:      "PosAcqEarlyUS_test",
		FixedProb:  true,
		NBlocks:    5,
		NSequences: 2,
		Permute:    false,
	},
	"PosAcq_A100B25": {
		Desc:       "Positive valence acquisition: A_R_Pos 100%, B at 25%",
		Block:      "PosAcq_A100B25",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_A100B0": {
		Desc:       "Positive valence acquisition: A_R_Pos 100%, B at 0%",
		Block:      "PosAcq_A100B0",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosExt_A0": {
		Desc:       "Pavlovian extinction: A_NR_Pos, A = 0%",
		Block:      "PosExt_A0",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosExt_A0_Blk10": {
		Desc:       "Pavlovian extinction: A_NR_Pos, A = 0% -- 10 blocks",
		Block:      "PosExt_A0",
		FixedProb:  false,
		NBlocks:    10,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosExt_A0_Blk20": {
		Desc:       "Pavlovian extinction: A_NR_Pos, A = 0% -- 20 blocks",
		Block:      "PosExt_A0",
		FixedProb:  false,
		NBlocks:    20,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosExt_A0B0": {
		Desc:       "Pavlovian extinction: A_NR_Pos, B_NR_Pos",
		Block:      "PosExt_A0B0",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"NegAcq_D100": {
		Desc:       "Pavlovian conditioning w/ negatively valenced US: D_R_NEG",
		Block:      "NegAcq_D100",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"NegAcq_D100E25": {
		Desc:       "Pavlovian conditioning w/ negatively valenced US: D_R_NEG, E 25%",
		Block:      "NegAcq_D100E25",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"NegExt_D0": {
		Desc:       "Pavlovian conditioning w/ negatively valenced US: A_R_NEG",
		Block:      "NegExt_D0",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"NegExt_D0E0": {
		Desc:       "Pavlovian conditioning w/ negatively valenced US: A_R_NEG",
		Block:      "NegExt_D0E0",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosAcqPreSecondOrder": {
		Desc:       "Positive valence acquisition: A_R_Pos, B at 50%",
		Block:      "PosAcqPreSecondOrder",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosReAcq_A100B50": {
		Desc:       "Positive valence acquisition: A_R_Pos, B at 50% reinf, tags further learning as reacq",
		Block:      "PosReAcq_A100B50",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosReAcq_A100": {
		Desc:       "Positive valence acquisition: A_R_Pos, tags further learning as reacq",
		Block:      "PosReAcq_A100",
		FixedProb:  true,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    true,
	},
	"PosAcq_cxA": {
		Desc:       "Positive valence acquisition: A_R_Pos, A_R_Pos_omit trials, interleaved",
		Block:      "PosAcq_cxA",
		FixedProb:  false,
		NBlocks:    10,
		NSequences: 10,
		Permute:    false,
	},
	"PosExtinct_cxB": {
		Desc:       "Positive valence acquisition: A_R_Pos, A_R_Pos_omit trials, interleaved",
		Block:      "PosExtinct_cxB",
		FixedProb:  false,
		NBlocks:    25,
		NSequences: 10,
		Permute:    false,
	},
	"PosAcqOmit": {
		Desc:       "Positive valence acquisition: A_R_Pos, A_NR_Pos trials, interleaved",
		Block:      "PosAcqOmit",
		FixedProb:  false,
		NBlocks:    10,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosRenewal_cxA": {
		Desc:       "Positive valence acquisition: A_R_Pos, A_R_Pos_omit trials, interleaved",
		Block:      "PosRenewal_cxA",
		FixedProb:  false,
		NBlocks:    1,
		NSequences: 2,
		Permute:    false,
	},
	"PosBlocking_A_train": {
		Desc:       "Blocking experiment",
		Block:      "PosBlocking_A_train",
		FixedProb:  false,
		NBlocks:    20,
		NSequences: 1,
		Permute:    false,
	},
	"PosBlocking": {
		Desc:       "Blocking experiment",
		Block:      "PosBlocking",
		FixedProb:  false,
		NBlocks:    20,
		NSequences: 2,
		Permute:    false,
	},
	"PosBlocking_test": {
		Desc:       "Blocking experiment",
		Block:      "PosBlocking_test",
		FixedProb:  false,
		NBlocks:    25,
		NSequences: 1,
		Permute:    false,
	},
	"PosBlocking2_test": {
		Desc:       "Blocking experiment",
		Block:      "PosBlocking2_test",
		FixedProb:  false,
		NBlocks:    25,
		NSequences: 2,
		Permute:    false,
	},
	"NegBlocking_E_train": {
		Desc:       "Blocking experiment",
		Block:      "NegBlocking_E_train",
		FixedProb:  false,
		NBlocks:    300,
		NSequences: 1,
		Permute:    false,
	},
	"NegBlocking": {
		Desc:       "Blocking experiment",
		Block:      "NegBlocking",
		FixedProb:  false,
		NBlocks:    200,
		NSequences: 2,
		Permute:    false,
	},
	"NegBlocking_test": {
		Desc:       "Blocking experiment",
		Block:      "NegBlocking_test",
		FixedProb:  false,
		NBlocks:    25,
		NSequences: 1,
		Permute:    false,
	},
	"PosAcqMag": {
		Desc:       "Magnitude experiment",
		Block:      "PosAcqMagnitude",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2 * TrialsPerCond,
		Permute:    false,
	},
	"PosSumAcq": {
		Desc:       "Conditioned Inhibition - A+, C+",
		Block:      "PosSumAcq",
		FixedProb:  false,
		NBlocks:    450,
		NSequences: 3,
		Permute:    false,
	},
	"PosSumCondInhib": {
		Desc:       "Conditioned Inhibition - AX-, A+",
		Block:      "PosCondInhib_BY",
		FixedProb:  false,
		NBlocks:    300,
		NSequences: 3,
		Permute:    false,
	},
	"PosSum_test": {
		Desc:       "Conditioned Inhibition Summation Test",
		Block:      "PosSumCondInhib_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 6,
		Permute:    false,
	},
	"NegSumAcq": {
		Desc:       "Conditioned Inhibition - D-, E-",
		Block:      "NegSumAcq",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 3,
		Permute:    false,
	},
	"NegSumCondInhib": {
		Desc:       "Conditioned Inhibition - DU, D-",
		Block:      "NegCondInhib_FV",
		FixedProb:  false,
		NBlocks:    100,
		NSequences: 3,
		Permute:    false,
	},
	"NegSum_test": {
		Desc:       "Conditioned Inhibition Summation Test",
		Block:      "NegSumCondInhib_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 6,
		Permute:    false,
	},
	"Unblocking_train": {
		Desc:       "A+++,B+++,C+",
		Block:      "Unblocking_train",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 2,
		Permute:    false,
	},
	"UnblockingValue": {
		Desc:       "AX+++,CZ+++",
		Block:      "UnblockingValue",
		FixedProb:  false,
		NBlocks:    25,
		NSequences: 1,
		Permute:    false,
	},
	"UnblockingValue_test": {
		Desc:       "A,X,C,Z",
		Block:      "UnblockingValue_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 1,
		Permute:    false,
	},
	"Unblocking_trainUS": {
		Desc:       "A+++ (water) ,B+++ (food)",
		Block:      "Unblocking_trainUS",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: 15,
		Permute:    false,
	},
	"UnblockingIdentity": {
		Desc:       "AX+++(water),BY+++(water)",
		Block:      "UnblockingIdentity",
		FixedProb:  false,
		NBlocks:    DefBlocks / 2,
		NSequences: 20,
		Permute:    false,
	},
	"UnblockingIdentity_test": {
		Desc:       "A,X,B,Y",
		Block:      "UnblockingIdentity_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: TrialsPerCond,
		Permute:    false,
	},
	"PosAcqMagChange": {
		Desc:       "Magnitude experiment",
		Block:      "PosAcqMagnitudeChange",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    false,
	},
	"NegAcqMag": {
		Desc:       "Magnitude experiment",
		Block:      "NegAcqMagnitude",
		FixedProb:  false,
		NBlocks:    51,
		NSequences: 2 * TrialsPerCond,
		Permute:    false,
	},
	"NegAcqMagChange": {
		Desc:       "Magnitude experiment",
		Block:      "NegAcqMagnitudeChange",
		FixedProb:  false,
		NBlocks:    DefBlocks,
		NSequences: TrialsPerCond,
		Permute:    false,
	},
	"Overexpect_train": {
		Desc:       "Overexpectation training (A+, B+, C+, X+, Y-)",
		Block:      "Overexpectation_train",
		FixedProb:  false,
		NBlocks:    150,
		NSequences: 5,
		Permute:    false,
	},
	"OverexpectCompound": {
		Desc:       "Overexpectation compound training (AX+, BY-, CX+, X+, Y-)",
		Block:      "OverexpectationCompound",
		FixedProb:  false,
		NBlocks:    150,
		NSequences: 5,
		Permute:    false,
	},
	"Overexpect_test": {
		Desc:       "Overexpectation test ( A-, B-, C-, X-)",
		Block:      "Overexpectation_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 5,
		Permute:    false,
	},
	"PosNeg": {
		Desc:       "Positive negative test - W equally reinforced with reward + punishment",
		Block:      "PosNeg",
		FixedProb:  false,
		NBlocks:    150,
		NSequences: 6,
		Permute:    false,
	},
	"PosOrNegAcq": {
		Desc:       "Positive negative acquisition - with reward or punishment on interleaved trials according to user-set probabilities",
		Block:      "PosOrNegAcq",
		FixedProb:  false,
		NBlocks:    150,
		NSequences: 6,
		Permute:    true,
	},
	"NegCondInh": {
		Desc:       "condition inhibition w/ negatively valenced US: CZ_NR_NEG, C_R_NEG interleaved; i.e.,  Z = security signal",
		Block:      "NegCondInhib",
		FixedProb:  false,
		NBlocks:    75,
		NSequences: 10,
		Permute:    true,
	},
	"NegCondInh_test": {
		Desc:       "condition inhibition w/ negatively valenced US: CZ_NR_NEG, C_R_NEG interleaved; i.e.,  Z = security signal",
		Block:      "NegCondInhib_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 6,
		Permute:    false,
	},
	"PosCondInhib": {
		Desc:       "conditioned inhibition training: AX_NR_Pos, A_R_Pos interleaved",
		Block:      "PosCondInhib",
		FixedProb:  false,
		NBlocks:    10,
		NSequences: 2 * TrialsPerCond,
		Permute:    true,
	},
	"PosSecondOrderCond": {
		Desc:       "second order conditioning training: AB_NR_Pos, A_R_Pos interleaved; A = 1st order, F = 2nd order CS",
		Block:      "PosSecondOrderCond",
		FixedProb:  false,
		NBlocks:    10,
		NSequences: 20,
		Permute:    true,
	},
	"PosCondInhib_test": {
		Desc:       "Testing session: A_NR_Pos, AX_NR_Pos, and X_NR_Pos cases",
		Block:      "PosCondInhib_test",
		FixedProb:  false,
		NBlocks:    5,
		NSequences: 6,
		Permute:    false,
	},
}