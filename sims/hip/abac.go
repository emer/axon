// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/tensorcore"
)

// ConfigInputs generates the AB-AC input patterns
func (ss *Sim) ConfigInputs() {
	abac := ss.Root.Dir("ABAC")
	hp := &ss.Config.Hip
	ecY := hp.EC3NPool.Y
	ecX := hp.EC3NPool.X
	plY := hp.EC3NNrn.Y // good idea to get shorter vars when used frequently
	plX := hp.EC3NNrn.X // makes much more readable
	trials := ss.Config.Run.Trials
	pctAct := ss.Config.Env.ECPctAct
	nOn := patterns.NFromPct(pctAct, plY*plX)
	nDiff := patterns.NFromPct(ss.Config.Env.MinDiffPct, nOn)
	ctxtFlip := patterns.NFromPct(ss.Config.Env.CtxtFlipPct, nOn)

	voc := abac.Dir("Vocab")
	empty := voc.Float32("empty", trials, plY, plX)
	a := voc.Float32("A", trials, plY, plX)
	b := voc.Float32("B", trials, plY, plX)
	c := voc.Float32("C", trials, plY, plX)
	la := voc.Float32("lA", trials, plY, plX)
	lb := voc.Float32("lB", trials, plY, plX)
	ctxt := voc.Float32("ctxt", 3, plY, plX)

	// patterns.MinDiffPrintIterations = true

	patterns.PermutedBinaryMinDiff(a, nOn, 1, 0, nDiff)
	patterns.PermutedBinaryMinDiff(b, nOn, 1, 0, nDiff)
	patterns.PermutedBinaryMinDiff(c, nOn, 1, 0, nDiff)
	patterns.PermutedBinaryMinDiff(la, nOn, 1, 0, nDiff)
	patterns.PermutedBinaryMinDiff(lb, nOn, 1, 0, nDiff)
	patterns.PermutedBinaryMinDiff(ctxt, nOn, 1, 0, nDiff)

	// 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
	for i := range (ecY - 1) * ecX * 3 {
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i)
		tsr := voc.Float32(ctxtNm, 0, plY, plX)
		patterns.ReplicateRows(tsr, ctxt.SubSpace(list), trials)
		patterns.FlipBitsRows(tsr, ctxtFlip, ctxtFlip, 1, 0)
	}

	abName := voc.StringValue("ABName", trials)
	acName := voc.StringValue("ACName", trials)
	lureName := voc.StringValue("LureName", trials)
	patterns.NameRows(abName, "AB_", 2)
	patterns.NameRows(acName, "AC_", 2)
	patterns.NameRows(lureName, "Lure_", 2)

	abFull := voc.Float32("ABFull", trials, ecY, ecX, plY, plX)
	patterns.Mix(abFull, trials, a, b, voc.Float32("ctxt0"), voc.Float32("ctxt1"), voc.Float32("ctxt2"), voc.Float32("ctxt3"))

	abTest := voc.Float32("ABTest", trials, ecY, ecX, plY, plX)
	patterns.Mix(abTest, trials, a, empty, voc.Float32("ctxt0"), voc.Float32("ctxt1"), voc.Float32("ctxt2"), voc.Float32("ctxt3"))

	acFull := voc.Float32("ACFull", trials, ecY, ecX, plY, plX)
	patterns.Mix(acFull, trials, a, b, voc.Float32("ctxt4"), voc.Float32("ctxt5"), voc.Float32("ctxt6"), voc.Float32("ctxt7"))

	acTest := voc.Float32("ACTest", trials, ecY, ecX, plY, plX)
	patterns.Mix(acTest, trials, a, empty, voc.Float32("ctxt4"), voc.Float32("ctxt5"), voc.Float32("ctxt6"), voc.Float32("ctxt7"))

	lureFull := voc.Float32("LureFull", trials, ecY, ecX, plY, plX)
	patterns.Mix(lureFull, trials, la, lb, voc.Float32("ctxt8"), voc.Float32("ctxt9"), voc.Float32("ctxt10"), voc.Float32("ctxt11"))

	lureTest := voc.Float32("LureTest", trials, ecY, ecX, plY, plX)
	patterns.Mix(lureTest, trials, la, empty, voc.Float32("ctxt8"), voc.Float32("ctxt9"), voc.Float32("ctxt10"), voc.Float32("ctxt11"))

	//////// Inputs

	inp := abac.Dir("Inputs")
	ab := inp.Dir("TrainAB")
	ab.Set("Name", abName.Clone())
	ab.Set("Input", abFull.Clone())
	ab.Set("EC5", abFull.Clone())

	ac := inp.Dir("TrainAC")
	ac.Set("Name", acName.Clone())
	ac.Set("Input", acFull.Clone())
	ac.Set("EC5", acFull.Clone())

	test := inp.Dir("TestAll")
	test.Set("Name", abName.Clone().AppendFrom(acName).AppendFrom(lureName))
	test.Set("Input", abTest.Clone().AppendFrom(acTest).AppendFrom(lureTest))
	test.Set("EC5", abFull.Clone().AppendFrom(acFull).AppendFrom(lureFull))

	sty := func(s *tensorcore.GridStyle) {
		s.Size.Min = 20
	}

	all := abac.ValuesFunc(nil)
	for _, vl := range all {
		tensorcore.AddGridStylerTo(vl, sty)
	}
}
