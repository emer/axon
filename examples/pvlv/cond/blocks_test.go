// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

import (
	"fmt"
	"strings"
	"testing"

	"github.com/goki/ki/ints"
)

func TestCSContext(t *testing.T) {
	css := make(map[string]int)
	ctxs := make(map[string]int)
	maxTicks := 0
	for blnm, bl := range AllBlocks {
		for _, trl := range bl {
			cnt := css[trl.CS]
			css[trl.CS] = cnt + 1
			cnt = ctxs[trl.Context]
			ctxs[trl.Context] = cnt + 1
			maxTicks = ints.MaxInt(maxTicks, trl.NTicks)

			if trl.CS == "" {
				t.Errorf("CS is empty: %s   in block: %s  trial: %s\n", trl.CS, blnm, trl.Name)
			}
			if trl.Context == "" {
				fmt.Printf("Context: %s empty -- will be copied to CS: %s   in block: %s  trial: %s\n", trl.Context, trl.CS, blnm, trl.Name)
			}
			if trl.Context != trl.CS {
				fmt.Printf("Context: %s != CS: %s   in block: %s  trial: %s\n", trl.Context, trl.CS, blnm, trl.Name)
			}
			if len(trl.CS) > 1 && trl.CS2Start <= 0 {
				t.Errorf("CS has multiple elements but CS2Start is not set: %s   in block: %s  trial: %s\n", trl.CS, blnm, trl.Name)
			}
			if trl.CS2Start > 0 {
				if len(trl.CS) != 2 {
					t.Errorf("CS2Start is set but CS != 2 elements: %s   in block: %s  trial: %s\n", trl.CS, blnm, trl.Name)
				}
				// fmt.Printf("CS2Start: %d  CS: %s   in block: %s  trial: %s\n", trl.CS2Start, trl.CS, blnm, trl.Name)
			}
			if strings.Contains(trl.Name, "_R") && trl.USProb == 0 {
				fmt.Printf("_R trial with USProb = 0 in block: %s  trial: %s\n", blnm, trl.Name)
			}
			if strings.Contains(trl.Name, "_NR") && trl.USProb != 0 {
				fmt.Printf("_NR trial with USProb != 0 in block: %s  trial: %s\n", blnm, trl.Name)
			}
			if strings.Contains(trl.Name, "_test") && !trl.Test {
				fmt.Printf("_test Trial.Name with Test = false in block: %s  trial: %s\n", blnm, trl.Name)
			}
			if strings.Contains(blnm, "_test") && !trl.Test {
				fmt.Printf("_test Block name with Test = false in block: %s  trial: %s\n", blnm, trl.Name)
			}
			cs := trl.CS[0:1]
			if _, ok := Stims[cs]; !ok {
				t.Errorf("CS not found in list of Stims: %s\n", cs)
			}
			if len(trl.CS) > 1 {
				cs = trl.CS[1:2]
				if _, ok := Stims[cs]; !ok {
					t.Errorf("CS2 not found in list of Stims: %s\n", cs)
				}
			}
			if _, ok := Contexts[trl.Context]; !ok {
				t.Errorf("Context not found in list of Contexts: %s\n", trl.Context)
			}
		}
	}
	fmt.Printf("\nList of unique CSs and use count:\n")
	for cs, cnt := range css {
		fmt.Printf("%s \t %d\n", cs, cnt)
	}
	fmt.Printf("\nList of unique Contexts and use count:\n")
	for ctx, cnt := range ctxs {
		fmt.Printf("%s \t %d\n", ctx, cnt)
	}
	fmt.Printf("MaxTicks: %d\n", maxTicks)
}

func TestConds(t *testing.T) {
	for cnm, cd := range AllConditions {
		_, ok := AllBlocks[cd.Block]
		if !ok {
			t.Errorf("Block name: %s not found in Condition: %s\n", cd.Block, cnm)
		}
	}
}

func TestRuns(t *testing.T) {
	for rnm, run := range AllRuns {
		nc := run.NConds()
		if nc == 0 {
			t.Errorf("Run name: %s has no Conds\n", rnm)
		}
		for i := 0; i < nc; i++ {
			cnm, _ := run.Cond(i)
			_, ok := AllConditions[cnm]
			if !ok {
				t.Errorf("Run: %s Condition name: %s number: %d not found\n", rnm, cnm, i)
			}
		}
	}
}
