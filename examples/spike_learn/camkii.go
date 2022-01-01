// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CaMVars are intracellular Ca-driven signaling variables for the
// CaMKII+CaM binding -- each can have different numbers of Ca bound
type CaMVars struct {
	CaM         float32 `desc:"CaM = Ca calmodulin, [0-3]Ca bound but unbound to CaMKII"`
	CaM_CaMKII  float32 `desc:"CaMKII-CaM bound together"`
	CaM_CaMKIIP float32 `desc:"CaMKIIP-CaM bound together, P = phosphorylated at Thr286"`
}

// CaMKIIVars are intracellular Ca-driven signaling states
// for CaMKII binding and phosphorylation with CaM + Ca
type CaMKIIVars struct {
	Ca        [4]CaMVars `desc:"increasing levels of Ca binding, 0-3"`
	CaMKII    float32    `desc:"unbound CaMKII = CaM kinase II"`
	CaMKIIP   float32    `desc:"unbound CaMKII P = phosphorylated at Thr286 -- shown with * in Figure S13"`
	CaMKIIact float32    `desc:"total active CaMKII: .75 * Ca[1..3]CaM_CaMKII + 1 * Ca[3]CaM_CaMKIIP + .8 * Ca[1..2]CaM_CaMKIIP"`
	CaMKIItot float32    `desc:"total CaMKII across all states"`
	ActPct    float32    `desc:"proportion of active: CaMKIIact / Total CaMKII (constant?)"`
}

// CaMKIIState is overall intracellular Ca-driven signaling states
// for CaMKII in Cyt and PSD
type CaMKIIState struct {
	Cyt CaMKIIVars `desc:"in cytosol"`
	PSD CaMKIIVars `desc:"in PSD"`
}

// CaCaMParams are the parameters governing the Ca+CaM binding
type CaCaMParams struct {
	CaCaM01        React `desc:"1: Ca+CaM -> 1CaCaM"`
	CaCaM12        React `desc:"2: Ca+1CaM -> 2CaCaM"`
	CaCaM23        React `desc:"3: Ca+2CaM -> 3CaCaM"`
	CaMCaMKII      React `desc:"4: CaM+CaMKII -> CaM-CaMKII"`
	CaMCaMKII3     React `desc:"5: 3CaCaM+CaMKII -> 3CaCaM-CaMKII"`
	CaCaM23_CaMKII React `desc:"6: Ca+2CaCaM-CaMKII -> 3CaCaM-CaMKII"`
	CaCaM_CaMKIIP  React `desc:"8: Ca+nCaCaM-CaMKIIP -> n+1CaCaM-CaMKIIP"`
	CaMCaMKIIP     React `desc:"9: CaM+CaMKIIP -> CaM-CaMKIIP"` // note: typo in SI3 for top PP1, PP2A
	PP1Thr286      Enz   `desc:"10: PP1 dephosphorylating CaMKIIP"`
	PP2AThr286     Enz   `desc:"11: PP2A dephosphorylating CaMKIIP"`
}

func (cp *CaCaMParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SId (0.02083333)
	cp.CaCaM01.SetSec(1.0667, 200)     // 51.2 μM-1, PSD 4.2667
	cp.CaCaM12.SetSec(2.7771, 1000)    // 133.3 μM-1, PSD 11.108
	cp.CaCaM23.SetSec(0.53333, 400)    // 25.6 μM-1, PSD 2.1333
	cp.CaMCaMKII.SetSec(8.3333e-06, 1) // 0.0004 μM-1, PSD 3.3333e-5
	cp.CaMCaMKII3.SetSec(0.16667, 1)   // 0.0004 μM-1, PSD 3.3333e-5

	cp.CaCaM23_CaMKII.SetSec(0.53333, 0.02) // 25.6 μM-1, PSD 2.1333
	cp.CaCaM_CaMKIIP.SetSec(0.020833, 1)    // 1 μM-1, PSD 0.0833335
	cp.CaMCaMKIIP.SetSec(0.16667, 0.001)    // 8 μM-1, PSD 0.66667 -- kAT -> kTA

	cp.PP1Thr286.SetSec(0.0031724, 1.34, 0.335)  // 11 μM Km
	cp.PP2AThr286.SetSec(0.0031724, 1.34, 0.335) // 11 μM Km
}

// StepCaMKIIP is the special CaMKII phosphorylation function from Dupont et al, 2003
func (cp *CaCaMParams) StepCaMKIIP(c, t float32, n *float32) {
	fact := t * (-0.22 + 1.826*t + 0.8*t*t)
	if fact < 0 {
		return
	}
	*n += 0.00029 * fact * c
}

// StepCaMKII does the bulk of Ca + CaM + CaMKII binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCa, nCa = current new Ca
func (cp *CaCaMParams) StepCaMKII(kf float32, c, n *CaMKIIVars, cCa, pp1, pp2a float32, nCa *float32) {
	cp.CaCaM01.StepKf(kf, c.Ca[0].CaM, cCa, c.Ca[1].CaM, &n.Ca[0].CaM, nCa, &n.Ca[1].CaM) // 1
	cp.CaCaM12.StepKf(kf, c.Ca[1].CaM, cCa, c.Ca[2].CaM, &n.Ca[1].CaM, nCa, &n.Ca[2].CaM) // 2
	cp.CaCaM23.StepKf(kf, c.Ca[2].CaM, cCa, c.Ca[3].CaM, &n.Ca[2].CaM, nCa, &n.Ca[3].CaM) // 3

	cp.CaCaM01.StepKf(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &n.Ca[0].CaM_CaMKII, nCa, &n.Ca[1].CaM_CaMKII)        // 1
	cp.CaCaM12.StepKf(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &n.Ca[1].CaM_CaMKII, nCa, &n.Ca[2].CaM_CaMKII)        // 2
	cp.CaCaM23_CaMKII.StepKf(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &n.Ca[2].CaM_CaMKII, nCa, &n.Ca[3].CaM_CaMKII) // 6

	for i := 0; i < 3; i++ {
		cp.CaMCaMKII.StepKf(kf, c.Ca[i].CaM, c.CaMKII, c.Ca[i].CaM_CaMKII, &n.Ca[i].CaM, &n.CaMKII, &n.Ca[i].CaM_CaMKII) // 4
	}
	cp.CaMCaMKII3.StepKf(kf, c.Ca[3].CaM, c.CaMKII, c.Ca[3].CaM_CaMKII, &n.Ca[3].CaM, &n.CaMKII, &n.Ca[3].CaM_CaMKII) // 5

	cp.CaMCaMKIIP.StepKf(kf, c.Ca[0].CaM, c.CaMKIIP, c.Ca[0].CaM_CaMKIIP, &n.Ca[0].CaM, &n.CaMKIIP, &n.Ca[0].CaM_CaMKIIP) // 9
	for i := 1; i < 4; i++ {
		cp.CaCaM_CaMKIIP.StepKf(kf, c.Ca[i].CaM_CaMKIIP, cCa, c.Ca[i+1].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKIIP, nCa, &n.Ca[i+1].CaM_CaMKIIP) // 8
	}

	for i := 0; i < 4; i++ {
		cp.StepCaMKIIP(c.Ca[i].CaM_CaMKII, c.ActPct, &n.Ca[i].CaM_CaMKIIP) // 7
		cp.PP1Thr286.Step(c.Ca[i].CaM_CaMKIIP, pp1, n.Ca[i].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKII, &n.Ca[i].CaM_CaMKIIP)
		cp.PP2AThr286.Step(c.Ca[i].CaM_CaMKIIP, pp1, n.Ca[i].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKII, &n.Ca[i].CaM_CaMKIIP)
	}

	// when all done, update act
	cp.Active(n)
}

// Active updates active total and pct
func (cp *CaCaMParams) Active(n *CaMKIIVars) {
	var act float32

	tot := n.CaMKII + n.CaMKIIP
	for i := 0; i < 4; i++ {
		tot += n.Ca[i].CaM_CaMKII + n.Ca[i].CaM_CaMKIIP
		if i >= 1 && i < 3 {
			act += 0.75*n.Ca[i].CaM_CaMKII + 0.8*n.Ca[i].CaM_CaMKII
		} else if i == 3 {
			act += 0.75*n.Ca[i].CaM_CaMKII + n.Ca[i].CaM_CaMKII
		}
	}
	n.CaMKIIact = act
	n.CaMKIItot = tot
	n.ActPct = act / tot
}

// todo: need pp* per Cyt, PSD
// cCa, nCa = current new Ca
func (cp *CaCaMParams) Step(c, n *CaMKIIState, cCa, pp1, pp2a float32, nCa *float32) {
	*n = *c
	cp.StepCaMKII(1, &c.Cyt, &n.Cyt, cCa, pp1, pp2a, nCa)
	cp.StepCaMKII(4, &c.PSD, &n.PSD, cCa, pp1, 0, nCa)
}
