// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: code converted directly from Urakubo et al (2008)
// MODEL/genesis_customizing/NMDAR.c

package main

import (
	"math"

	"github.com/emer/emergent/chem"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// NMDARState holds NMDA receptor states, with allosteric dynamics
// from Urakubo et al, (2008)
// The [3] arrays correspond to Nt0, Nt1, Nt2: plain NMDA, 2CaM phos, 3CaM phos
type NMDARState struct {
	Mg     float64    `desc:"level of Mg block as a function of membrane potential: 1/(1 + (1.5/3.57)exp(-0.062*Vm)"`
	Vca    float64    `desc:"voltage-dependent calcium flux driver: determines Jca as function of V, includes Mg factor"`
	Jca    float64    `desc:"overall calcium current = Vca * Pca * Nopen"`
	G      float64    `desc:"ionic conductance through the NMDA channel for driving Vm changes = Mg * GMax * Nopen"`
	N0     [3]float64 `desc:"Number in state 1"`
	N1     [3]float64 `desc:"Number in state 2"`
	N2     [3]float64 `desc:"Number in state 3"`
	N3     [3]float64 `desc:"Number in state 4"`
	No     [3]float64 `desc:"Number in Open state -- actually open to allow Ca to flow"`
	Nt0    float64    `inactive:"+" desc:"Total N of NMDAR plain = sum of 0 index in N* states"`
	Nt1    float64    `inactive:"+" desc:"Total N of NMDAR_2Ca2+CaM = sum of 1 index in N* states"`
	Nt2    float64    `inactive:"+" desc:"Total N of NMDAR_3Ca2+CaM = sum of 2 index in N* states"`
	Nopen  float64    `inactive:"+" desc:"Total N in open state = sum(No[0..2])"`
	Ntotal float64    `inactive:"+" desc:"overall total -- should be conserved"`
	GluN2B float64    `desc:"number of available non-bound GluN2B binding sites -- CaMKII and DAPK1 compete to bind here -- Thr286 phosphorylated CaMKII binds, as does *de* phosphorylated DAPK1 at Ser308 -- the number of each bound is tracked in the CaMKII and DAPK1 PSD states"`
}

func (cs *NMDARState) Init() {
	cs.Zero()
	cs.N1[0] = 1
	cs.Total()
	cs.GluN2B = cs.Ntotal // presumably

	// todo InitBaseline for basline binding
}

func (cs *NMDARState) Zero() {
	cs.Mg = 0
	cs.Vca = 0
	cs.Jca = 0
	for k := 0; k < 3; k++ {
		cs.N0[k] = 0
		cs.N1[k] = 0
		cs.N2[k] = 0
		cs.N3[k] = 0
		cs.No[k] = 0
	}
	cs.GluN2B = 0
}

func (cs *NMDARState) Integrate(d *NMDARState) {
	for k := 0; k < 3; k++ {
		chem.Integrate(&cs.N0[k], d.N0[k])
		chem.Integrate(&cs.N1[k], d.N1[k])
		chem.Integrate(&cs.N2[k], d.N2[k])
		chem.Integrate(&cs.N3[k], d.N3[k])
		chem.Integrate(&cs.No[k], d.No[k])
	}
	cs.Total()
	chem.Integrate(&cs.GluN2B, d.GluN2B)
}

func (cs *NMDARState) Total() {
	cs.Nt0 = cs.N0[0] + cs.N1[0] + cs.N2[0] + cs.N3[0] + cs.No[0]
	cs.Nt1 = cs.N0[1] + cs.N1[1] + cs.N2[1] + cs.N3[1] + cs.No[1]
	cs.Nt2 = cs.N0[2] + cs.N1[2] + cs.N2[2] + cs.N3[2] + cs.No[2]
	cs.Nopen = (cs.No[0] + cs.No[1] + cs.No[2])
	cs.Ntotal = cs.Nt0 + cs.Nt1 + cs.Nt2
}

func (cs *NMDARState) Log(dt *etable.Table, row int) {
	pre := "NMDA_"
	dt.SetCellFloat(pre+"Mg", row, cs.Mg)
	dt.SetCellFloat(pre+"Nopen", row, cs.Nopen)
	dt.SetCellFloat(pre+"Jca", row, cs.Jca)
	dt.SetCellFloat(pre+"G", row, cs.G)
	dt.SetCellFloat(pre+"Nt0", row, cs.Nt0)
	dt.SetCellFloat(pre+"Nt1", row, cs.Nt1)
	dt.SetCellFloat(pre+"Nt2", row, cs.Nt2)
	// dt.SetCellFloat(pre+"N0[0]", row, cs.N0[0])
	// dt.SetCellFloat(pre+"N0[1]", row, cs.N0[1])
	// dt.SetCellFloat(pre+"N0[2]", row, cs.N0[2])
	dt.SetCellFloat("GluN2B", row, cs.GluN2B)
}

func (cs *NMDARState) ConfigLog(sch *etable.Schema) {
	pre := "NMDA_"
	*sch = append(*sch, etable.Column{pre + "Mg", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Nopen", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Jca", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "G", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Nt0", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Nt1", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Nt2", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "N0[0]", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "N0[1]", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "N0[2]", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{"GluN2B", etensor.FLOAT64, nil, nil})
}

// NMDARParams holds parameters for NMDA receptor with allosteric dynamics
// from Urakubo et al, (2008)
// The [3] arrays correspond to Nt0, Nt1, Nt2: plain NMDA, 2CaM, 3CaM
type NMDARParams struct {
	Erev   float64 `def:"0" desc:"reversal potential for NMDARs"`
	Pca    float64 `def:"89635" desc:"Normalization for Ca flux (pmol sec-1 mV-1)"`
	Gmax   float64 `def:"10" desc:"maximum conductance (nS)"`
	Kfcam1 float64 `def:"400" desc:"CaM forward rate constant for CaM + C0"`
	Kbcam1 float64 `def:"34.8" desc:"CaM backward rate constant for CaM + C0"`
	Kfcam2 float64 `def:"400" desc:"CaM forward rate constant for CaM + C1"`
	Kbcam2 float64 `def:"34.8" desc:"CaM backward rate constant for CaM + C0"`
	Kfcam3 float64 `def:"4" desc:"CaM forward rate constant for CaM + C2"`
	Kbcam3 float64 `def:"0.348" desc:"CaM backward rate constant for CaM + C2"`
	Kfcam4 float64 `def:"3.458" desc:"CaM forward rate constant for CaM + C3"`
	Kbcam4 float64 `def:"0.891" desc:"CaM backward rate constant for CaM + C3"`
	Kfcam5 float64 `def:"1.994" desc:"CaM forward rate constant for CaM + O = open state"`
	Kbcam5 float64 `def:"2.355" desc:"CaM forward rate constant for CaM + O = open state"`
	Kf1    float64 `def:"10" desc:"forward rate constants for Glu binding driving C0 -> C1, for all CaM variants"`
	Kb1    float64 `def:"25" desc:"backward rate constants for Glu binding driving C0 <- C1, for all CaM variants"`
	Kf2    float64 `def:"5" desc:"forward rate constants for Glu binding driving C1 -> C2 for all CaM variants"`
	Kb2    float64 `def:"50" desc:"backward rate constants for Glu binding driving C1 <- C2 for all CaM variants"`
	KC2C3  float64 `def:"25.6" desc:"NMDAR_2Ca2+CaM +Ca2 -> NMDAR_3Ca2+CaM (/uM /sec)"`
	KC3C2  float64 `def:"400" desc:"NMDAR_2Ca2+CaM +Ca2 <- NMDAR_3Ca2+CaM (/uM /sec)"`
	Glu    float64 `def:"0.4" desc:"Glu quantity (uM sec)"`
}

func (nr *NMDARParams) Defaults() {
	nr.Erev = 0
	nr.Pca = 1.7927e5 * 0.5 // SVR_PSD
	nr.Gmax = 10

	nr.Kfcam1 = 400
	nr.Kbcam1 = 34.8
	nr.Kfcam2 = 400
	nr.Kbcam2 = 34.8
	nr.Kfcam3 = 4
	nr.Kbcam3 = 0.348
	nr.Kfcam4 = 3.458
	nr.Kbcam4 = 0.891
	nr.Kfcam5 = 1.994
	nr.Kbcam5 = 2.355

	nr.Kf1 = 10
	nr.Kb1 = 25
	nr.Kf2 = 5
	nr.Kb2 = 50

	nr.KC2C3 = 25.6
	nr.KC3C2 = 400

	nr.Glu = 0.4 // was 0.12
}

// Step increments NMDAR state in response to Ca/CaM binding
// ca = Ca2+ Co, c2 = 2Ca2+CaM Co, c3 = 3Ca2+CaM Co
func (nr *NMDARParams) StepCaCaM(c, d *NMDARState, vm, ca, c2, c3 float64, spike bool, dca *float64) {
	dt := chem.IntegrationDt

	if spike {
		T := nr.Glu
		for k := 0; k < 3; k++ {
			d.N0[k] = c.N0[k] * math.Exp(-nr.Kf1*T)
			d.N1[k] = (c.N0[k] * nr.Kf1) / (nr.Kf2 - nr.Kf1)
			d.N1[k] = (math.Exp(-nr.Kf1*T) - math.Exp(-nr.Kf2*T)) * d.N1[k]
			d.N1[k] = c.N1[k]*math.Exp(-nr.Kf2*T) + d.N1[k]
			d.N2[k] = c.N2[k] - (d.N1[k] - c.N1[k]) - (d.N0[k] - c.N0[k])

			c.N0[k] = d.N0[k] // immediate reset
			c.N1[k] = d.N1[k]
			c.N2[k] = d.N2[k]
		}
	}

	j := int(math.Ceil(dt / 0.00003))

	for i := 0; i < j; i++ {
		d.N0[0] = c.N1[0] * nr.Kb1
		d.N1[0] = c.N2[0]*nr.Kb2 - c.N1[0]*nr.Kb1
		d.N2[0] = c.N3[0]*1.8 + c.No[0]*275 - c.N2[0]*(nr.Kb2+8+280)
		d.N3[0] = c.N2[0]*8 - c.N3[0]*1.8
		d.No[0] = c.N2[0]*280 - c.No[0]*275

		for k := 1; k < 3; k++ {
			d.N0[k] = c.N1[k] * nr.Kb1
			d.N1[k] = c.N2[k]*nr.Kb2 - c.N1[k]*nr.Kb1
			d.N2[k] = c.N3[k]*2 + c.No[k]*2000 - c.N2[k]*(nr.Kb2+3+150)
			d.N3[k] = c.N2[k]*3 - c.N3[k]*2
			d.No[k] = c.N2[k]*150 - c.No[k]*2000
		}
		//
		// NMDAR binding to 2Ca2+CaM
		//
		d.N0[0] += -nr.Kfcam1*c.N0[0]*c2 + nr.Kbcam1*c.N0[1]
		d.N1[0] += -nr.Kfcam2*c.N1[0]*c2 + nr.Kbcam2*c.N1[1]
		d.N2[0] += -nr.Kfcam3*c.N2[0]*c2 + nr.Kbcam3*c.N2[1]
		d.N3[0] += -nr.Kfcam4*c.N3[0]*c2 + nr.Kbcam4*c.N3[1]
		d.No[0] += -nr.Kfcam5*c.No[0]*c2 + nr.Kbcam5*c.No[1]

		d.N0[1] += nr.Kfcam1*c.N0[0]*c2 - nr.Kbcam1*c.N0[1]
		d.N1[1] += nr.Kfcam2*c.N1[0]*c2 - nr.Kbcam2*c.N1[1]
		d.N2[1] += nr.Kfcam3*c.N2[0]*c2 - nr.Kbcam3*c.N2[1]
		d.N3[1] += nr.Kfcam4*c.N3[0]*c2 - nr.Kbcam4*c.N3[1]
		d.No[1] += nr.Kfcam5*c.No[0]*c2 - nr.Kbcam5*c.No[1]
		//
		// NMDAR binding to 3Ca2+CaM
		//
		d.N0[0] += -nr.Kfcam1*c.N0[0]*c2 + nr.Kbcam1*c.N0[2]
		d.N1[0] += -nr.Kfcam2*c.N1[0]*c2 + nr.Kbcam2*c.N1[2]
		d.N2[0] += -nr.Kfcam3*c.N2[0]*c2 + nr.Kbcam3*c.N2[2]
		d.N3[0] += -nr.Kfcam4*c.N3[0]*c2 + nr.Kbcam4*c.N3[2]
		d.No[0] += -nr.Kfcam5*c.No[0]*c2 + nr.Kbcam5*c.No[2]

		d.N0[2] += nr.Kfcam1*c.N0[0]*c2 - nr.Kbcam1*c.N0[2]
		d.N1[2] += nr.Kfcam2*c.N1[0]*c2 - nr.Kbcam2*c.N1[2]
		d.N2[2] += nr.Kfcam3*c.N2[0]*c2 - nr.Kbcam3*c.N2[2]
		d.N3[2] += nr.Kfcam4*c.N3[0]*c2 - nr.Kbcam4*c.N3[2]
		d.No[2] += nr.Kfcam5*c.No[0]*c2 - nr.Kbcam5*c.No[2]

		//
		// NMDAR-3Ca2+CaM dissociating to NMDAR-2Ca2+CaM + Ca2+
		//
		d.N0[1] += -nr.KC2C3*c.N0[1]*ca + nr.KC3C2*c.N0[2]
		d.N1[1] += -nr.KC2C3*c.N1[1]*ca + nr.KC3C2*c.N1[2]
		d.N2[1] += -nr.KC2C3*c.N2[1]*ca + nr.KC3C2*c.N2[2]
		d.N3[1] += -nr.KC2C3*c.N3[1]*ca + nr.KC3C2*c.N3[2]
		d.No[1] += -nr.KC2C3*c.No[1]*ca + nr.KC3C2*c.No[2]

		d.N0[2] += nr.KC2C3*c.N0[1]*ca - nr.KC3C2*c.N0[2]
		d.N1[2] += nr.KC2C3*c.N1[1]*ca - nr.KC3C2*c.N1[2]
		d.N2[2] += nr.KC2C3*c.N2[1]*ca - nr.KC3C2*c.N2[2]
		d.N3[2] += nr.KC2C3*c.N3[1]*ca - nr.KC3C2*c.N3[2]
		d.No[2] += nr.KC2C3*c.No[1]*ca - nr.KC3C2*c.No[2]
	}
}

// Step increments NMDAR state
// ca = Ca2+ Co, c2 = 2Ca2+CaM Co, c3 = 3Ca2+CaM Co
func (nr *NMDARParams) Step(c, d *NMDARState, vm, ca, c2, c3 float64, spike bool, dca *float64) {
	nr.StepCaCaM(c, d, vm, ca, c2, c3, spike, dca)

	c.Mg = 1 / (1 + 0.4202*math.Exp(-0.062*vm)) // Mg(1.5)/3.57
	if vm > -0.1 && vm < 0.1 {
		c.Vca = (1.0 / (0.0756 + 0.5*vm)) * c.Mg
	} else {
		c.Vca = -vm / (1 - math.Exp(0.0756*vm)) * c.Mg
	}
	c.Jca = c.Vca * nr.Pca * c.Nopen
	c.G = c.Mg * nr.Gmax * c.Nopen

	*dca += c.Jca * PSDVol
}
