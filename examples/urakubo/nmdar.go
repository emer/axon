// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: code converted directly from Urakubo et al (2008)
// MODEL/genesis_customizing/NMDAR.c

package main

import (
	"math"

	"github.com/emer/emergent/chem"
)

// NMDARState holds NMDA receptor states, with allosteric dynamics
// from Urakubo et al, (2008)
// The [3] arrays correspond to Nt0, Nt1, Nt2: plain NMDA, 2CaM, 3CaM
type NMDARState struct {
	Mg    float64     `desc:"level of Mg block as a function of membrane potential: 1/(1 + (1.5/3.57)exp(-0.062*Vm)"`
	Ei    float64     `desc:""`
	Jca   float64     `desc:""`
	Ji    float64     `desc:""`
	Vca   float64     `desc:""`
	Vi    float64     `desc:""`
	N0    [3]float64  `desc:"Number in state 1"`
	N1    [3]float64  `desc:"Number in state 2"`
	N2    [3]float64  `desc:"Number in state 3"`
	N3    [3]float64  `desc:"Number in state 4"`
	No    [3]float64  `desc:"Number in Open state"`
	Nt0   float64     `desc:"Total N of NMDAR -- 0 index in N* states"`
	Nt1   float64     `desc:"Total N of NMDAR_2Ca2+CaM"`
	Nt2   float64     `desc:"Total N of NMDAR_3Ca2+CaM"`
	Nopen float64     `desc:"Total N in open state"`
	stack int         `desc:""`
	GGlu  [10]float64 `desc:""`
	ttime [10]float64 `desc:""`
}

// NMDARParams holds parameters for NMDA receptor with allosteric dynamics
// from Urakubo et al, (2008)
// The [3] arrays correspond to Nt0, Nt1, Nt2: plain NMDA, 2CaM, 3CaM
type NMDARParams struct {
	Erev   float64     `def:"0" desc:"reversal potential for NMDARs"`
	Pca    float64     `def:"1" desc:"Normalization for Ca flux (pmol sec-1 mV-1)"`
	Gmax   float64     `def:"1" desc:"maximum conductance (nS)"`
	Kfcam1 float64     `def:"400" desc:"CaM forward rate constant for CaM + C0"`
	Kbcam1 float64     `def:"34.8" desc:"CaM backward rate constant for CaM + C0"`
	Kfcam2 float64     `def:"400" desc:"CaM forward rate constant for CaM + C1"`
	Kbcam2 float64     `def:"34.8" desc:"CaM backward rate constant for CaM + C0"`
	Kfcam3 float64     `def:"4" desc:"CaM forward rate constant for CaM + C2"`
	Kbcam3 float64     `def:"0.348" desc:"CaM backward rate constant for CaM + C2"`
	Kfcam4 float64     `def:"3.458" desc:"CaM forward rate constant for CaM + C3"`
	Kbcam4 float64     `def:"0.891" desc:"CaM backward rate constant for CaM + C3"`
	Kfcam5 float64     `def:"1.994" desc:"CaM forward rate constant for CaM + O = open state"`
	Kbcam5 float64     `def:"2.355" desc:"CaM forward rate constant for CaM + O = open state"`
	Kf1    float64     `def:"10" desc:"forward rate constants for Glu binding driving C0 -> C1, for all CaM variants"`
	Kb1    float64     `def:"25" desc:"backward rate constants for Glu binding driving C0 <- C1, for all CaM variants"`
	Kf2    float64     `def:"5" desc:"forward rate constants for Glu binding driving C1 -> C2 for all CaM variants"`
	Kb2    float64     `def:"50" desc:"backward rate constants for Glu binding driving C1 <- C2 for all CaM variants"`
	KC2C3  float64     `def:"25.6" desc:"NMDAR_2Ca2+CaM +Ca2 -> NMDAR_3Ca2+CaM (/uM /sec)"`
	KC3C2  float64     `def:"400" desc:"NMDAR_2Ca2+CaM +Ca2 <- NMDAR_3Ca2+CaM (/uM /sec)"`
	Glu    float64     `def:"0.12" desc:"Glu quantity (uM sec)"`
	flag   int         `desc:""`
	stack  int         `desc:""`
	GGlu   [10]float64 `desc:""`
	ttime  [10]float64 `desc:""`
	delay  float64     `desc:""`
}

func (nr *NMDARParams) Defaults() {
	nr.Eref = 0
	nr.Pca = 1
	nr.Gmax = 1

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

	nr.Glu = 0.12
}

// Step increments NMDAR state
// ca = Ca2+ Co, c2 = 2Ca2+CaM Co, c3 = 3Ca2+CaM Co
func (nr *NMDAR) Step(vm, ca, c2, c3 float64) {

	var NN0, NN1, NN2, NN3, NNo [3]float64

	//
	// Spike stored at buffer
	//
	t := 0.0 // todo: real time
	dt := chem.IntegrationDt
	i := nr.stack
	if nr.flag == 1 {
		nr.flag = 0
		for {
			if i <= 0 {
				break
			}
			nr.ttime[i] = nr.ttime[i-1]
			nr.GGlu[i] = nr.GGlu[i-1]
			i--
		}
		nr.ttime[0] = t + nr.delay
		nr.GGlu[0] = nr.Glu
		nr.stack += 1
		/*
		   printf("NMDAR buffered at %lf with delay %g\n",
		    t, nr.delay);
		*/
	}

	//
	// Input
	//
	i = nr.stack
	if (i > 0) && (nr.ttime[i-1] <= t) {
		T := nr.GGlu[i-1]
		for k := 0; k < 3; k++ {
			NN0[k] = nr.N0[k] * math.Exp(-nr.Kf1*T)
			NN1[k] = (nr.N0[k] * nr.Kf1) / (nr.Kf2 - nr.Kf1)
			NN1[k] = (math.Exp(-nr.Kf1*T) - math.Exp(-nr.Kf2*T)) * NN1[k]
			NN1[k] = nr.N1[k]*math.Exp(-nr.Kf2*T) + NN1[k]
			NN2[k] = nr.N2[k] - (NN1[k] - nr.N1[k]) - (NN0[k] - nr.N0[k])

			nr.N0[k] = NN0[k]
			nr.N1[k] = NN1[k]
			nr.N2[k] = NN2[k]
		}
		nr.stack -= 1
		/* printf("NMDAR at %lf\n", t); */
	}

	j := int(math.Ceil(dt / 0.00003))
	ddt := dt / float64(j)

	for i = 0; i < j; i++ {
		NN0[0] = nr.N1[0] * nr.Kb1
		NN1[0] = nr.N2[0]*nr.Kb2 - nr.N1[0]*nr.Kb1
		NN2[0] = nr.N3[0]*1.8 + nr.No[0]*275 - nr.N2[0]*(nr.Kb2+8+280)
		NN3[0] = nr.N2[0]*8 - nr.N3[0]*1.8
		NNo[0] = nr.N2[0]*280 - nr.No[0]*275

		for k := 1; k < 3; k++ {
			NN0[k] = nr.N1[k] * nr.Kb1
			NN1[k] = nr.N2[k]*nr.Kb2 - nr.N1[k]*nr.Kb1
			NN2[k] = nr.N3[k]*2 + nr.No[k]*2000 - nr.N2[k]*(nr.Kb2+3+150)
			NN3[k] = nr.N2[k]*3 - nr.N3[k]*2
			NNo[k] = nr.N2[k]*150 - nr.No[k]*2000
		}
		//
		// NMDAR binding to 2Ca2+CaM
		//
		NN0[0] += -nr.Kfcam1*nr.N0[0]*c2 + nr.Kbcam1*nr.N0[1]
		NN1[0] += -nr.Kfcam2*nr.N1[0]*c2 + nr.Kbcam2*nr.N1[1]
		NN2[0] += -nr.Kfcam3*nr.N2[0]*c2 + nr.Kbcam3*nr.N2[1]
		NN3[0] += -nr.Kfcam4*nr.N3[0]*c2 + nr.Kbcam4*nr.N3[1]
		NNo[0] += -nr.Kfcam5*nr.No[0]*c2 + nr.Kbcam5*nr.No[1]

		NN0[1] += nr.Kfcam1*nr.N0[0]*c2 - nr.Kbcam1*nr.N0[1]
		NN1[1] += nr.Kfcam2*nr.N1[0]*c2 - nr.Kbcam2*nr.N1[1]
		NN2[1] += nr.Kfcam3*nr.N2[0]*c2 - nr.Kbcam3*nr.N2[1]
		NN3[1] += nr.Kfcam4*nr.N3[0]*c2 - nr.Kbcam4*nr.N3[1]
		NNo[1] += nr.Kfcam5*nr.No[0]*c2 - nr.Kbcam5*nr.No[1]
		//
		// NMDAR binding to 3Ca2+CaM
		//
		NN0[0] += -nr.Kfcam1*nr.N0[0]*c2 + nr.Kbcam1*nr.N0[2]
		NN1[0] += -nr.Kfcam2*nr.N1[0]*c2 + nr.Kbcam2*nr.N1[2]
		NN2[0] += -nr.Kfcam3*nr.N2[0]*c2 + nr.Kbcam3*nr.N2[2]
		NN3[0] += -nr.Kfcam4*nr.N3[0]*c2 + nr.Kbcam4*nr.N3[2]
		NNo[0] += -nr.Kfcam5*nr.No[0]*c2 + nr.Kbcam5*nr.No[2]

		NN0[2] += nr.Kfcam1*nr.N0[0]*c2 - nr.Kbcam1*nr.N0[2]
		NN1[2] += nr.Kfcam2*nr.N1[0]*c2 - nr.Kbcam2*nr.N1[2]
		NN2[2] += nr.Kfcam3*nr.N2[0]*c2 - nr.Kbcam3*nr.N2[2]
		NN3[2] += nr.Kfcam4*nr.N3[0]*c2 - nr.Kbcam4*nr.N3[2]
		NNo[2] += nr.Kfcam5*nr.No[0]*c2 - nr.Kbcam5*nr.No[2]

		//
		// NMDAR-3Ca2+CaM dissociating to NMDAR-2Ca2+CaM + Ca2+
		//
		NN0[1] += -nr.KC2C3*nr.N0[1]*ca + nr.KC3C2*nr.N0[2]
		NN1[1] += -nr.KC2C3*nr.N1[1]*ca + nr.KC3C2*nr.N1[2]
		NN2[1] += -nr.KC2C3*nr.N2[1]*ca + nr.KC3C2*nr.N2[2]
		NN3[1] += -nr.KC2C3*nr.N3[1]*ca + nr.KC3C2*nr.N3[2]
		NNo[1] += -nr.KC2C3*nr.No[1]*ca + nr.KC3C2*nr.No[2]

		NN0[2] += nr.KC2C3*nr.N0[1]*ca - nr.KC3C2*nr.N0[2]
		NN1[2] += nr.KC2C3*nr.N1[1]*ca - nr.KC3C2*nr.N1[2]
		NN2[2] += nr.KC2C3*nr.N2[1]*ca - nr.KC3C2*nr.N2[2]
		NN3[2] += nr.KC2C3*nr.N3[1]*ca - nr.KC3C2*nr.N3[2]
		NNo[2] += nr.KC2C3*nr.No[1]*ca - nr.KC3C2*nr.No[2]

		for k := 0; k < 3; k++ {
			nr.N0[k] += ddt * NN0[k]
			nr.N1[k] += ddt * NN1[k]
			nr.N2[k] += ddt * NN2[k]
			nr.N3[k] += ddt * NN3[k]
			nr.No[k] += ddt * NNo[k]
		}
	}

	nr.Nt0 = nr.N0[0] + nr.N1[0] + nr.N2[0] + nr.N3[0] + nr.No[0]
	nr.Nt1 = nr.N0[1] + nr.N1[1] + nr.N2[1] + nr.N3[1] + nr.No[1]
	nr.Nt2 = nr.N0[2] + nr.N1[2] + nr.N2[2] + nr.N3[2] + nr.No[2]

	nr.Mg = 1 / (1 + 0.4202*math.Exp(-0.062*vm)) // Mg(1.5)/3.57
	if vm > -0.1 && vm < 0.1 {
		nr.Vca = -1/0.0756 + 0.5*vm
	} else {
		nr.Vca = -vm / (1 - math.Exp(0.0756*vm)) * nr.Mg
	}
	nr.Vi = (nr.Erev - vm) * nr.Mg
	nr.Jca = nr.Vca * nr.Pca * (nr.No[0] + nr.No[1] + nr.No[2])
	nr.Ji = nr.Vi * nr.Gmax * (nr.No[0] + nr.No[1] + nr.No[2])
	nr.Nopen = (nr.No[0] + nr.No[1] + nr.No[2])

}
