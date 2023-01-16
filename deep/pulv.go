// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// PulvParams provides parameters for how the plus-phase (outcome)
// state of Pulvinar thalamic relay cell neurons is computed from
// the corresponding driver neuron Burst activation.
// Drivers are hard clamped using Clamp.Rate.
type PulvParams struct {
	DriversOff   bool    `def:"false" desc:"Turn off the driver inputs, in which case this layer behaves like a standard layer"`
	DriveScale   float32 `def:"0.05" min:"0.0" desc:"multiplier on driver input strength, multiplies CaSpkP from driver layer to produce Ge excitatory input to Pulv unit."`
	FullDriveAct float32 `def:"0.6" min:"0.01" desc:"Level of Max driver layer CaSpkP at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (Max/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
}

func (tp *PulvParams) Update() {
}

func (tp *PulvParams) Defaults() {
	tp.DriveScale = 0.05
	tp.FullDriveAct = 0.6
}

// DriveGe returns effective excitatory conductance
// to use for given driver input Burst activation
func (tp *PulvParams) DriveGe(act float32) float32 {
	return tp.DriveScale * act
}

// PulvLayer is the Pulvinar thalamic relay cell layer for DeepAxon.
// It has normal activity during the minus phase, as activated by CT etc inputs,
// and is then driven by strong 5IB driver inputs in the plus phase.
type PulvLayer struct {
	axon.Layer            // access as .Layer
	Pulv       PulvParams `view:"inline" desc:"parameters for computing Pulv plus-phase (outcome) activations based on activation from corresponding driver neuron"`
	Driver     string     `desc:"name of SuperLayer that sends 5IB Burst driver inputs to this layer"`
}

var KiT_PulvLayer = kit.Types.AddType(&PulvLayer{}, LayerProps)

func (ly *PulvLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
	ly.Pulv.Defaults()
	ly.Typ = Pulv
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *PulvLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.Pulv.Update()
}

func (ly *PulvLayer) Class() string {
	return "Pulv " + ly.Cls
}

func (ly *PulvLayer) IsTarget() bool {
	return true // We are a Target-like layer: don't do various adaptive steps
}

///////////////////////////////////////////////////////////////////////////////////////
// Drivers

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// reads pool Gi values
func (ly *PulvLayer) GInteg(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	if ly.Pulv.DriversOff || !ctime.PlusPhase {
		ly.Layer.GInteg(ni, nrn, ctime)
		return
	}
	dly, err := ly.DriverLayer(ly.Driver)
	if err != nil {
		return
	}
	sly, issuper := dly.AxonLay.(*SuperLayer)
	drvMax := dly.Vals.ActAvg.CaSpkP.Max
	nonDriverPct := 1.0 - mat32.Min(1, drvMax/ly.Pulv.FullDriveAct) // how much non-driver to keep
	drvAct := DriveAct(ni, dly, sly, issuper)
	drvGe := ly.Pulv.DriveGe(drvAct)
	ly.GFmSpikeRaw(ni, nrn, ctime)
	nrn.GeRaw = nonDriverPct*nrn.GeRaw + drvGe
	nrn.GeSyn = nonDriverPct*nrn.GeSyn + ly.Act.Dt.GeSynFmRawSteady(drvGe)
	ly.GFmRawSyn(ni, nrn, ctime)
	ly.GiInteg(ni, nrn, ctime)
}

// DriverLayer returns the driver layer for given Driver
func (ly *PulvLayer) DriverLayer(drv string) (*axon.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(drv)
	if err != nil {
		err = fmt.Errorf("PulvLayer %s: Driver Layer: %v", ly.Name(), err)
		log.Println(err)
		return nil, err
	}
	return tly.(axon.AxonLayer).AsAxon(), nil
}

// DriveAct returns the driver activation -- Burst for Super, else CaSpkP
func DriveAct(dni int, dly *axon.Layer, sly *SuperLayer, issuper bool) float32 {
	act := float32(0)
	if issuper {
		act = sly.SuperNeurs[dni].Burst
	} else {
		act = dly.Neurons[dni].CaSpkP
	}
	return act
}
