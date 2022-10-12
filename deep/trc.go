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

// TRCParams provides parameters for how the plus-phase (outcome) state of thalamic relay cell
// (e.g., Pulvinar) neurons is computed from the corresponding driver neuron Burst activation.
// Drivers are hard clamped using Clamp.Rate.
type TRCParams struct {
	DriversOff   bool    `def:"false" desc:"Turn off the driver inputs, in which case this layer behaves like a standard layer"`
	DriveScale   float32 `def:"0.05" min:"0.0" desc:"multiplier on driver input strength, multiplies CaSpkP from driver layer to produce Ge excitatory input to TRC unit."`
	FullDriveAct float32 `def:"0.6" min:"0.01" desc:"Level of Max driver layer CaSpkP at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (Max/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
}

func (tp *TRCParams) Update() {
}

func (tp *TRCParams) Defaults() {
	tp.DriveScale = 0.05
	tp.FullDriveAct = 0.6
}

// DriveGe returns effective excitatory conductance to use for given driver input Burst activation
func (tp *TRCParams) DriveGe(act float32) float32 {
	return tp.DriveScale * act
}

// TRCLayer is the thalamic relay cell layer for DeepAxon.
// It has normal activity during the minus phase, as activated by CT etc inputs,
// and is then driven by strong 5IB driver inputs in the plus phase.
type TRCLayer struct {
	axon.Layer           // access as .Layer
	TRC        TRCParams `view:"inline" desc:"parameters for computing TRC plus-phase (outcome) activations based on activation from corresponding driver neuron"`
	Driver     string    `desc:"name of SuperLayer that sends 5IB Burst driver inputs to this layer"`
}

var KiT_TRCLayer = kit.Types.AddType(&TRCLayer{}, LayerProps)

func (ly *TRCLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0.5
	ly.Act.Decay.Glong = 1
	ly.Act.Decay.AHP = 0
	ly.Learn.RLrate.SigmoidMin = 1 // don't use!
	ly.TRC.Defaults()
	ly.Typ = TRC
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *TRCLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.TRC.Update()
}

func (ly *TRCLayer) Class() string {
	return "TRC " + ly.Cls
}

func (ly *TRCLayer) IsTarget() bool {
	return true // We are a Target-like layer: don't do various adaptive steps
}

///////////////////////////////////////////////////////////////////////////////////////
// Drivers

// DriverLayer returns the driver layer for given Driver
func (ly *TRCLayer) DriverLayer(drv string) (*axon.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(drv)
	if err != nil {
		err = fmt.Errorf("TRCLayer %s: Driver Layer: %v", ly.Name(), err)
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

// GeFmDriverNeuron sets the driver activation for given Neuron,
// based on given Ge driving value (use DriveFmMaxAvg) from driver layer (Burst or Act)
func (ly *TRCLayer) GeFmDriverNeuron(nrn *axon.Neuron, drvGe, drvInhib float32) {
	nrn.GeRaw = (1-drvInhib)*nrn.GeRaw + drvGe
	ly.Act.NMDAFmRaw(nrn, 0)
	ly.Learn.LrnNMDAFmRaw(nrn, 0)
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmRaw(nrn, nrn.GeRaw, nrn.Gnmda+nrn.Gvgcc)
	nrn.GeRaw = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}

// GeFmDrivers computes excitatory conductance from driver neurons
func (ly *TRCLayer) GeFmDrivers(ltime *axon.Time) {
	dly, err := ly.DriverLayer(ly.Driver)
	if err != nil {
		return
	}
	sly, issuper := dly.AxonLay.(*SuperLayer)
	drvMax := dly.ActAvg.CaSpkP.Max
	drvInhib := mat32.Min(1, drvMax/ly.TRC.FullDriveAct)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			return
		}
		drvAct := DriveAct(ni, dly, sly, issuper)
		ly.GeFmDriverNeuron(nrn, ly.TRC.DriveGe(drvAct), drvInhib)
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *TRCLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	if ly.TRC.DriversOff || !ltime.PlusPhase {
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			ly.GFmIncNeur(ltime, nrn, 0) // regular
		}
		return
	}
	ly.GeFmDrivers(ltime)
}
