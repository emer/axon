// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// TRCParams provides parameters for how the plus-phase (outcome) state of thalamic relay cell
// (e.g., Pulvinar) neurons is computed from the corresponding driver neuron Burst activation.
// Drivers are hard clamped using Clamp.Rate.
type TRCParams struct {
	DriversOff   bool    `def:"false" desc:"Turn off the driver inputs, in which case this layer behaves like a standard layer"`
	DriveScale   float32 `def:"0.15" min:"0.0" desc:"multiplier on driver input strength, multiplies activation of driver layer to produce Ge excitatory input to TRC unit -- see also Act.Clamp.Burst settings which can produce extra bursting in Ge inputs."`
	FullDriveAct float32 `def:"0.6" min:"0.01" desc:"Level of Max driver layer activation at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (MaxAct/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
	Binarize     bool    `desc:"Apply threshold to driver burst input for computing plus-phase activations -- above BinThr, then Act = BinOn, below = BinOff.  This is beneficial for layers with weaker graded activations, such as V1 or other perceptual inputs."`
	BinThr       float32 `viewif:"Binarize" desc:"Threshold for binarizing in terms of sending Burst activation"`
	BinOn        float32 `def:"0.3" viewif:"Binarize" desc:"Resulting driver Ge value for units above threshold -- lower value around 0.3 or so seems best (DriveScale is NOT applied -- generally same range as that)."`
	BinOff       float32 `def:"0" viewif:"Binarize" desc:"Resulting driver Ge value for units below threshold -- typically 0."`
}

func (tp *TRCParams) Update() {
}

func (tp *TRCParams) Defaults() {
	tp.DriveScale = 0.15
	tp.FullDriveAct = 0.6
	tp.Binarize = false
	tp.BinThr = 0.4
	tp.BinOn = 0.3
	tp.BinOff = 0
}

// DriveGe returns effective excitatory conductance to use for given driver input Burst activation
func (tp *TRCParams) DriveGe(act float32) float32 {
	if !tp.Binarize {
		return tp.DriveScale * act
	}
	if act >= tp.BinThr {
		return tp.BinOn
	} else {
		return tp.BinOff
	}
}

// TRCLayer is the thalamic relay cell layer for DeepAxon.
// It has normal activity during the minus phase, as activated by CT etc inputs,
// and is then driven by strong 5IB driver inputs in the Time.PlusPhase.
// For attentional modulation, TRC maintains pool-level correspondence with CT inputs
// which creates challenges for aligning with driver inputs.
// * Max operation used to integrate across multiple drivers, where necessary,
//   e.g., multiple driver pools map onto single TRC pool (common feedforward theme),
//   *even when there is no logical connection for the i'th unit in each pool* --
//   to make this dimensionality reduction more effective, using lateral connectivity
//   between pools that favors this correspondence is beneficial.  Overall, this is
//   consistent with typical DCNN max pooling organization.
// * Typically, pooled 4D TRC layers should have fewer pools than driver layers,
//   in which case the respective pool geometry is interpolated.  Ideally, integer size
//   differences are best (e.g., driver layer has 2x pools vs TRC).
// * Pooled 4D TRC layer should in general not predict flat 2D drivers, but if so
//   the drivers are replicated for each pool.
// * Similarly, there shouldn't generally be more TRC pools than driver pools, but
//   if so, drivers replicate across pools.
type TRCLayer struct {
	axon.Layer           // access as .Layer
	TRC        TRCParams `view:"inline" desc:"parameters for computing TRC plus-phase (outcome) activations based on Burst activation from corresponding driver neuron"`
	Driver     string    `desc:"name of SuperLayer that sends 5IB Burst driver inputs to this layer"`
}

var KiT_TRCLayer = kit.Types.AddType(&TRCLayer{}, LayerProps)

func (ly *TRCLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0.5
	ly.Act.Decay.Glong = 1
	ly.Act.Decay.KNa = 0
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

func DriveAct(dni int, dly *axon.Layer, sly *SuperLayer, issuper bool) float32 {
	act := float32(0)
	if issuper {
		act = sly.SuperNeurs[dni].Burst
	} else {
		act = dly.Neurons[dni].Act
	}
	lmax := dly.Pools[0].Inhib.Act.Max // normalize by drive layer max act
	if lmax > 0.1 {                    // this puts all layers on equal footing for driving..
		return act / lmax
	}
	return act
}

// GeFmDriverNeuron sets the driver activation for given Neuron,
// based on given Ge driving value (use DriveFmMaxAvg) from driver layer (Burst or Act)
func (ly *TRCLayer) GeFmDriverNeuron(tni int, drvGe, drvInhib float32, cyc int) {
	if tni >= len(ly.Neurons) {
		return
	}
	nrn := &ly.Neurons[tni]
	if nrn.IsOff() {
		return
	}
	geRaw := (1-drvInhib)*nrn.GeRaw + drvGe

	nrn.ClearFlag(axon.NeurHasExt)

	ly.Act.NMDAFmRaw(nrn, 0) // note: could also do drv?
	ly.Act.GvgccFmVm(nrn)
	ly.Learn.LrnNMDAFmRaw(nrn, 0)

	// note: excluding gnmda during driving phase -- probably could exclude always due to ge context?

	// note: each step broken out here so other variants can add extra terms to Raw
	ly.Act.GeFmRaw(nrn, geRaw, nrn.Gnmda)
	nrn.GeRaw = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}

// GeFmDrivers computes excitatory conductance from driver neurons
func (ly *TRCLayer) GeFmDrivers(ltime *axon.Time) {
	cyc := ltime.Cycle // for bursting
	if ly.IsTarget() {
		cyc = ltime.PhaseCycle
	}
	dly, err := ly.DriverLayer(ly.Driver)
	if err != nil {
		return
	}
	sly, issuper := dly.AxonLay.(*SuperLayer)
	drvMax := dly.Pools[0].Inhib.Act.Max
	drvInhib := mat32.Min(1, drvMax/ly.TRC.FullDriveAct)
	for dni := range dly.Neurons {
		drvAct := DriveAct(dni, dly, sly, issuper)
		ly.GeFmDriverNeuron(dni, ly.TRC.DriveGe(drvAct), drvInhib, cyc)
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

// InitExt initializes external input state -- called prior to apply ext
func (ly *TRCLayer) InitExt() {
	msk := bitflag.Mask32(int(axon.NeurHasExt), int(axon.NeurHasTarg), int(axon.NeurHasCmpr))
	drvoff := ly.TRC.DriversOff
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Targ = 0
		nrn.ClearMask(msk)
		if !drvoff {
			nrn.SetFlag(axon.NeurHasTarg)
		}
	}
}
