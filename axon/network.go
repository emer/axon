// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/c2h5oh/datasize"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// axon.Network has parameters for running a basic rate-coded Axon network
type Network struct {
	NetworkBase
	SlowInterval int `def:"100" desc:"how frequently to perform slow adaptive processes such as synaptic scaling, inhibition adaptation -- in SlowAdapt method-- long enough for meaningful changes"`
	SlowCtr      int `inactive:"+" desc:"counter for how long it has been since last SlowAdapt step"`
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

// NewNetwork returns a new axon Network
func NewNetwork(name string) *Network {
	net := &Network{}
	net.InitName(net, name)
	return net
}

func (nt *Network) AsAxon() *Network {
	return nt
}

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.SetNThreads(0) // default
	nt.SlowInterval = 100
	nt.SlowCtr = 0
	for _, ly := range nt.Layers {
		ly.Defaults()
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// UnitVarNames returns a list of variable names available on the units in this network.
// Not all layers need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) UnitVarNames() []string {
	return NeuronVarNames
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// SynVarNames returns the names of all the variables on the synapses in this network.
// Not all projections need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

// ////////////////////////////////////////////////////////////////////////////////////
//
//	Primary Algorithmic interface.
//
// The following methods constitute the primary user-called API during Alpha Cycle
// to compute one complete algorithmic alpha cycle update.

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (nt *Network) NewState(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunNewState()
		return
	}
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.NewState(ctx)
	}
}

// Cycle runs one cycle of activation updating using threading methods.
func (nt *Network) Cycle(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunCycle()
		return
	}
	nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.GatherSpikes(ctx, ni) }, "GatherSpikes")
	nt.LayerMapPar(func(ly *Layer) { ly.GiFmSpikes(ctx) }, "GiFmSpikes")         // note: important to be Par for linux / amd64
	nt.LayerMapSeq(func(ly *Layer) { ly.PoolGiFmSpikes(ctx) }, "PoolGiFmSpikes") // note: Par not useful
	nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.CycleNeuron(ctx, ni) }, "CycleNeuron")
	nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.PostSpike(ctx, ni) }, "PostSpike")
	nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.SendSpike(ctx, ni) }, "SendSpike")
	if ctx.Testing.IsFalse() {
		nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.SynCa(ctx, ni) }, "SynCa")
	}
	var ldt, vta *Layer
	for _, ly := range nt.Layers {
		if ly.LayerType() == VTALayer {
			vta = ly
		} else if ly.LayerType() == LDTLayer {
			ldt = ly
		} else {
			ly.CyclePost(ctx)
		}
	}
	// ordering of these is important
	if ldt != nil {
		ldt.CyclePost(ctx)
	}
	if vta != nil {
		vta.CyclePost(ctx)
	}
}

// MinusPhase does updating after end of minus phase
func (nt *Network) MinusPhase(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunMinusPhase()
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.MinusPhase(ctx)
		}
	}
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.MinusPhasePost(ctx)
	}
}

// PlusPhaseStart does updating at the start of the plus phase:
// applies Target inputs as External inputs.
func (nt *Network) PlusPhaseStart(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunPlusPhaseStart()
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.PlusPhaseStart(ctx)
		}
	}
}

// PlusPhase does updating after end of plus phase
func (nt *Network) PlusPhase(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunPlusPhase() // copies all state back down: Neurons, LayerVals, Pools
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.PlusPhase(ctx)
		}
	}
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.PlusPhasePost(ctx)
	}
	nt.GPU.SyncStateToGPU() // plus phase post can do anything
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (nt *Network) TargToExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.TargToExt(ctx)
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (nt *Network) ClearTargExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.ClearTargExt(ctx)
	}
}

// SpkSt1 saves current acts into SpkSt1 (using CaSpkP)
func (nt *Network) SpkSt1(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.SpkSt1(ctx)
	}
}

// SpkSt2 saves current acts into SpkSt2 (using CaSpkP)
func (nt *Network) SpkSt2(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.SpkSt2(ctx)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunDWt()
		return
	}
	nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.DWt(ctx, ni) }, "DWt")
	// nt.PrjnMapSeq(func(pj *Prjn) { pj.DWt(ctx) }, "DWt") // todo: neuron level threaded
}

// WtFmDWt updates the weights from delta-weight changes.
// Also calls SlowAdapt every Interval times
func (nt *Network) WtFmDWt(ctx *Context) {
	nt.LayerMapSeq(func(ly *Layer) { ly.WtFmDWtLayer(ctx) }, "WtFmDWtLayer") // lightweight
	if nt.GPU.On {
		nt.GPU.RunWtFmDWt()
	} else {
		nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.DWtSubMean(ctx, ni) }, "DWtSubMean")
		nt.NeuronMapPar(func(ly *Layer, ni uint32) { ly.WtFmDWt(ctx, ni) }, "WtFmDWt")
		// nt.PrjnMapSeq(func(pj *Prjn) { pj.DWtSubMean(ctx) }, "DWtSubMean") // todo: neuron level threaded
		// nt.PrjnMapSeq(func(pj *Prjn) { pj.WtFmDWt(ctx) }, "WtFmDWt")
	}
	nt.SlowAdapt(ctx)
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (nt *Network) SlowAdapt(ctx *Context) {
	nt.SlowCtr++
	if nt.SlowCtr < nt.SlowInterval {
		return
	}
	nt.SlowCtr = 0

	// note: for now doing all this slow stuff CPU-side
	// These Sync calls always check if GPU is On
	nt.GPU.SyncAllFmGPU()

	nt.LayerMapSeq(func(ly *Layer) { ly.SlowAdapt(ctx) }, "SlowAdapt")
	nt.PrjnMapSeq(func(pj *Prjn) { pj.SlowAdapt(ctx) }, "SlowAdapt")

	nt.GPU.SyncAllToGPU()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts(ctx *Context) {
	nt.BuildPrjnGBuf()
	nt.SlowCtr = 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitWts(ctx, nt) // calls InitActs too
	}
	// separate pass to enforce symmetry
	// st := time.Now()
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitWtSym(ctx)
	}
	// dur := time.Now().Sub(st)
	// fmt.Printf("sym: %v\n", dur)
	nt.GPU.SyncAllToGPU()
	nt.GPU.SyncGBufToGPU()
}

// InitTopoSWts initializes SWt structural weight parameters from
// prjn types that support topographic weight patterns, having flags set to support it,
// includes: prjn.PoolTile prjn.Circle.
// call before InitWts if using Topo wts
func (nt *Network) InitTopoSWts() {
	swts := &etensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		for i := 0; i < ly.NRecvPrjns(); i++ {
			pj := ly.RcvPrjns[i]
			if pj.IsOff() {
				continue
			}
			pat := pj.Pattern()
			switch pt := pat.(type) {
			case *prjn.PoolTile:
				if !pt.HasTopoWts() {
					continue
				}
				slay := pj.Send
				pt.TopoWts(slay.Shape(), ly.Shape(), swts)
				pj.SetSWtsRPool(swts)
			case *prjn.Circle:
				if !pt.TopoWts {
					continue
				}
				pj.SetSWtsFunc(pt.GaussWts)
			}
		}
	}
}

// InitGScale computes the initial scaling factor for synaptic input conductances G,
// stored in GScale.Scale, based on sending layer initial activation.
func (nt *Network) InitGScale(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitGScale(ctx)
	}
}

// DecayState decays activation state by given proportion
// e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
// This is called automatically in NewState, but is avail
// here for ad-hoc decay cases.
func (nt *Network) DecayState(ctx *Context, decay, glong, ahp float32) {
	nt.GPU.SyncStateFmGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.DecayState(ctx, decay, glong, ahp)
	}
	nt.GPU.SyncStateToGPU()
}

// DecayStateByType decays activation state for given layer types
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
func (nt *Network) DecayStateByType(ctx *Context, decay, glong, ahp float32, types ...LayerTypes) {
	nt.DecayStateLayers(ctx, decay, glong, ahp, nt.LayersByType(types...)...)
}

// DecayStateByClass decays activation state for given class name(s)
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
func (nt *Network) DecayStateByClass(ctx *Context, decay, glong, ahp float32, classes ...string) {
	nt.DecayStateLayers(ctx, decay, glong, ahp, nt.LayersByClass(classes...)...)
}

// DecayStateLayers decays activation state for given layers
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g).
// If this is not being called at the start, around NewState call,
// then you should also call: nt.GPU.SyncGBufToGPU()
// to zero the GBuf values which otherwise will persist spikes in flight.
func (nt *Network) DecayStateLayers(ctx *Context, decay, glong, ahp float32, layers ...string) {
	nt.GPU.SyncStateFmGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, lynm := range layers {
		ly := nt.AxonLayerByName(lynm)
		if ly.IsOff() {
			continue
		}
		ly.DecayState(ctx, decay, glong, ahp)
	}
	nt.GPU.SyncStateToGPU()
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitActs(ctx)
	}
	nt.GPU.SyncStateToGPU()
	nt.GPU.SyncGBufToGPU() // zeros everyone
}

// InitExt initializes external input state.
// Call prior to applying external inputs to layers.
func (nt *Network) InitExt(ctx *Context) {
	// note: important to do this for GPU
	// to ensure partial inputs work the same way on CPU and GPU.
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitExt(ctx)
	}
}

// ApplyExts applies external inputs to layers, based on values
// that were set in prior layer-specific ApplyExt calls.
// This does nothing on the CPU, but is critical for the GPU,
// and should be added to all sims where GPU will be used.
func (nt *Network) ApplyExts(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunApplyExts()
		return
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.UpdateExtFlags(ctx)
	}
}

// SynFail updates synaptic failure
func (nt *Network) SynFail(ctx *Context) {
	nt.PrjnMapSeq(func(pj *Prjn) { pj.SynFail(ctx) }, "SynFail")
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateMod(mod float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateSched(sched float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// prjn is for the prjns Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (nt *Network) SetSubMean(trgAvg, prjn float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.SetSubMean(trgAvg, prjn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion methods

// LayersSetOff sets the Off flag for all layers to given setting
func (nt *Network) LayersSetOff(off bool) {
	for _, ly := range nt.Layers {
		ly.SetOff(off)
	}
}

// UnLesionNeurons unlesions neurons in all layers in the network.
// Provides a clean starting point for subsequent lesion experiments.
func (nt *Network) UnLesionNeurons(ctx *Context) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.UnLesionNeurons(ctx)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
func (nt *Network) CollectDWts(ctx *Context, dwts *[]float32) bool {
	idx := 0
	made := false
	if *dwts == nil {
		nwts := 0
		for _, ly := range nt.Layers {
			nwts += 5                // ActAvgVals
			nwts += int(ly.NNeurons) // ActAvg
			if ly.Params.IsLearnTrgAvg() {
				nwts += int(ly.NNeurons)
			}
			for _, pj := range ly.SndPrjns {
				nwts += len(pj.Syns) + 3 // Scale, AvgAvg, MaxAvg
			}
		}
		*dwts = make([]float32, nwts)
		made = true
	}
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		(*dwts)[idx+0] = ly.LayerVals(0).ActAvg.ActMAvg
		(*dwts)[idx+1] = ly.LayerVals(0).ActAvg.ActPAvg
		(*dwts)[idx+2] = ly.LayerVals(0).ActAvg.AvgMaxGeM
		(*dwts)[idx+3] = ly.LayerVals(0).ActAvg.AvgMaxGiM
		(*dwts)[idx+4] = ly.LayerVals(0).ActAvg.GiMult
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			(*dwts)[idx+int(ni)] = NrnAvgV(ctx, ni, ActAvg)
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIdx + lni
				(*dwts)[idx+int(ni)] = NrnAvgV(ctx, ni, DTrgAvg)
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPrjns {
			for j := range pj.Syns {
				sy := &(pj.Syns[j])
				(*dwts)[idx+j] = sy.DWt
			}
			idx += len(pj.Syns)
		}
	}
	return made
}

// SetDWts sets the DWt weight changes from given array of floats, which must be correct size
// navg is the number of processors aggregated in these dwts -- some variables need to be
// averaged instead of summed (e.g., ActAvg)
func (nt *Network) SetDWts(ctx *Context, dwts []float32, navg int) {
	idx := 0
	davg := 1 / float32(navg)
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		ly.LayerVals(0).ActAvg.ActMAvg = davg * dwts[idx+0]
		ly.LayerVals(0).ActAvg.ActPAvg = davg * dwts[idx+1]
		ly.LayerVals(0).ActAvg.AvgMaxGeM = davg * dwts[idx+2]
		ly.LayerVals(0).ActAvg.AvgMaxGiM = davg * dwts[idx+3]
		ly.LayerVals(0).ActAvg.GiMult = davg * dwts[idx+4]
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			SetNrnAvgV(ctx, ni, ActAvg, davg*dwts[idx+int(ni)])
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIdx + lni
				SetNrnAvgV(ctx, ni, DTrgAvg, dwts[idx+int(ni)])
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPrjns {
			ns := len(pj.Syns)
			for j := range pj.Syns {
				sy := &(pj.Syns[j])
				sy.DWt = dwts[idx+j]
			}
			idx += ns
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports / Threading Allocation

// SizeReport returns a string reporting the size of each layer and projection
// in the network, and total memory footprint.
func (nt *Network) SizeReport() string {
	var b strings.Builder
	globaNumNeurons := 0
	globalMemNeurons := 0
	globalNumSynapses := 0
	globalMemSynapses := 0

	memNeuron := int(unsafe.Sizeof(Neuron{}))
	memSynapse := int(unsafe.Sizeof(Synapse{}))

	for _, ly := range nt.Layers {
		layerNumNeurons := int(ly.NNeurons)
		// Sizeof returns size of struct in bytes
		layerMemNeurons := layerNumNeurons * memNeuron
		globaNumNeurons += layerNumNeurons
		globalMemNeurons += layerMemNeurons
		fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Nm, layerNumNeurons,
			(datasize.ByteSize)(layerMemNeurons).HumanReadable())
		for _, pj := range ly.SndPrjns {
			projNumSynapses := len(pj.Syns)
			globalNumSynapses += projNumSynapses
			// We only calculate the size of the important parts of the proj struct:
			//  1. Synapse slice (consists of Synapse struct)
			//  2. RecvConIdx + RecvSynIdx + SendConIdx (consists of int32 indices = 4B)
			// Everything else (like eg the GBuf) is not included in the size calculation, as their size
			// doesn't grow quadratically with the number of neurons, and hence pales when compared to the synapses
			// It's also useful to run a -memprofile=mem.prof to validate actual memory usage
			projMemSynapses := projNumSynapses * memSynapse
			projMemIdxs := len(pj.RecvConIdx)*4 + len(pj.RecvSynIdx)*4 + len(pj.SendConIdx)*4
			globalMemSynapses += projMemSynapses + projMemIdxs
			fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name(),
				projNumSynapses, (datasize.ByteSize)(projMemSynapses).HumanReadable())
		}
	}
	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynMem: %v\n",
		nt.Nm, globaNumNeurons, (datasize.ByteSize)(globalMemNeurons).HumanReadable(), globalNumSynapses,
		(datasize.ByteSize)(globalMemSynapses).HumanReadable())
	return b.String()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network props for gui

var NetworkProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"SaveWtsJSON", ki.Props{
			"label": "Save Wts...",
			"icon":  "file-save",
			"desc":  "Save json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"OpenWtsJSON", ki.Props{
			"label": "Open Wts...",
			"icon":  "file-open",
			"desc":  "Open json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"sep-file", ki.BlankProp{}},
		{"Build", ki.Props{
			"icon": "update",
			"desc": "build the network's neurons and synapses according to current params",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the network weight values according to prjn parameters",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the network activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"AddLayer", ki.Props{
			"label": "Add Layer...",
			"icon":  "new",
			"desc":  "add a new layer to network",
			"Args": ki.PropSlice{
				{"Layer Name", ki.Props{}},
				{"Layer Shape", ki.Props{
					"desc": "shape of layer, typically 2D (Y, X) or 4D (Pools Y, Pools X, Units Y, Units X)",
				}},
				{"Layer Type", ki.Props{
					"desc": "type of layer -- used for determining how inputs are applied",
				}},
			},
		}},
		{"ConnectLayerNames", ki.Props{
			"label": "Connect Layers...",
			"icon":  "new",
			"desc":  "add a new connection between layers in the network",
			"Args": ki.PropSlice{
				{"Send Layer Name", ki.Props{}},
				{"Recv Layer Name", ki.Props{}},
				{"Pattern", ki.Props{
					"desc": "pattern to connect with",
				}},
				{"Prjn Type", ki.Props{
					"desc": "type of projection -- direction, or other more specialized factors",
				}},
			},
		}},
		{"AllPrjnScales", ki.Props{
			"icon":        "file-sheet",
			"desc":        "AllPrjnScales returns a listing of all PrjnScale parameters in the Network in all Layers, Recv projections.  These are among the most important and numerous of parameters (in larger networks) -- this helps keep track of what they all are set to.",
			"show-return": true,
		}},
	},
}
