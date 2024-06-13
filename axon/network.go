// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	"cogentcore.org/core/base/datasize"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/views"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
)

// axon.Network implements the Axon spiking model,
// building on the algorithm-independent NetworkBase that manages
// all the infrastructure.
type Network struct {
	NetworkBase
}

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name,
// and initializes NetIndex in global list of Network
func (nt *Network) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
	nt.MaxData = 1
	nt.NetIndex = uint32(len(Networks))
	Networks = append(Networks, nt)
	TheNetwork = nt
}

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

// NewPath returns new path of proper type
func (nt *Network) NewPath() emer.Path {
	return &Path{}
}

// Defaults sets all the default parameters for all layers and pathways
func (nt *Network) Defaults() {
	nt.Rubicon.Defaults()
	nt.SetNThreads(0) // default
	for _, ly := range nt.Layers {
		ly.Defaults()
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and pathways
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// ////////////////////////////////////////////////////////////////////////////////////
//
//	Primary Algorithmic interface.
//
// The following methods constitute the primary user-called API during Alpha Cycle
// to compute one complete algorithmic alpha cycle update.

// NewState handles all initialization at start of new input pattern.
// This is called *before* applying external input data and operates across
// all data parallel values.  The current Context.NData should be set
// properly prior to calling this and subsequent Cycle methods.
func (nt *Network) NewState(ctx *Context) {
	nt.Ctx.NetIndexes.NData = ctx.NetIndexes.NData
	// if nt.GPU.On { // todo: this has a bug in neuron-level access in updating SpkPrv
	// 	nt.GPU.RunNewState()
	// 	return
	// }
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.NewState(ctx)
	}
	if nt.GPU.On {
		nt.GPU.SyncStateGBufToGPU()
	}
}

// Cycle runs one cycle of activation updating using threading methods.
func (nt *Network) Cycle(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunCycle()
		return
	}
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.GatherSpikes(ctx, ni) }, "GatherSpikes")
	nt.LayerMapPar(func(ly *Layer) { ly.GiFromSpikes(ctx) }, "GiFromSpikes")         // note: important to be Par for linux / amd64
	nt.LayerMapSeq(func(ly *Layer) { ly.PoolGiFromSpikes(ctx) }, "PoolGiFromSpikes") // note: Par not useful
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.CycleNeuron(ctx, ni) }, "CycleNeuron")
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.PostSpike(ctx, ni) }, "PostSpike")
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.SendSpike(ctx, ni) }, "SendSpike")
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
	if nt.GPU.On {
		nt.GPU.SyncStateToGPU()
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
		nt.GPU.RunPlusPhase() // copies all state back down: Neurons, LayerValues, Pools
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
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWt(ctx, ni) }, "DWt")
}

// WtFromDWt updates the weights from delta-weight changes.
// Also does ctx.SlowInc() and calls SlowAdapt at SlowInterval
func (nt *Network) WtFromDWt(ctx *Context) {
	nt.LayerMapSeq(func(ly *Layer) { ly.WtFromDWtLayer(ctx) }, "WtFromDWtLayer") // lightweight
	if nt.GPU.On {
		nt.GPU.RunWtFromDWt()
	} else {
		nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWtSubMean(ctx, ni) }, "DWtSubMean")
		nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.WtFromDWt(ctx, ni) }, "WtFromDWt")
	}
	if ctx.SlowInc() {
		nt.SlowAdapt(ctx)
	}
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (nt *Network) SlowAdapt(ctx *Context) {
	// note: for now doing all this slow stuff CPU-side
	// These Sync calls always check if GPU is On
	nt.GPU.SyncAllFromGPU()

	nt.LayerMapSeq(func(ly *Layer) { ly.SlowAdapt(ctx) }, "SlowAdapt")
	nt.PathMapSeq(func(pj *Path) { pj.SlowAdapt(ctx) }, "SlowAdapt")

	nt.GPU.SyncAllToGPU()
	nt.GPU.SyncSynCaToGPU() // was cleared
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts(ctx *Context) { //types:add
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		nt.Rubicon.Reset(ctx, di)
	}
	nt.BuildPathGBuf()
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
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
	nt.GPU.SyncSynCaToGPU() // only time we call this
	nt.GPU.SyncGBufToGPU()
}

// InitTopoSWts initializes SWt structural weight parameters from
// path types that support topographic weight patterns, having flags set to support it,
// includes: paths.PoolTile paths.Circle.
// call before InitWts if using Topo wts
func (nt *Network) InitTopoSWts() {
	ctx := &nt.Ctx
	swts := &tensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		for i := 0; i < ly.NRecvPaths(); i++ {
			pj := ly.RcvPaths[i]
			if pj.IsOff() {
				continue
			}
			pat := pj.Pattern()
			switch pt := pat.(type) {
			case *paths.PoolTile:
				if !pt.HasTopoWts() {
					continue
				}
				slay := pj.Send
				pt.TopoWts(slay.Shape(), ly.Shape(), swts)
				pj.SetSWtsRPool(ctx, swts)
			case *paths.Circle:
				if !pt.TopoWts {
					continue
				}
				pj.SetSWtsFunc(ctx, pt.GaussWts)
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
	nt.GPU.SyncStateFromGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
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
	nt.GPU.SyncStateFromGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, lynm := range layers {
		ly := nt.AxonLayerByName(lynm)
		if ly.IsOff() {
			continue
		}
		for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
	}
	nt.GPU.SyncStateToGPU()
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs(ctx *Context) { //types:add
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
	nt.PathMapSeq(func(pj *Path) { pj.SynFail(ctx) }, "SynFail")
}

// LRateMod sets the LRate modulation parameter for Paths, which is
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
// path is for the paths Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (nt *Network) SetSubMean(trgAvg, path float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.SetSubMean(trgAvg, path)
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
		ly.UnLesionNeurons()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
// This calls SyncSynapsesFromGPU() (nop if not GPU) first.
func (nt *Network) CollectDWts(ctx *Context, dwts *[]float32) bool {
	nt.GPU.SyncSynapsesFromGPU()
	idx := 0
	made := false
	if *dwts == nil {
		nwts := 0
		for _, ly := range nt.Layers {
			nwts += 5                // ActAvgValues
			nwts += int(ly.NNeurons) // ActAvg
			if ly.Params.IsLearnTrgAvg() {
				nwts += int(ly.NNeurons)
			}
			for _, pj := range ly.SndPaths {
				nwts += int(pj.NSyns) + 3 // Scale, AvgAvg, MaxAvg
			}
		}
		*dwts = make([]float32, nwts)
		made = true
	}
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		(*dwts)[idx+0] = ly.LayerValues(0).ActAvg.ActMAvg
		(*dwts)[idx+1] = ly.LayerValues(0).ActAvg.ActPAvg
		(*dwts)[idx+2] = ly.LayerValues(0).ActAvg.AvgMaxGeM
		(*dwts)[idx+3] = ly.LayerValues(0).ActAvg.AvgMaxGiM
		(*dwts)[idx+4] = ly.LayerValues(0).ActAvg.GiMult
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			(*dwts)[idx+int(lni)] = NrnAvgV(ctx, ni, ActAvg)
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				(*dwts)[idx+int(lni)] = NrnAvgV(ctx, ni, DTrgAvg)
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					(*dwts)[idx+int(syi)] = SynV(ctx, syni, DWt)
					// if syni < 100 {
					// 	fmt.Printf("%d: %d = %g\n", syni, syi, (*dwts)[idx+int(syi)])
					// }
				}
			}
			idx += int(pj.NSyns)
		}
	}
	return made
}

// SetDWts sets the DWt weight changes from given array of floats, which must be correct size
// navg is the number of processors aggregated in these dwts -- some variables need to be
// averaged instead of summed (e.g., ActAvg)
// This calls SyncSynapsesToGPU() (nop if not GPU) after.
func (nt *Network) SetDWts(ctx *Context, dwts []float32, navg int) {
	idx := 0
	davg := 1 / float32(navg)
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		ly.LayerValues(0).ActAvg.ActMAvg = davg * dwts[idx+0]
		ly.LayerValues(0).ActAvg.ActPAvg = davg * dwts[idx+1]
		ly.LayerValues(0).ActAvg.AvgMaxGeM = davg * dwts[idx+2]
		ly.LayerValues(0).ActAvg.AvgMaxGiM = davg * dwts[idx+3]
		ly.LayerValues(0).ActAvg.GiMult = davg * dwts[idx+4]
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			SetNrnAvgV(ctx, ni, ActAvg, davg*dwts[idx+int(lni)])
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				SetNrnAvgV(ctx, ni, DTrgAvg, dwts[idx+int(lni)])
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					SetSynV(ctx, syni, DWt, dwts[idx+int(syi)])
					// if syni < 100 {
					// 	fmt.Printf("%d: %d = %g = %g\n", syni, syi, dwts[idx+int(syi)], SynV(ctx, syni, DWt))
					// }
				}
			}
			idx += int(pj.NSyns)
		}
	}
	nt.GPU.SyncSynapsesToGPU() // gpu will use dwts to update
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports / Threading Allocation

// SizeReport returns a string reporting the size of each layer and pathway
// in the network, and total memory footprint.
// If detail flag is true, details per layer, pathway is included.
func (nt *Network) SizeReport(detail bool) string {
	var b strings.Builder

	varBytes := 4
	synVarBytes := 4
	maxData := int(nt.MaxData)
	memNeuron := int(NeuronVarsN)*maxData*varBytes + int(NeuronAvgVarsN)*varBytes + int(NeuronIndexesN)*varBytes
	memSynapse := int(SynapseVarsN)*varBytes + int(SynapseCaVarsN)*maxData*varBytes + int(SynapseIndexesN)*varBytes

	globalProjIndexes := 0

	for _, ly := range nt.Layers {
		if detail {
			nn := int(ly.NNeurons)
			// Sizeof returns size of struct in bytes
			nrnMem := nn * memNeuron
			fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Nm, nn,
				(datasize.Size)(nrnMem).String())
		}
		for _, pj := range ly.SndPaths {
			// We only calculate the size of the important parts of the proj struct:
			//  1. Synapse slice (consists of Synapse struct)
			//  2. RecvConIndex + RecvSynIndex + SendConIndex (consists of int32 indices = 4B)
			// Everything else (like eg the GBuf) is not included in the size calculation, as their size
			// doesn't grow quadratically with the number of neurons, and hence pales when compared to the synapses
			// It's also useful to run a -memprofile=mem.prof to validate actual memory usage
			projMemIndexes := len(pj.RecvConIndex)*varBytes + len(pj.RecvSynIndex)*varBytes + len(pj.SendConIndex)*varBytes
			globalProjIndexes += projMemIndexes
			if detail {
				nSyn := int(pj.NSyns)
				synMem := nSyn*memSynapse + projMemIndexes
				fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name(),
					nSyn, (datasize.Size)(synMem).String())
			}
		}
	}

	nrnMem := (len(nt.Neurons) + len(nt.NeuronAvgs) + len(nt.NeuronIxs)) * varBytes
	synIndexMem := len(nt.SynapseIxs) * varBytes
	synWtMem := (len(nt.Synapses)) * synVarBytes
	synCaMem := (len(nt.SynapseCas)) * synVarBytes

	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynIndexes: %v \t SynWts: %v \t SynCa: %v\n",
		nt.Nm, nt.NNeurons, (datasize.Size)(nrnMem).String(), nt.NSyns,
		(datasize.Size)(synIndexMem).String(), (datasize.Size)(synWtMem).String(), (datasize.Size)(synCaMem).String())
	return b.String()
}

func (nt *Network) ConfigToolbar(tb *core.Toolbar) {
	views.NewFuncButton(tb, nt.ShowAllGlobals).SetText("Global Vars").SetIcon(icons.Info)
	fb := views.NewFuncButton(tb, nt.SaveWtsJSON).SetText("Save Weights").SetIcon(icons.Save)
	fb.Args[0].SetTag("ext", ".wts,.wts.gz")
	fb = views.NewFuncButton(tb, nt.OpenWtsJSON).SetText("Open Weights").SetIcon(icons.Open)
	fb.Args[0].SetTag("ext", ".wts,.wts.gz")
	core.NewSeparator(tb)
	views.NewFuncButton(tb, nt.Build).SetIcon(icons.Reset)
	views.NewFuncButton(tb, nt.InitWts).SetIcon(icons.Reset)
	views.NewFuncButton(tb, nt.InitActs).SetIcon(icons.Reset)
}
