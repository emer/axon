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
	return NeuronVars
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

//////////////////////////////////////////////////////////////////////////////////////
//  Primary Algorithmic interface.
//
//  The following methods constitute the primary user-called API during AlphaCyc method
//  to compute one complete algorithmic alpha cycle update.
//
//  They just call the corresponding Impl method using the AxonNetwork interface
//  so that other network types can specialize any of these entry points.

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (nt *Network) NewState() {
	nt.EmerNet.(AxonNetwork).NewStateImpl()
}

// Cycle runs one cycle of activation updating.  It just calls the CycleImpl
// method through the AxonNetwork interface, thereby ensuring any specialized
// algorithm-specific version is called as needed (in general, strongly prefer
// updating the Layer specific version).
func (nt *Network) Cycle(ctime *Time) {
	nt.EmerNet.(AxonNetwork).CycleImpl(ctime)
}

// CycleImpl handles entire update for one cycle (msec) of neuron activity
func (nt *Network) CycleImpl(ctime *Time) {
	// todo: each of these methods should be tested for thread benefits -- some may not be worth it
	nt.LearnFun(func(pj AxonPrjn) { pj.GFmSpikes(ctime) }, "GFmSpikes", Thread, Wait)
	nt.LayerFun(func(ly AxonLayer) { ly.GiFmSpikes(ctime) }, "GiFmSpikes", NoThread, Wait)
	nt.NeuronFun(func(ly AxonLayer, ni int, nrn *Neuron) { ly.CycleNeuron(ni, nrn, ctime) }, "CycleNeuron", Thread, Wait)
	nt.SendSpikeFun(func(ly AxonLayer, ni int, nrn *Neuron) { ly.SendSpike(ni, nrn, ctime) }, "SendSpike", Thread, Wait)
	nt.LayerFun(func(ly AxonLayer) { ly.CyclePost(ctime) }, "CyclePost", NoThread, Wait) // def NoThread
	if !ctime.Testing {
		// todo: if use atomic in these functions, can avoid Wait
		// and these are definitely thread!
		nt.SynCaFun(func(pj AxonPrjn) { pj.SendSynCa(ctime) }, "SendSynCa", Thread, Wait)
		nt.SynCaFun(func(pj AxonPrjn) { pj.RecvSynCa(ctime) }, "RecvSynCa", Thread, Wait)
	}
}

// MinusPhase does updating after end of minus phase
func (nt *Network) MinusPhase(ctime *Time) {
	nt.EmerNet.(AxonNetwork).MinusPhaseImpl(ctime)
}

// PlusPhase does updating after end of plus phase
func (nt *Network) PlusPhase(ctime *Time) {
	nt.EmerNet.(AxonNetwork).PlusPhaseImpl(ctime)
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (nt *Network) TargToExt() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).AsAxon().TargToExt()
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (nt *Network) ClearTargExt() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).AsAxon().ClearTargExt()
	}
}

// SpkSt1 saves current acts into SpkSt1 (using SpkCaP)
func (nt *Network) SpkSt1(ctime *Time) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).SpkSt1(ctime)
	}
}

// SpkSt2 saves current acts into SpkSt2 (using SpkCaP)
func (nt *Network) SpkSt2(ctime *Time) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).SpkSt2(ctime)
	}
}

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt(ctime *Time) {
	nt.EmerNet.(AxonNetwork).DWtImpl(ctime)
}

// WtFmDWt updates the weights from delta-weight changes.
// Also calls SynScale every Interval times
func (nt *Network) WtFmDWt(ctime *Time) {
	nt.EmerNet.(AxonNetwork).WtFmDWtImpl(ctime)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts() {
	nt.SlowCtr = 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitWts()
	}
	// separate pass to enforce symmetry
	// st := time.Now()
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitWtSym()
	}
	// dur := time.Now().Sub(st)
	// fmt.Printf("sym: %v\n", dur)
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
		rpjn := ly.RecvPrjns()
		for _, p := range *rpjn {
			if p.IsOff() {
				continue
			}
			pat := p.Pattern()
			switch pt := pat.(type) {
			case *prjn.PoolTile:
				if !pt.HasTopoWts() {
					continue
				}
				pj := p.(AxonPrjn).AsAxon()
				slay := p.SendLay()
				pt.TopoWts(slay.Shape(), ly.Shape(), swts)
				pj.SetSWtsRPool(swts)
			case *prjn.Circle:
				if !pt.TopoWts {
					continue
				}
				pj := p.(AxonPrjn).AsAxon()
				pj.SetSWtsFunc(pt.GaussWts)
			}
		}
	}
}

// InitGScale computes the initial scaling factor for synaptic input conductances G,
// stored in GScale.Scale, based on sending layer initial activation.
func (nt *Network) InitGScale() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitGScale()
	}
}

// DecayState decays activation state by given proportion
// e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
// This is called automatically in NewState, but is avail
// here for ad-hoc decay cases.
func (nt *Network) DecayState(decay, glong float32) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).DecayState(decay, glong)
	}
}

// DecayStateByClass decays activation state for given class name(s)
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
func (nt *Network) DecayStateByClass(decay, glong float32, class ...string) {
	lnms := nt.LayersByClass(class...)
	for _, lynm := range lnms {
		ly := nt.LayerByName(lynm).(AxonLayer)
		if ly.IsOff() {
			continue
		}
		ly.DecayState(decay, glong)
	}
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitActs()
	}
}

// InitExt initializes external input state -- call prior to applying external inputs to layers
func (nt *Network) InitExt() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitExt()
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).UpdateExtFlags()
	}
}

// NewStateImpl handles all initialization at start of new input state
func (nt *Network) NewStateImpl() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).NewState()
	}
}

// MinusPhaseImpl does updating after end of minus phase
func (nt *Network) MinusPhaseImpl(ctime *Time) {
	// not worth threading this probably
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).MinusPhase(ctime)
	}
}

// PlusPhaseImpl does updating after end of plus phase
func (nt *Network) PlusPhaseImpl(ctime *Time) {
	// not worth threading this probably
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).PlusPhase(ctime)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWtImpl computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWtImpl(ctime *Time) {
	nt.LayerFun(func(ly AxonLayer) { ly.DWtLayer(ctime) }, "DWtLayer", NoThread, Wait) // def no thr
	nt.LearnFun(func(pj AxonPrjn) { pj.DWt(ctime) }, "DWt", Thread, Wait)              // def thread
}

// WtFmDWtImpl updates the weights from delta-weight changes.
func (nt *Network) WtFmDWtImpl(ctime *Time) {
	nt.LearnFun(func(pj AxonPrjn) { pj.DWtSubMean(ctime) }, "DWtSubMean", Thread, Wait)
	nt.LayerFun(func(ly AxonLayer) { ly.WtFmDWtLayer(ctime) }, "WtFmDWtLayer", NoThread, Wait) // def no
	nt.LearnFun(func(pj AxonPrjn) { pj.WtFmDWt(ctime) }, "WtFmDWt", Thread, Wait)
	nt.EmerNet.(AxonNetwork).SlowAdapt(ctime)
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// GScale conductance scaling, and adapting inhibition
func (nt *Network) SlowAdapt(ctime *Time) {
	nt.SlowCtr++
	if nt.SlowCtr < nt.SlowInterval {
		return
	}
	nt.SlowCtr = 0
	nt.LayerFun(func(ly AxonLayer) { ly.SlowAdapt(ctime) }, "SlowAdapt", NoThread, Wait)
	nt.LearnFun(func(pj AxonPrjn) { pj.SlowAdapt(ctime) }, "SlowAdapt", Thread, Wait)
}

// SynFail updates synaptic failure
func (nt *Network) SynFail(ctime *Time) {
	nt.LearnFun(func(pj AxonPrjn) { pj.SynFail(ctime) }, "SynFail", Thread, Wait)
}

// LrateMod sets the Lrate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LrateSched).
// Updates the effective learning rate factor accordingly.
func (nt *Network) LrateMod(mod float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.(AxonLayer).AsAxon().LrateMod(mod)
	}
}

// LrateSched sets the schedule-based learning rate multiplier.
// See also LrateMod.
// Updates the effective learning rate factor accordingly.
func (nt *Network) LrateSched(sched float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.(AxonLayer).AsAxon().LrateSched(sched)
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
func (nt *Network) UnLesionNeurons() {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.(AxonLayer).AsAxon().UnLesionNeurons()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
func (nt *Network) CollectDWts(dwts *[]float32) bool {
	idx := 0
	made := false
	if *dwts == nil {
		nwts := 0
		for _, lyi := range nt.Layers {
			ly := lyi.(AxonLayer).AsAxon()
			nwts += 5               // ActAvgVals
			nwts += len(ly.Neurons) // ActAvg
			if ly.IsLearnTrgAvg() {
				nwts += len(ly.Neurons)
			}
			for _, pji := range ly.SndPrjns {
				pj := pji.(AxonPrjn).AsAxon()
				nwts += len(pj.Syns) + 3 // Scale, AvgAvg, MaxAvg
			}
		}
		*dwts = make([]float32, nwts)
		made = true
	}
	for _, lyi := range nt.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		(*dwts)[idx+0] = ly.ActAvg.ActMAvg
		(*dwts)[idx+1] = ly.ActAvg.ActPAvg
		(*dwts)[idx+2] = ly.ActAvg.AvgMaxGeM
		(*dwts)[idx+3] = ly.ActAvg.AvgMaxGiM
		(*dwts)[idx+4] = ly.ActAvg.GiMult
		idx += 5
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			(*dwts)[idx+ni] = nrn.ActAvg
		}
		idx += len(ly.Neurons)
		if ly.IsLearnTrgAvg() {
			for ni := range ly.Neurons {
				nrn := &ly.Neurons[ni]
				(*dwts)[idx+ni] = nrn.DTrgAvg
			}
			idx += len(ly.Neurons)
		}
		for _, pji := range ly.SndPrjns {
			pj := pji.(AxonPrjn).AsAxon()
			(*dwts)[idx] = pj.GScale.Scale
			idx++
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
func (nt *Network) SetDWts(dwts []float32, navg int) {
	idx := 0
	davg := 1 / float32(navg)
	for _, lyi := range nt.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		ly.ActAvg.ActMAvg = davg * dwts[idx+0]
		ly.ActAvg.ActPAvg = davg * dwts[idx+1]
		ly.ActAvg.AvgMaxGeM = davg * dwts[idx+2]
		ly.ActAvg.AvgMaxGiM = davg * dwts[idx+3]
		ly.ActAvg.GiMult = davg * dwts[idx+4]
		idx += 5
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			nrn.ActAvg = davg * dwts[idx+ni]
		}
		idx += len(ly.Neurons)
		if ly.IsLearnTrgAvg() {
			for ni := range ly.Neurons {
				nrn := &ly.Neurons[ni]
				nrn.DTrgAvg = dwts[idx+ni]
			}
			idx += len(ly.Neurons)
		}
		for _, pji := range ly.SndPrjns {
			pj := pji.(AxonPrjn).AsAxon()
			pj.GScale.Scale = davg * dwts[idx]
			idx++
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
	neur := 0
	neurMem := 0
	syn := 0
	synMem := 0
	for _, lyi := range nt.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		nn := len(ly.Neurons)
		nmem := nn * int(unsafe.Sizeof(Neuron{}))
		neur += nn
		neurMem += nmem
		fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Nm, nn, (datasize.ByteSize)(nmem).HumanReadable())
		for _, pji := range ly.SndPrjns {
			pj := pji.(AxonPrjn).AsAxon()
			ns := len(pj.Syns)
			syn += ns
			pmem := 2*ns*int(unsafe.Sizeof(Synapse{})) + len(pj.GBuf)*4
			synMem += pmem
			fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name(), ns, (datasize.ByteSize)(pmem).HumanReadable())
		}
	}
	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynMem: %v\n", nt.Nm, neur, (datasize.ByteSize)(neurMem).HumanReadable(), syn, (datasize.ByteSize)(synMem).HumanReadable())
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
