// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"bufio"
	"compress/gzip"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/emer/emergent/econfig"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"
	"github.com/emer/emergent/weights"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/indent"
	"github.com/goki/kigen/dedupe"
	"github.com/goki/mat32"
)

// NetworkBase manages the basic structural components of a network (layers).
// The main Network then can just have the algorithm-specific code.
type NetworkBase struct {

	// [view: -] we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary.
	EmerNet emer.Network `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary."`

	// overall name of network -- helps discriminate if there are multiple
	Nm string `desc:"overall name of network -- helps discriminate if there are multiple"`

	// filename of last weights file loaded or saved
	WtsFile string `desc:"filename of last weights file loaded or saved"`

	// PVLV system for phasic dopamine signaling, including internal drives, US outcomes.  Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine are computed in equations using inputs from specialized network layers (LDTLayer driven by BLA, CeM layers, VSPatchLayer).  Renders USLayer, PVLayer, DrivesLayer representations based on state updated here.
	PVLV PVLV `desc:"PVLV system for phasic dopamine signaling, including internal drives, US outcomes.  Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine are computed in equations using inputs from specialized network layers (LDTLayer driven by BLA, CeM layers, VSPatchLayer).  Renders USLayer, PVLayer, DrivesLayer representations based on state updated here."`

	// [view: -] map of name to layers -- layer names must be unique
	LayMap map[string]*Layer `view:"-" desc:"map of name to layers -- layer names must be unique"`

	// [view: -] map of layer classes -- made during Build
	LayClassMap map[string][]string `view:"-" desc:"map of layer classes -- made during Build"`

	// [view: -] minimum display position in network
	MinPos mat32.Vec3 `view:"-" desc:"minimum display position in network"`

	// [view: -] maximum display position in network
	MaxPos mat32.Vec3 `view:"-" desc:"maximum display position in network"`

	// optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save
	MetaData map[string]string `desc:"optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save"`

	// if true, the neuron and synapse variables will be organized into a gpu-optimized memory order, otherwise cpu-optimized. This must be set before network Build() is called.
	UseGPUOrder bool `inactive:"+" desc:"if true, the neuron and synapse variables will be organized into a gpu-optimized memory order, otherwise cpu-optimized. This must be set before network Build() is called."`

	// [view: -] network index in global Networks list of networks -- needed for GPU shader kernel compatible network variable access functions (e.g., NrnV, SynV etc) in CPU mode
	NetIdx uint32 `view:"-" desc:"network index in global Networks list of networks -- needed for GPU shader kernel compatible network variable access functions (e.g., NrnV, SynV etc) in CPU mode"`

	// [view: -] maximum synaptic delay across any projection in the network -- used for sizing the GBuf accumulation buffer.
	MaxDelay uint32 `inactive:"+" view:"-" desc:"maximum synaptic delay across any projection in the network -- used for sizing the GBuf accumulation buffer."`

	// maximum number of data inputs that can be processed in parallel in one pass of the network. Neuron storage is allocated to hold this amount during Build process, and this value reflects that.
	MaxData uint32 `inactive:"+" desc:"maximum number of data inputs that can be processed in parallel in one pass of the network. Neuron storage is allocated to hold this amount during Build process, and this value reflects that."`

	// total number of neurons
	NNeurons uint32 `inactive:"+" desc:"total number of neurons"`

	// total number of synapses
	NSyns uint32 `inactive:"+" desc:"total number of synapses"`

	// [view: -] storage for global vars
	Globals []float32 `view:"-" desc:"storage for global vars"`

	// array of layers
	Layers []*Layer `desc:"array of layers"`

	// [view: -] [Layers] array of layer parameters, in 1-to-1 correspondence with Layers
	LayParams []LayerParams `view:"-" desc:"[Layers] array of layer parameters, in 1-to-1 correspondence with Layers"`

	// [view: -] [Layers][MaxData] array of layer values, with extra per data
	LayVals []LayerVals `view:"-" desc:"[Layers][MaxData] array of layer values, with extra per data"`

	// [view: -] [Layers][Pools][MaxData] array of inhibitory pools for all layers.
	Pools []Pool `view:"-" desc:"[Layers][Pools][MaxData] array of inhibitory pools for all layers."`

	// [view: -] [Layers][Neurons][MaxData] entire network's allocation of neuron variables, accessed via NrnV function with flexible striding
	Neurons []float32 `view:"-" desc:"[Layers][Neurons][MaxData] entire network's allocation of neuron variables, accessed via NrnV function with flexible striding"`

	// [view: -] [Layers][Neurons][MaxData]] entire network's allocation of neuron average avariables, accessed via NrnAvgV function with flexible striding
	NeuronAvgs []float32 `view:"-" desc:"[Layers][Neurons][MaxData]] entire network's allocation of neuron average avariables, accessed via NrnAvgV function with flexible striding"`

	// [view: -] [Layers][Neurons] entire network's allocation of neuron index variables, accessed via NrnI function with flexible striding
	NeuronIxs []uint32 `view:"-" desc:"[Layers][Neurons] entire network's allocation of neuron index variables, accessed via NrnI function with flexible striding"`

	// [view: -] [Layers][SendPrjns] pointers to all projections in the network, sender-based
	Prjns []*Prjn `view:"-" desc:"[Layers][SendPrjns] pointers to all projections in the network, sender-based"`

	// [view: -] [Layers][SendPrjns] array of projection parameters, in 1-to-1 correspondence with Prjns, sender-based
	PrjnParams []PrjnParams `view:"-" desc:"[Layers][SendPrjns] array of projection parameters, in 1-to-1 correspondence with Prjns, sender-based"`

	// [view: -] [Layers][SendPrjns][SendNeurons][RecvNeurons] entire network's allocation of synapse idx vars, organized sender-based, with flexible striding, accessed via SynI function
	SynapseIxs []uint32 `view:"-" desc:"[Layers][SendPrjns][SendNeurons][RecvNeurons] entire network's allocation of synapse idx vars, organized sender-based, with flexible striding, accessed via SynI function"`

	// [view: -] [Layers][SendPrjns][SendNeurons][RecvNeurons] entire network's allocation of synapses, organized sender-based, with flexible striding, accessed via SynV function
	Synapses []float32 `view:"-" desc:"[Layers][SendPrjns][SendNeurons][RecvNeurons] entire network's allocation of synapses, organized sender-based, with flexible striding, accessed via SynV function"`

	// [view: -] [Layers][SendPrjns][SendNeurons][RecvNeurons][MaxData] entire network's allocation of synapse Ca vars, organized sender-based, with flexible striding, accessed via SynCaV function
	SynapseCas []float32 `view:"-" desc:"[Layers][SendPrjns][SendNeurons][RecvNeurons][MaxData] entire network's allocation of synapse Ca vars, organized sender-based, with flexible striding, accessed via SynCaV function"`

	// [view: -] [Layers][SendPrjns][SendNeurons] starting offset and N cons for each sending neuron, for indexing into the Syns synapses, which are organized sender-based.
	PrjnSendCon []StartN `view:"-" desc:"[Layers][SendPrjns][SendNeurons] starting offset and N cons for each sending neuron, for indexing into the Syns synapses, which are organized sender-based."`

	// [view: -] [Layers][RecvPrjns][RecvNeurons] starting offset and N cons for each recv neuron, for indexing into the RecvSynIdx array of indexes into the Syns synapses, which are organized sender-based.
	PrjnRecvCon []StartN `view:"-" desc:"[Layers][RecvPrjns][RecvNeurons] starting offset and N cons for each recv neuron, for indexing into the RecvSynIdx array of indexes into the Syns synapses, which are organized sender-based."`

	// [view: -] [Layers][RecvPrjns][RecvNeurons][MaxDelay][MaxData] conductance buffer for accumulating spikes -- subslices are allocated to each projection -- uses int-encoded float values for faster GPU atomic integration
	PrjnGBuf []int32 `view:"-" desc:"[Layers][RecvPrjns][RecvNeurons][MaxDelay][MaxData] conductance buffer for accumulating spikes -- subslices are allocated to each projection -- uses int-encoded float values for faster GPU atomic integration"`

	// [view: -] [Layers][RecvPrjns][RecvNeurons][MaxData] synaptic conductance integrated over time per projection per recv neurons -- spikes come in via PrjnBuf -- subslices are allocated to each projection
	PrjnGSyns []float32 `view:"-" desc:"[Layers][RecvPrjns][RecvNeurons][MaxData] synaptic conductance integrated over time per projection per recv neurons -- spikes come in via PrjnBuf -- subslices are allocated to each projection"`

	// [view: -] [Layers][RecvPrjns] indexes into Prjns (organized by SendPrjn) organized by recv projections -- needed for iterating through recv prjns efficiently on GPU.
	RecvPrjnIdxs []uint32 `view:"-" desc:"[Layers][RecvPrjns] indexes into Prjns (organized by SendPrjn) organized by recv projections -- needed for iterating through recv prjns efficiently on GPU."`

	// [view: -] [Layers][RecvPrjns][RecvNeurons][Syns] indexes into Synapses for each recv neuron, organized into blocks according to PrjnRecvCon, for receiver-based access.
	RecvSynIdxs []uint32 `view:"-" desc:"[Layers][RecvPrjns][RecvNeurons][Syns] indexes into Synapses for each recv neuron, organized into blocks according to PrjnRecvCon, for receiver-based access."`

	// [In / Targ Layers][Neurons][Data] external input values for all Input / Target / Compare layers in the network -- the ApplyExt methods write to this per layer, and it is then actually applied in one consistent method.
	Exts []float32 `desc:"[In / Targ Layers][Neurons][Data] external input values for all Input / Target / Compare layers in the network -- the ApplyExt methods write to this per layer, and it is then actually applied in one consistent method."`

	// [view: -] context used only for accessing neurons for display -- NetIdxs.NData in here is copied from active context in NewState
	Ctx Context `view:"-" desc:"context used only for accessing neurons for display -- NetIdxs.NData in here is copied from active context in NewState"`

	// [view: -] random number generator for the network -- all random calls must use this -- set seed here for weight initialization values
	Rand erand.SysRand `view:"-" desc:"random number generator for the network -- all random calls must use this -- set seed here for weight initialization values"`

	// random seed to be set at the start of configuring the network and initializing the weights -- set this to get a different set of weights
	RndSeed int64 `inactive:"+" desc:"random seed to be set at the start of configuring the network and initializing the weights -- set this to get a different set of weights"`

	// number of threads to use for parallel processing
	NThreads int `desc:"number of threads to use for parallel processing"`

	// [view: inline] GPU implementation
	GPU GPU `view:"inline" desc:"GPU implementation"`

	// [view: -] record function timer information
	RecFunTimes bool `view:"-" desc:"record function timer information"`

	// [view: -] timers for each major function (step of processing)
	FunTimes map[string]*timer.Time `view:"-" desc:"timers for each major function (step of processing)"`
}

// emer.Network interface methods:
func (nt *NetworkBase) Name() string                  { return nt.Nm }
func (nt *NetworkBase) Label() string                 { return nt.Nm }
func (nt *NetworkBase) NLayers() int                  { return len(nt.Layers) }
func (nt *NetworkBase) Layer(idx int) emer.Layer      { return nt.Layers[idx] }
func (nt *NetworkBase) Bounds() (min, max mat32.Vec3) { min = nt.MinPos; max = nt.MaxPos; return }
func (nt *NetworkBase) MaxParallelData() int          { return int(nt.MaxData) }
func (nt *NetworkBase) NParallelData() int            { return int(nt.Ctx.NetIdxs.NData) }

// LayByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *NetworkBase) AxonLayerByName(name string) *Layer {
	if nt.LayMap == nil || len(nt.LayMap) != len(nt.Layers) {
		nt.MakeLayMap()
	}
	ly := nt.LayMap[name]
	return ly
}

// LayByNameTry returns a layer by looking it up by name -- returns error message
// if layer is not found
func (nt *NetworkBase) LayByNameTry(name string) (*Layer, error) {
	ly := nt.AxonLayerByName(name)
	if ly == nil {
		err := fmt.Errorf("Layer named: %v not found in Network: %v\n", name, nt.Nm)
		// log.Println(err)
		return ly, err
	}
	return ly, nil
}

// LayByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *NetworkBase) LayerByName(name string) emer.Layer {
	return nt.AxonLayerByName(name)
}

// LayerByNameTry returns a layer by looking it up by name -- returns error message
// if layer is not found
func (nt *NetworkBase) LayerByNameTry(name string) (emer.Layer, error) {
	return nt.LayByNameTry(name)
}

// MakeLayMap updates layer map based on current layers
func (nt *NetworkBase) MakeLayMap() {
	nt.LayMap = make(map[string]*Layer, len(nt.Layers))
	for _, ly := range nt.Layers {
		nt.LayMap[ly.Name()] = ly
	}
}

// LayersByType returns a list of layer names by given layer types.
// Lists are compiled when network Build() function called.
// The layer Type is always included as a Class, along with any other
// space-separated strings specified in Class for parameter styling, etc.
// If no classes are passed, all layer names in order are returned.
func (nt *NetworkBase) LayersByType(layType ...LayerTypes) []string {
	var nms []string
	for _, tp := range layType {
		nm := tp.String()
		nms = append(nms, nm)
	}
	return nt.LayersByClass(nms...)
}

// LayersByClass returns a list of layer names by given class(es).
// Lists are compiled when network Build() function called.
// The layer Type is always included as a Class, along with any other
// space-separated strings specified in Class for parameter styling, etc.
// If no classes are passed, all layer names in order are returned.
func (nt *NetworkBase) LayersByClass(classes ...string) []string {
	var nms []string
	if len(classes) == 0 {
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			nms = append(nms, ly.Name())
		}
		return nms
	}
	for _, lc := range classes {
		nms = append(nms, nt.LayClassMap[lc]...)
	}
	layers := dedupe.DeDupe(nms)
	if len(layers) == 0 {
		panic(fmt.Sprintf("No Layers found for query: %#v. Basic layer types have been renamed since v1.7, use LayersByType for forward compatibility.", classes))
	}
	return layers
}

// LayerVal returns LayerVals for given layer and data parallel indexes
func (nt *NetworkBase) LayerVals(li, di uint32) *LayerVals {
	return &nt.LayVals[li*nt.MaxData+di]
}

// UnitVarNames returns a list of variable names available on the units in this network.
// Not all layers need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *NetworkBase) UnitVarNames() []string {
	return NeuronVarNames
}

// UnitVarProps returns properties for variables
func (nt *NetworkBase) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// SynVarNames returns the names of all the variables on the synapses in this network.
// Not all projections need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *NetworkBase) SynVarNames() []string {
	return SynapseVarNames
}

// SynVarProps returns properties for variables
func (nt *NetworkBase) SynVarProps() map[string]string {
	return SynapseVarProps
}

// StdVertLayout arranges layers in a standard vertical (z axis stack) layout, by setting
// the Rel settings
func (nt *NetworkBase) StdVertLayout() {
	lstnm := ""
	for li, ly := range nt.Layers {
		if li == 0 {
			ly.SetRelPos(relpos.Rel{Rel: relpos.NoRel})
			lstnm = ly.Name()
		} else {
			ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstnm, XAlign: relpos.Middle, YAlign: relpos.Front})
		}
	}
}

// Layout computes the 3D layout of layers based on their relative position settings
func (nt *NetworkBase) Layout() {
	for itr := 0; itr < 5; itr++ {
		var lstly *Layer
		for _, ly := range nt.Layers {
			rp := ly.RelPos()
			var oly emer.Layer
			if lstly != nil && rp.Rel == relpos.NoRel {
				if ly.Pos().X != 0 || ly.Pos().Y != 0 || ly.Pos().Z != 0 {
					// Position has been modified, don't mess with it.
					continue
				}
				oly = lstly
				ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
			} else {
				if rp.Other != "" {
					var err error
					oly, err = nt.LayByNameTry(rp.Other)
					if err != nil {
						log.Println(err)
						continue
					}
				} else if lstly != nil {
					oly = lstly
					ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
				}
			}
			if oly != nil {
				ly.SetPos(rp.Pos(oly.Pos(), oly.Size(), ly.Size()))
			}
			lstly = ly
		}
	}
	nt.BoundsUpdt()
}

// BoundsUpdt updates the Min / Max display bounds for 3D display
func (nt *NetworkBase) BoundsUpdt() {
	mn := mat32.NewVec3Scalar(mat32.Infinity)
	mx := mat32.Vec3Zero
	for _, ly := range nt.Layers {
		ps := ly.Pos()
		sz := ly.Size()
		ru := ps
		ru.X += sz.X
		ru.Y += sz.Y
		mn.SetMax(ps)
		mx.SetMax(ru)
	}
	nt.MaxPos = mn
	nt.MaxPos = mx
}

// ParamsHistoryReset resets parameter application history
func (nt *NetworkBase) ParamsHistoryReset() {
	for _, ly := range nt.Layers {
		ly.ParamsHistoryReset()
	}
}

// ParamsApplied is just to satisfy History interface so reset can be applied
func (nt *NetworkBase) ParamsApplied(sel *params.Sel) {
}

// ApplyParams applies given parameter style Sheet to layers and prjns in this network.
// Calls UpdateParams to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (nt *NetworkBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	for _, ly := range nt.Layers {
		app, err := ly.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Network that
// are not at their default values -- useful for setting param styles etc.
func (nt *NetworkBase) NonDefaultParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.NonDefaultParams()
		nds += nd
	}
	return nds
}

// AllParams returns a listing of all parameters in the Network.
func (nt *NetworkBase) AllParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.AllParams()
		nds += nd
	}
	return nds
}

// KeyLayerParams returns a listing for all layers in the network,
// of the most important layer-level params (specific to each algorithm).
func (nt *NetworkBase) KeyLayerParams() string {
	return nt.AllLayerInhibs()
}

// KeyPrjnParams returns a listing for all Recv projections in the network,
// of the most important projection-level params (specific to each algorithm).
func (nt *NetworkBase) KeyPrjnParams() string {
	return nt.AllPrjnScales()
}

// AllLayerInhibs returns a listing of all Layer Inhibition parameters in the Network
func (nt *NetworkBase) AllLayerInhibs() string {
	str := ""
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		lp := ly.Params
		ph := ly.ParamsHistory.ParamsHistory()
		lh := ph["Layer.Inhib.ActAvg.Nominal"]
		if lh != "" {
			lh = "Params: " + lh
		}
		str += fmt.Sprintf("%15s\t\tNominal:\t%6.2f\t%s\n", ly.Name(), lp.Inhib.ActAvg.Nominal, lh)
		if lp.Inhib.Layer.On.IsTrue() {
			lh := ph["Layer.Inhib.Layer.Gi"]
			if lh != "" {
				lh = "Params: " + lh
			}
			str += fmt.Sprintf("\t\t\t\t\t\tLayer.Gi:\t%6.2f\t%s\n", lp.Inhib.Layer.Gi, lh)
		}
		if lp.Inhib.Pool.On.IsTrue() {
			lh := ph["Layer.Inhib.Pool.Gi"]
			if lh != "" {
				lh = "Params: " + lh
			}
			str += fmt.Sprintf("\t\t\t\t\t\tPool.Gi: \t%6.2f\t%s\n", lp.Inhib.Pool.Gi, lh)
		}
		str += fmt.Sprintf("\n")
	}
	return str
}

// AllPrjnScales returns a listing of all PrjnScale parameters in the Network
// in all Layers, Recv projections.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *NetworkBase) AllPrjnScales() string {
	str := ""
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		str += "\nLayer: " + ly.Name() + "\n"
		for i := 0; i < ly.NRecvPrjns(); i++ {
			pj := ly.RcvPrjns[i]
			if pj.IsOff() {
				continue
			}
			sn := pj.Send.Name()
			str += fmt.Sprintf("\t%15s\t%15s\tAbs:\t%6.2f\tRel:\t%6.2f\tGScale:\t%6.2f\tRel:%6.2f\n", sn, pj.PrjnType().String(), pj.Params.PrjnScale.Abs, pj.Params.PrjnScale.Rel, pj.Params.GScale.Scale, pj.Params.GScale.Rel)
			ph := pj.ParamsHistory.ParamsHistory()
			rh := ph["Prjn.PrjnScale.Rel"]
			ah := ph["Prjn.PrjnScale.Abs"]
			if ah != "" {
				str += fmt.Sprintf("\t\t\t\t\t\t\t\t    Abs Params: %s\n", ah)
			}
			if rh != "" {
				str += fmt.Sprintf("\t\t\t\t\t\t\t\t    Rel Params: %s\n", rh)
			}
		}
	}
	return str
}

// SaveParamsSnapshot saves various views of current parameters
// to either `params_good` if good = true (for current good reference params)
// or `params_2006_01_02` (year, month, day) datestamp,
// providing a snapshot of the simulation params for easy diffs and later reference.
// Also saves current Config and Params state.
func (nt *NetworkBase) SaveParamsSnapshot(pars *netparams.Sets, cfg any, good bool) error {
	date := time.Now().Format("2006_01_02")
	if good {
		date = "good"
	}
	dir := "params_" + date
	err := os.Mkdir(dir, 0775)
	if err != nil {
		log.Println(err) // notify but OK if it exists
	}
	econfig.Save(cfg, filepath.Join(dir, "config.toml"))
	pars.SaveTOML(gi.FileName(filepath.Join(dir, "params.toml")))
	nt.SaveAllParams(gi.FileName(filepath.Join(dir, "params_all.txt")))
	nt.SaveNonDefaultParams(gi.FileName(filepath.Join(dir, "params_nondef.txt")))
	nt.SaveAllLayerInhibs(gi.FileName(filepath.Join(dir, "params_layers.txt")))
	nt.SaveAllPrjnScales(gi.FileName(filepath.Join(dir, "params_prjns.txt")))
	return nil
}

// SaveAllParams saves list of all parameters in Network to given file.
func (nt *NetworkBase) SaveAllParams(filename gi.FileName) error {
	str := nt.AllParams()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SaveNonDefaultParams saves list of all non-default parameters in Network to given file.
func (nt *NetworkBase) SaveNonDefaultParams(filename gi.FileName) error {
	str := nt.NonDefaultParams()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SaveAllLayerInhibs saves list of all layer Inhibition parameters to given file
func (nt *NetworkBase) SaveAllLayerInhibs(filename gi.FileName) error {
	str := nt.AllLayerInhibs()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SavePrjnScales saves a listing of all PrjnScale parameters in the Network
// in all Layers, Recv projections.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *NetworkBase) SaveAllPrjnScales(filename gi.FileName) error {
	str := nt.AllPrjnScales()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// AllGlobals returns a listing of all Global variables and values.
func (nt *NetworkBase) AllGlobals() string {
	ctx := &nt.Ctx
	str := ""
	for di := uint32(0); di < nt.MaxData; di++ {
		str += fmt.Sprintf("\n###############################\nData Index: %02d\n\n", di)
		for vv := GvRew; vv < GvUSneg; vv++ {
			str += fmt.Sprintf("%20s:\t%7.4f\n", vv.String(), GlbV(ctx, di, vv))
		}
		str += fmt.Sprintf("%20s:\t", "USNeg")
		for ui := uint32(0); ui < ctx.NetIdxs.PVLVNNegUSs; ui++ {
			str += fmt.Sprintf("%d: %7.4f\t", ui, GlbUSneg(ctx, di, GvUSneg, ui))
		}
		str += "\n"
		str += fmt.Sprintf("%20s:\t", "USNegRaw")
		for ui := uint32(0); ui < ctx.NetIdxs.PVLVNNegUSs; ui++ {
			str += fmt.Sprintf("%d: %7.4f\t", ui, GlbUSneg(ctx, di, GvUSnegRaw, ui))
		}
		str += "\n"
		for vv := GvDrives; vv < GlobalVarsN; vv++ {
			str += fmt.Sprintf("%20s:\t", vv.String())
			for ui := uint32(0); ui < ctx.NetIdxs.PVLVNPosUSs; ui++ {
				str += fmt.Sprintf("%d:\t%7.4f\t", ui, GlbDrvV(ctx, di, ui, vv))
			}
			str += "\n"
		}
	}
	return str
}

// AllGlobalVals adds to map of all Global variables and values.
// ctrKey is a key of counters to contextualize values.
func (nt *NetworkBase) AllGlobalVals(ctrKey string, vals map[string]float32) {
	ctx := &nt.Ctx
	for di := uint32(0); di < nt.MaxData; di++ {
		for vv := GvRew; vv < GvUSneg; vv++ {
			key := fmt.Sprintf("%s  Di: %d\t%s", ctrKey, di, vv.String())
			vals[key] = GlbV(ctx, di, vv)
		}
		for ui := uint32(0); ui < ctx.NetIdxs.PVLVNNegUSs; ui++ {
			key := fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, "USneg", ui)
			vals[key] = GlbUSneg(ctx, di, GvUSneg, ui)
			key = fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, "USnegRaw", ui)
			vals[key] = GlbUSneg(ctx, di, GvUSnegRaw, ui)
		}
		for vv := GvDrives; vv < GlobalVarsN; vv++ {
			for ui := uint32(0); ui < ctx.NetIdxs.PVLVNPosUSs; ui++ {
				key := fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, vv.String(), ui)
				vals[key] = GlbDrvV(ctx, di, ui, vv)
			}
		}
	}
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *NetworkBase) AddLayerInit(ly *Layer, name string, shape []int, typ LayerTypes) {
	if nt.EmerNet == nil {
		log.Printf("Network EmerNet is nil -- you MUST call InitName on network, passing a pointer to the network to initialize properly!")
		return
	}
	ly.InitName(ly, name, nt.EmerNet)
	ly.Config(shape, emer.LayerType(typ))
	nt.Layers = append(nt.Layers, ly)
	ly.SetIndex(len(nt.Layers) - 1)
	nt.MakeLayMap()
}

// AddLayer adds a new layer with given name and shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential -- see
// AddLayer2D and 4D for convenience methods for those.  4D layers enable
// pool (unit-group) level inhibition in Axon networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each unit
// group having 4 rows (Y) of 5 (X) units.
func (nt *NetworkBase) AddLayer(name string, shape []int, typ LayerTypes) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *NetworkBase) AddLayer2D(name string, shapeY, shapeX int, typ LayerTypes) *Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Axon networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *NetworkBase) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ LayerTypes) *Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a projection between two layers, referenced by name
// adding to the recv and send projection lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) ConnectLayerNames(send, recv string, pat prjn.Pattern, typ PrjnTypes) (rlay, slay *Layer, pj *Prjn, err error) {
	rlay, err = nt.LayByNameTry(recv)
	if err != nil {
		return
	}
	slay, err = nt.LayByNameTry(send)
	if err != nil {
		return
	}
	pj = nt.ConnectLayers(slay, rlay, pat, typ)
	return
}

// ConnectLayers establishes a projection between two layers,
// adding to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) ConnectLayers(send, recv *Layer, pat prjn.Pattern, typ PrjnTypes) *Prjn {
	pj := &Prjn{}
	pj.Init(pj)
	pj.Connect(send, recv, pat, typ)
	recv.RcvPrjns.Add(pj)
	send.SndPrjns.Add(pj)
	return pj
}

// BidirConnectLayerNames establishes bidirectional projections between two layers,
// referenced by name, with low = the lower layer that sends a Forward projection
// to the high layer, and receives a Back projection in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) BidirConnectLayerNames(low, high string, pat prjn.Pattern) (lowlay, highlay *Layer, fwdpj, backpj *Prjn, err error) {
	lowlay, err = nt.LayByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, ForwardPrjn)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, BackPrjn)
	return
}

// BidirConnectLayers establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) BidirConnectLayers(low, high *Layer, pat prjn.Pattern) (fwdpj, backpj *Prjn) {
	fwdpj = nt.ConnectLayers(low, high, pat, ForwardPrjn)
	backpj = nt.ConnectLayers(high, low, pat, BackPrjn)
	return
}

// BidirConnectLayersPy establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *NetworkBase) BidirConnectLayersPy(low, high *Layer, pat prjn.Pattern) {
	nt.ConnectLayers(low, high, pat, ForwardPrjn)
	nt.ConnectLayers(high, low, pat, BackPrjn)
}

// LateralConnectLayer establishes a self-projection within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayer(lay *Layer, pat prjn.Pattern) *Prjn {
	pj := &Prjn{}
	return nt.LateralConnectLayerPrjn(lay, pat, pj)
}

// LateralConnectLayerPrjn makes lateral self-projection using given projection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayerPrjn(lay *Layer, pat prjn.Pattern, pj *Prjn) *Prjn {
	pj.Init(pj)
	pj.Connect(lay, lay, pat, LateralPrjn)
	lay.RcvPrjns.Add(pj)
	lay.SndPrjns.Add(pj)
	return pj
}

// SetCtxStrides sets the given simulation context strides for accessing
// variables on this network -- these must be set properly before calling
// any compute methods with the context.
func (nt *NetworkBase) SetCtxStrides(simCtx *Context) {
	simCtx.CopyNetStridesFrom(&nt.Ctx)
}

// SetMaxData sets the MaxData and current NData for both the Network and the Context
func (nt *NetworkBase) SetMaxData(simCtx *Context, maxData int) {
	nt.MaxData = uint32(maxData)
	simCtx.NetIdxs.NData = uint32(maxData)
	simCtx.NetIdxs.MaxData = uint32(maxData)
}

// Build constructs the layer and projection state based on the layer shapes
// and patterns of interconnectivity. Configures threading using heuristics based
// on final network size.  Must set UseGPUOrder properly prior to calling.
// Configures the given Context object used in the simulation with the memory
// access strides for this network -- must be set properly -- see SetCtxStrides.
func (nt *NetworkBase) Build(simCtx *Context) error {
	nt.UseGPUOrder = true // todo: set externally
	ctx := &nt.Ctx
	*ctx = *simCtx
	ctx.NetIdxs.NetIdx = nt.NetIdx
	nt.FunTimes = make(map[string]*timer.Time)
	nt.LayClassMap = make(map[string][]string)
	maxData := int(nt.MaxData)
	emsg := ""
	totNeurons := 0
	totPrjns := 0
	totExts := 0
	nLayers := len(nt.Layers)
	totPools := nLayers // layer pool for each layer at least
	for _, ly := range nt.Layers {
		if ly.IsOff() { // note: better not turn on later!
			continue
		}
		totPools += ly.NSubPools()
		nn := ly.Shape().Len()
		totNeurons += nn
		if ly.LayerType().IsExt() {
			totExts += nn
		}
		totPrjns += ly.NSendPrjns() // either way
		cls := strings.Split(ly.Class(), " ")
		for _, cl := range cls {
			ll := nt.LayClassMap[cl]
			ll = append(ll, ly.Name())
			nt.LayClassMap[cl] = ll
		}
	}
	nt.LayParams = make([]LayerParams, nLayers)
	nt.LayVals = make([]LayerVals, nLayers*maxData)
	nt.Pools = make([]Pool, totPools*maxData)
	nt.NNeurons = uint32(totNeurons)
	nneurv := uint32(totNeurons) * nt.MaxData * uint32(NeuronVarsN)
	nt.Neurons = make([]float32, nneurv)
	nneurav := uint32(totNeurons) * uint32(NeuronAvgVarsN)
	nt.NeuronAvgs = make([]float32, nneurav)
	nneuri := uint32(totNeurons) * uint32(NeuronIdxsN)
	nt.NeuronIxs = make([]uint32, nneuri)
	nt.Prjns = make([]*Prjn, totPrjns)
	nt.PrjnParams = make([]PrjnParams, totPrjns)
	nt.Exts = make([]float32, totExts*maxData)

	if nt.UseGPUOrder {
		ctx.NeuronVars.SetVarOuter(totNeurons, maxData)
		ctx.NeuronAvgVars.SetVarOuter(totNeurons)
		ctx.NeuronIdxs.SetIdxOuter(totNeurons)
	} else {
		ctx.NeuronVars.SetNeuronOuter(maxData)
		ctx.NeuronAvgVars.SetNeuronOuter()
		ctx.NeuronIdxs.SetNeuronOuter()
	}

	totSynapses := 0
	totRecvCon := 0
	totSendCon := 0
	neurIdx := 0
	prjnIdx := 0
	rprjnIdx := 0
	poolIdx := 0
	extIdx := 0
	for li, ly := range nt.Layers {
		ly.Params = &nt.LayParams[li]
		ly.Params.LayType = LayerTypes(ly.Typ)
		ly.Vals = nt.LayVals[li*maxData : (li+1)*maxData]
		if ly.IsOff() {
			continue
		}
		shp := ly.Shape()
		nn := shp.Len()
		ly.NNeurons = uint32(nn)
		ly.NeurStIdx = uint32(neurIdx)
		ly.MaxData = nt.MaxData
		np := ly.NSubPools() + 1
		npd := np * maxData
		ly.NPools = uint32(np)
		ly.Pools = nt.Pools[poolIdx : poolIdx+npd]
		ly.Params.Idxs.LayIdx = uint32(li)
		ly.Params.Idxs.MaxData = nt.MaxData
		ly.Params.Idxs.PoolSt = uint32(poolIdx)
		ly.Params.Idxs.NeurSt = uint32(neurIdx)
		ly.Params.Idxs.NeurN = uint32(nn)
		if shp.NumDims() == 2 {
			ly.Params.Idxs.ShpUnY = int32(shp.Dim(0))
			ly.Params.Idxs.ShpUnX = int32(shp.Dim(1))
			ly.Params.Idxs.ShpPlY = 1
			ly.Params.Idxs.ShpPlX = 1
		} else {
			ly.Params.Idxs.ShpPlY = int32(shp.Dim(0))
			ly.Params.Idxs.ShpPlX = int32(shp.Dim(1))
			ly.Params.Idxs.ShpUnY = int32(shp.Dim(2))
			ly.Params.Idxs.ShpUnX = int32(shp.Dim(3))
		}
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Vals[di].LayIdx = uint32(li)
			ly.Vals[di].DataIdx = uint32(di)
		}
		for pi := 0; pi < np; pi++ {
			for di := 0; di < maxData; di++ {
				ix := pi*int(ly.MaxData) + di
				pl := &ly.Pools[ix]
				pl.LayIdx = uint32(li)
				pl.DataIdx = uint32(di)
				pl.PoolIdx = uint32(poolIdx + ix)
			}
		}
		if ly.LayerType().IsExt() {
			ly.Exts = nt.Exts[extIdx : extIdx+nn*maxData]
			ly.Params.Idxs.ExtsSt = uint32(extIdx)
			extIdx += nn * maxData
		} else {
			ly.Exts = nil
			ly.Params.Idxs.ExtsSt = 0 // sticking with uint32 here -- otherwise could be -1
		}
		sprjns := *ly.SendPrjns()
		ly.Params.Idxs.SendSt = uint32(prjnIdx)
		ly.Params.Idxs.SendN = uint32(len(sprjns))
		for pi, pj := range sprjns {
			pii := prjnIdx + pi
			pj.Params = &nt.PrjnParams[pii]
			nt.Prjns[pii] = pj
		}
		err := ly.Build() // also builds prjns and sets SubPool indexes
		if err != nil {
			emsg += err.Error() + "\n"
		}
		// now collect total number of synapses after layer build
		for _, pj := range sprjns {
			totSynapses += len(pj.SendConIdx)
			totSendCon += nn // sep vals for each send neuron per prjn
		}
		rprjns := *ly.RecvPrjns()
		ly.Params.Idxs.RecvSt = uint32(rprjnIdx)
		ly.Params.Idxs.RecvN = uint32(len(rprjns))
		totRecvCon += nn * len(rprjns)
		rprjnIdx += len(rprjns)
		neurIdx += nn
		prjnIdx += len(sprjns)
		poolIdx += npd
	}
	if totSynapses > math.MaxUint32 {
		log.Fatalf("ERROR: total number of synapses is greater than uint32 capacity\n")
	}

	nt.NSyns = uint32(totSynapses)
	nSynFloat := totSynapses * int(SynapseVarsN)
	nt.Synapses = make([]float32, nSynFloat)
	nSynCaFloat := totSynapses * int(SynapseCaVarsN) * int(nt.MaxData)
	nt.SynapseCas = make([]float32, nSynCaFloat)
	nt.SynapseIxs = make([]uint32, totSynapses*int(SynapseIdxsN))
	nt.PrjnSendCon = make([]StartN, totSendCon)
	nt.PrjnRecvCon = make([]StartN, totRecvCon)
	nt.RecvPrjnIdxs = make([]uint32, rprjnIdx)
	nt.RecvSynIdxs = make([]uint32, totSynapses)

	if nt.UseGPUOrder {
		ctx.SynapseVars.SetVarOuter(totSynapses)
		ctx.SynapseCaVars.SetVarOuter(totSynapses, maxData)
		ctx.SynapseIdxs.SetIdxOuter(totSynapses)
	} else {
		ctx.SynapseVars.SetSynapseOuter()
		ctx.SynapseCaVars.SetSynapseOuter(maxData)
		ctx.SynapseIdxs.SetSynapseOuter()
	}

	// distribute synapses, send
	syIdx := 0
	pjidx := 0
	sendConIdx := 0
	for _, ly := range nt.Layers {
		for _, pj := range ly.SndPrjns {
			rlay := pj.Recv
			pj.Params.Idxs.RecvLay = uint32(rlay.Idx)
			pj.Params.Idxs.RecvNeurSt = uint32(rlay.NeurStIdx)
			pj.Params.Idxs.RecvNeurN = rlay.NNeurons
			pj.Params.Idxs.SendLay = uint32(ly.Idx)
			pj.Params.Idxs.SendNeurSt = uint32(ly.NeurStIdx)
			pj.Params.Idxs.SendNeurN = ly.NNeurons

			nsyn := len(pj.SendConIdx)
			pj.Params.Idxs.SendConSt = uint32(sendConIdx)
			pj.Params.Idxs.SynapseSt = uint32(syIdx)
			pj.SynStIdx = uint32(syIdx)
			pj.Params.Idxs.PrjnIdx = uint32(pjidx)
			pj.NSyns = uint32(nsyn)
			for sni := uint32(0); sni < ly.NNeurons; sni++ {
				si := ly.NeurStIdx + sni
				scon := pj.SendCon[sni]
				nt.PrjnSendCon[sendConIdx] = scon
				sendConIdx++
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIdx + syi
					SetSynI(ctx, syni, SynSendIdx, uint32(si)) // network-global idx
					SetSynI(ctx, syni, SynRecvIdx, pj.SendConIdx[syi]+uint32(rlay.NeurStIdx))
					SetSynI(ctx, syni, SynPrjnIdx, uint32(pjidx))
					syIdx++
				}
			}
			pjidx++
		}
	}

	// update recv synapse / prjn info
	rprjnIdx = 0
	recvConIdx := 0
	syIdx = 0
	for _, ly := range nt.Layers {
		for _, pj := range ly.RcvPrjns {
			nt.RecvPrjnIdxs[rprjnIdx] = pj.Params.Idxs.PrjnIdx
			pj.Params.Idxs.RecvConSt = uint32(recvConIdx)
			pj.Params.Idxs.RecvSynSt = uint32(syIdx)
			synSt := pj.Params.Idxs.SynapseSt
			for rni := uint32(0); rni < ly.NNeurons; rni++ {
				if len(pj.RecvCon) <= int(rni) {
					continue
				}
				rcon := pj.RecvCon[rni]
				nt.PrjnRecvCon[recvConIdx] = rcon
				recvConIdx++
				syIdxs := pj.RecvSynIdxs(rni)
				for _, ssi := range syIdxs {
					nt.RecvSynIdxs[syIdx] = ssi + synSt
					syIdx++
				}
			}
			rprjnIdx++
		}
	}

	ctx.NetIdxs.MaxData = nt.MaxData
	ctx.NetIdxs.NLayers = uint32(nLayers)
	ctx.NetIdxs.NNeurons = nt.NNeurons
	ctx.NetIdxs.NPools = uint32(totPools)
	ctx.NetIdxs.NSyns = nt.NSyns
	ctx.NetIdxs.PVLVNPosUSs = nt.PVLV.NPosUSs
	ctx.NetIdxs.PVLVNNegUSs = nt.PVLV.NNegUSs
	ctx.SetGlobalStrides()

	nt.SetCtxStrides(simCtx)
	nt.BuildGlobals(simCtx)

	nt.Layout()
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// BuildPrjnGBuf builds the PrjnGBuf, PrjnGSyns,
// based on the MaxDelay values in thePrjnParams,
// which should have been configured by this point.
// Called by default in InitWts()
func (nt *NetworkBase) BuildPrjnGBuf() {
	nt.MaxDelay = 0
	npjneur := uint32(0)
	for _, ly := range nt.Layers {
		nneur := uint32(ly.NNeurons)
		for _, pj := range ly.RcvPrjns {
			if pj.Params.Com.MaxDelay > nt.MaxDelay {
				nt.MaxDelay = pj.Params.Com.MaxDelay
			}
			npjneur += nneur
		}
	}
	mxlen := nt.MaxDelay + 1
	gbsz := npjneur * mxlen * nt.MaxData
	gsynsz := npjneur * nt.MaxData
	if uint32(cap(nt.PrjnGBuf)) >= gbsz {
		nt.PrjnGBuf = nt.PrjnGBuf[:gbsz]
	} else {
		nt.PrjnGBuf = make([]int32, gbsz)
	}
	if uint32(cap(nt.PrjnGSyns)) >= gsynsz {
		nt.PrjnGSyns = nt.PrjnGSyns[:gsynsz]
	} else {
		nt.PrjnGSyns = make([]float32, gsynsz)
	}

	gbi := uint32(0)
	gsi := uint32(0)
	for _, ly := range nt.Layers {
		nneur := uint32(ly.NNeurons)
		for _, pj := range ly.RcvPrjns {
			gbs := nneur * mxlen * nt.MaxData
			pj.Params.Idxs.GBufSt = gbi
			pj.GBuf = nt.PrjnGBuf[gbi : gbi+gbs]
			gbi += gbs
			pj.Params.Idxs.GSynSt = gsi
			pj.GSyns = nt.PrjnGSyns[gsi : gsi+nneur*nt.MaxData]
			gsi += nneur * nt.MaxData
		}
	}
}

// BuildGlobals builds Globals vars, using params set in given context
func (nt *NetworkBase) BuildGlobals(ctx *Context) {
	nt.Globals = make([]float32, ctx.GlobalVNFloats())
}

// DeleteAll deletes all layers, prepares network for re-configuring and building
func (nt *NetworkBase) DeleteAll() {
	nt.Layers = nil
	nt.LayMap = nil
	nt.FunTimes = nil
	nt.Globals = nil
	nt.LayParams = nil
	nt.LayVals = nil
	nt.Pools = nil
	nt.Neurons = nil
	nt.NeuronAvgs = nil
	nt.NeuronIxs = nil
	nt.Prjns = nil
	nt.PrjnParams = nil
	nt.Synapses = nil
	nt.SynapseCas = nil
	nt.SynapseIxs = nil
	nt.PrjnSendCon = nil
	nt.PrjnRecvCon = nil
	nt.PrjnGBuf = nil
	nt.PrjnGSyns = nil
	nt.RecvPrjnIdxs = nil
	nt.Exts = nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Weights File

// SaveWtsJSON saves network weights (and any other state that adapts with learning)
// to a JSON-formatted file.  If filename has .gz extension, then file is gzip compressed.
func (nt *NetworkBase) SaveWtsJSON(filename gi.FileName) error {
	fp, err := os.Create(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr := gzip.NewWriter(fp)
		err = nt.WriteWtsJSON(gzr)
		gzr.Close()
	} else {
		bw := bufio.NewWriter(fp)
		err = nt.WriteWtsJSON(bw)
		bw.Flush()
	}
	return err
}

// OpenWtsJSON opens network weights (and any other state that adapts with learning)
// from a JSON-formatted file.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *NetworkBase) OpenWtsJSON(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr, err := gzip.NewReader(fp)
		defer gzr.Close()
		if err != nil {
			log.Println(err)
			return err
		}
		return nt.ReadWtsJSON(gzr)
	} else {
		return nt.ReadWtsJSON(bufio.NewReader(fp))
	}
}

// todo: proper error handling here!

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (nt *NetworkBase) WriteWtsJSON(w io.Writer) error {
	nt.GPU.SyncAllFmGPU()
	depth := 0
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Network\": %q,\n", nt.Nm))) // note: can't use \n in `` so need "
	w.Write(indent.TabBytes(depth))
	onls := make([]emer.Layer, 0, len(nt.Layers))
	for _, ly := range nt.Layers {
		if !ly.IsOff() {
			onls = append(onls, ly)
		}
	}
	nl := len(onls)
	if nl == 0 {
		w.Write([]byte("\"Layers\": null\n"))
	} else {
		w.Write([]byte("\"Layers\": [\n"))
		depth++
		for li, ly := range onls {
			ly.WriteWtsJSON(w, depth)
			if li == nl-1 {
				w.Write([]byte("\n"))
			} else {
				w.Write([]byte(",\n"))
			}
		}
		depth--
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("]\n"))
	}
	depth--
	w.Write(indent.TabBytes(depth))
	_, err := w.Write([]byte("}\n"))
	return err
}

// ReadWtsJSON reads network weights from the receiver-side perspective
// in a JSON text format.  Reads entire file into a temporary weights.Weights
// structure that is then passed to Layers etc using SetWts method.
func (nt *NetworkBase) ReadWtsJSON(r io.Reader) error {
	nw, err := weights.NetReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	err = nt.SetWts(nw)
	if err != nil {
		log.Println(err)
	}
	nt.GPU.SyncAllToGPU() // needs loaded adapting layer params too
	return err
}

// SetWts sets the weights for this network from weights.Network decoded values
func (nt *NetworkBase) SetWts(nw *weights.Network) error {
	var err error
	if nw.Network != "" {
		nt.Nm = nw.Network
	}
	if nw.MetaData != nil {
		if nt.MetaData == nil {
			nt.MetaData = nw.MetaData
		} else {
			for mk, mv := range nw.MetaData {
				nt.MetaData[mk] = mv
			}
		}
	}
	for li := range nw.Layers {
		lw := &nw.Layers[li]
		ly, er := nt.LayByNameTry(lw.Layer)
		if er != nil {
			err = er
			continue
		}
		ly.SetWts(lw)
	}
	return err
}

// OpenWtsCpp opens network weights (and any other state that adapts with learning)
// from old C++ emergent format.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *NetworkBase) OpenWtsCpp(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr, err := gzip.NewReader(fp)
		defer gzr.Close()
		if err != nil {
			log.Println(err)
			return err
		}
		return nt.ReadWtsCpp(gzr)
	} else {
		return nt.ReadWtsCpp(fp)
	}
}

// ReadWtsCpp reads the weights from old C++ emergent format.
// Reads entire file into a temporary weights.Weights
// structure that is then passed to Layers etc using SetWts method.
func (nt *NetworkBase) ReadWtsCpp(r io.Reader) error {
	nw, err := weights.NetReadCpp(r)
	if err != nil {
		return err // note: already logged
	}
	err = nt.SetWts(nw)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SynsSlice returns a slice of synaptic values, in natural sending order,
// using given synaptic variable, resizing as needed.
func (nt *Network) SynsSlice(vals *[]float32, synvar SynapseVars) {
	ctx := &nt.Ctx
	if cap(*vals) >= int(nt.NSyns) {
		*vals = (*vals)[:nt.NSyns]
	} else {
		*vals = make([]float32, nt.NSyns)
	}
	i := 0
	for _, ly := range nt.Layers {
		for _, pj := range ly.SndPrjns {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIdx + syi
					(*vals)[i] = SynV(ctx, syni, synvar)
					i++
				}
			}
		}
	}
}

// NeuronsSlice returns a slice of neuron values
// using given neuron variable, resizing as needed.
func (nt *Network) NeuronsSlice(vals *[]float32, nrnVar string, di int) {
	if cap(*vals) >= int(nt.NNeurons) {
		*vals = (*vals)[:nt.NNeurons]
	} else {
		*vals = make([]float32, nt.NNeurons)
	}
	i := 0
	for _, ly := range nt.Layers {
		varIdx, _ := ly.UnitVarIdx(nrnVar)
		nn := int(ly.NNeurons)
		for lni := 0; lni < nn; lni++ {
			(*vals)[i] = ly.UnitVal1D(varIdx, lni, di)
			i++
		}
	}
}

// WtsHash returns a hash code of all weight values
func (nt *Network) WtsHash() string {
	var wts []float32
	nt.SynsSlice(&wts, Wt)
	return HashEncodeSlice(wts)
}

func HashEncodeSlice(slice []float32) string {
	byteSlice := make([]byte, len(slice)*4)
	for i, f := range slice {
		binary.LittleEndian.PutUint32(byteSlice[i*4:], math.Float32bits(f))
	}

	md5Hasher := md5.New()
	md5Hasher.Write(byteSlice)
	md5Sum := md5Hasher.Sum(nil)

	return hex.EncodeToString(md5Sum)
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (nt *NetworkBase) VarRange(varNm string) (min, max float32, err error) {
	first := true
	for _, ly := range nt.Layers {
		lmin, lmax, lerr := ly.VarRange(varNm)
		if lerr != nil {
			err = lerr
			return
		}
		if first {
			min = lmin
			max = lmax
			continue
		}
		if lmin < min {
			min = lmin
		}
		if lmax > max {
			max = lmax
		}
	}
	return
}

// SetRndSeed sets random seed and calls ResetRndSeed
func (nt *NetworkBase) SetRndSeed(seed int64) {
	nt.RndSeed = seed
	nt.ResetRndSeed()
}

// ResetRndSeed sets random seed to saved RndSeed, ensuring that the
// network-specific random seed generator has been created.
func (nt *NetworkBase) ResetRndSeed() {
	if nt.Rand.Rand == nil {
		nt.Rand.NewRand(nt.RndSeed)
	} else {
		nt.Rand.Seed(nt.RndSeed)
	}
}
