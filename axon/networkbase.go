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
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cogentcore.org/core/base/indent"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/texteditor"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/emergent/v2/weights"
)

// NetworkBase manages the basic structural components of a network (layers).
// The main Network then can just have the algorithm-specific code.
type NetworkBase struct {

	// we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary.
	EmerNet emer.Network `copier:"-" json:"-" xml:"-" view:"-"`

	// overall name of network -- helps discriminate if there are multiple
	Nm string

	// filename of last weights file loaded or saved
	WtsFile string

	// Rubicon system for goal-driven motivated behavior, including Rubicon phasic dopamine signaling.  Manages internal drives, US outcomes. Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine are computed in equations using inputs from specialized network layers (LDTLayer driven by BLA, CeM layers, VSPatchLayer). Renders USLayer, PVLayer, DrivesLayer representations based on state updated here.
	Rubicon Rubicon

	// map of name to layers -- layer names must be unique
	LayMap map[string]*Layer `view:"-"`

	// map of layer classes -- made during Build
	LayClassMap map[string][]string `view:"-"`

	// minimum display position in network
	MinPos math32.Vector3 `view:"-"`

	// maximum display position in network
	MaxPos math32.Vector3 `view:"-"`

	// optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save
	MetaData map[string]string

	// if true, the neuron and synapse variables will be organized into a gpu-optimized memory order, otherwise cpu-optimized. This must be set before network Build() is called.
	UseGPUOrder bool `edit:"-"`

	// network index in global Networks list of networks -- needed for GPU shader kernel compatible network variable access functions (e.g., NrnV, SynV etc) in CPU mode
	NetIndex uint32 `view:"-"`

	// maximum synaptic delay across any pathway in the network -- used for sizing the GBuf accumulation buffer.
	MaxDelay uint32 `edit:"-" view:"-"`

	// maximum number of data inputs that can be processed in parallel in one pass of the network. Neuron storage is allocated to hold this amount during Build process, and this value reflects that.
	MaxData uint32 `edit:"-"`

	// total number of neurons
	NNeurons uint32 `edit:"-"`

	// total number of synapses
	NSyns uint32 `edit:"-"`

	// storage for global vars
	Globals []float32 `view:"-"`

	// array of layers
	Layers []*Layer

	// array of layer parameters, in 1-to-1 correspondence with Layers
	LayParams []LayerParams `view:"-"`

	// array of layer values, with extra per data
	LayValues []LayerValues `view:"-"`

	// array of inhibitory pools for all layers.
	Pools []Pool `view:"-"`

	// entire network's allocation of neuron variables, accessed via NrnV function with flexible striding
	Neurons []float32 `view:"-"`

	// ] entire network's allocation of neuron average avariables, accessed via NrnAvgV function with flexible striding
	NeuronAvgs []float32 `view:"-"`

	// entire network's allocation of neuron index variables, accessed via NrnI function with flexible striding
	NeuronIxs []uint32 `view:"-"`

	// pointers to all pathways in the network, sender-based
	Paths []*Path `view:"-"`

	// array of pathway parameters, in 1-to-1 correspondence with Paths, sender-based
	PathParams []PathParams `view:"-"`

	// entire network's allocation of synapse idx vars, organized sender-based, with flexible striding, accessed via SynI function
	SynapseIxs []uint32 `view:"-"`

	// entire network's allocation of synapses, organized sender-based, with flexible striding, accessed via SynV function
	Synapses []float32 `view:"-"`

	// entire network's allocation of synapse Ca vars, organized sender-based, with flexible striding, accessed via SynCaV function
	SynapseCas []float32 `view:"-"`

	// starting offset and N cons for each sending neuron, for indexing into the Syns synapses, which are organized sender-based.
	PathSendCon []StartN `view:"-"`

	// starting offset and N cons for each recv neuron, for indexing into the RecvSynIndex array of indexes into the Syns synapses, which are organized sender-based.
	PathRecvCon []StartN `view:"-"`

	// conductance buffer for accumulating spikes -- subslices are allocated to each pathway -- uses int-encoded float values for faster GPU atomic integration
	PathGBuf []int32 `view:"-"`

	// synaptic conductance integrated over time per pathway per recv neurons -- spikes come in via PathBuf -- subslices are allocated to each pathway
	PathGSyns []float32 `view:"-"`

	// indexes into Paths (organized by SendPath) organized by recv pathways -- needed for iterating through recv paths efficiently on GPU.
	RecvPathIndexes []uint32 `view:"-"`

	// indexes into Synapses for each recv neuron, organized into blocks according to PathRecvCon, for receiver-based access.
	RecvSynIndexes []uint32 `view:"-"`

	// external input values for all Input / Target / Compare layers in the network -- the ApplyExt methods write to this per layer, and it is then actually applied in one consistent method.
	Exts []float32

	// context used only for accessing neurons for display -- NetIndexes.NData in here is copied from active context in NewState
	Ctx Context `view:"-"`

	// random number generator for the network -- all random calls must use this -- set seed here for weight initialization values
	Rand randx.SysRand `view:"-"`

	// random seed to be set at the start of configuring the network and initializing the weights -- set this to get a different set of weights
	RandSeed int64 `edit:"-"`

	// number of threads to use for parallel processing
	NThreads int

	// GPU implementation
	GPU GPU `view:"inline"`

	// record function timer information
	RecFunTimes bool `view:"-"`

	// timers for each major function (step of processing)
	FunTimes map[string]*timer.Time `view:"-"`
}

// emer.Network interface methods:
func (nt *NetworkBase) Name() string                      { return nt.Nm }
func (nt *NetworkBase) Label() string                     { return nt.Nm }
func (nt *NetworkBase) NLayers() int                      { return len(nt.Layers) }
func (nt *NetworkBase) Layer(idx int) emer.Layer          { return nt.Layers[idx] }
func (nt *NetworkBase) Bounds() (min, max math32.Vector3) { min = nt.MinPos; max = nt.MaxPos; return }
func (nt *NetworkBase) MaxParallelData() int              { return int(nt.MaxData) }
func (nt *NetworkBase) NParallelData() int                { return int(nt.Ctx.NetIndexes.NData) }

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

// AxonPathByName returns a Path by looking it up by name in the list of pathways
// (nil if not found).
func (nt *NetworkBase) AxonPathByName(name string) *Path {
	for _, pj := range nt.Paths {
		if pj.Name() == name {
			return pj
		}
	}
	return nil
}

// PathByNameTry returns a Path by looking it up by name in the list of pathways
// (nil if not found).
func (nt *NetworkBase) PathByNameTry(name string) (emer.Path, error) {
	pj := nt.AxonPathByName(name)
	if pj != nil {
		return pj, nil
	}
	return nil, fmt.Errorf("Projection named: %q not found", name)
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
	// only get unique layers
	layers := []string{}
	has := map[string]bool{}
	for _, nm := range nms {
		if has[nm] {
			continue
		}
		layers = append(layers, nm)
		has[nm] = true
	}
	if len(layers) == 0 {
		panic(fmt.Sprintf("No Layers found for query: %#v. Basic layer types have been renamed since v1.7, use LayersByType for forward compatibility.", classes))
	}
	return layers
}

// LayerVal returns LayerValues for given layer and data parallel indexes
func (nt *NetworkBase) LayerValues(li, di uint32) *LayerValues {
	return &nt.LayValues[li*nt.MaxData+di]
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
// Not all pathways need to support all variables, but must safely return 0's for
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
	nt.BoundsUpdate()
}

// BoundsUpdate updates the Min / Max display bounds for 3D display
func (nt *NetworkBase) BoundsUpdate() {
	mn := math32.Vector3Scalar(math32.Infinity)
	mx := math32.Vector3{}
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

// ApplyParams applies given parameter style Sheet to layers and paths in this network.
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
	nt.Rubicon.Update()
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

// KeyPathParams returns a listing for all Recv pathways in the network,
// of the most important pathway-level params (specific to each algorithm).
func (nt *NetworkBase) KeyPathParams() string {
	return nt.AllPathScales()
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

// AllPathScales returns a listing of all PathScale parameters in the Network
// in all Layers, Recv pathways.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *NetworkBase) AllPathScales() string {
	str := ""
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		str += "\nLayer: " + ly.Name() + "\n"
		for i := 0; i < ly.NRecvPaths(); i++ {
			pj := ly.RcvPaths[i]
			if pj.IsOff() {
				continue
			}
			sn := pj.Send.Name()
			str += fmt.Sprintf("\t%15s\t%15s\tAbs:\t%6.2f\tRel:\t%6.2f\tGScale:\t%6.2f\tRel:%6.2f\n", sn, pj.PathType().String(), pj.Params.PathScale.Abs, pj.Params.PathScale.Rel, pj.Params.GScale.Scale, pj.Params.GScale.Rel)
			ph := pj.ParamsHistory.ParamsHistory()
			rh := ph["Path.PathScale.Rel"]
			ah := ph["Path.PathScale.Abs"]
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
	pars.SaveTOML(core.Filename(filepath.Join(dir, "params.toml")))
	nt.SaveAllParams(core.Filename(filepath.Join(dir, "params_all.txt")))
	nt.SaveNonDefaultParams(core.Filename(filepath.Join(dir, "params_nondef.txt")))
	nt.SaveAllLayerInhibs(core.Filename(filepath.Join(dir, "params_layers.txt")))
	nt.SaveAllPathScales(core.Filename(filepath.Join(dir, "params_paths.txt")))
	return nil
}

// SaveAllParams saves list of all parameters in Network to given file.
func (nt *NetworkBase) SaveAllParams(filename core.Filename) error {
	str := nt.AllParams()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SaveNonDefaultParams saves list of all non-default parameters in Network to given file.
func (nt *NetworkBase) SaveNonDefaultParams(filename core.Filename) error {
	str := nt.NonDefaultParams()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SaveAllLayerInhibs saves list of all layer Inhibition parameters to given file
func (nt *NetworkBase) SaveAllLayerInhibs(filename core.Filename) error {
	str := nt.AllLayerInhibs()
	err := os.WriteFile(string(filename), []byte(str), 0666)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SavePathScales saves a listing of all PathScale parameters in the Network
// in all Layers, Recv pathways.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *NetworkBase) SaveAllPathScales(filename core.Filename) error {
	str := nt.AllPathScales()
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
		for vv := GvRew; vv < GvCost; vv++ {
			str += fmt.Sprintf("%20s:\t%7.4f\n", vv.String(), GlbV(ctx, di, vv))
		}
		for vv := GvCost; vv <= GvCostRaw; vv++ {
			str += fmt.Sprintf("%20s:\t", vv.String())
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNCosts; ui++ {
				str += fmt.Sprintf("%d: %7.4f\t", ui, GlbCostV(ctx, di, vv, ui))
			}
			str += "\n"
		}
		for vv := GvUSneg; vv <= GvUSnegRaw; vv++ {
			str += fmt.Sprintf("%20s:\t", vv.String())
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNNegUSs; ui++ {
				str += fmt.Sprintf("%d: %7.4f\t", ui, GlbUSnegV(ctx, di, vv, ui))
			}
			str += "\n"
		}
		for vv := GvDrives; vv < GlobalVarsN; vv++ {
			str += fmt.Sprintf("%20s:\t", vv.String())
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNPosUSs; ui++ {
				str += fmt.Sprintf("%d:\t%7.4f\t", ui, GlbUSposV(ctx, di, vv, ui))
			}
			str += "\n"
		}
	}
	return str
}

// ShowAllGlobals shows a listing of all Global variables and values.
func (nt *NetworkBase) ShowAllGlobals() { //types:add
	agv := nt.AllGlobals()
	texteditor.TextDialog(nil, "All Global Vars: "+nt.Name(), agv)
}

// AllGlobalValues adds to map of all Global variables and values.
// ctrKey is a key of counters to contextualize values.
func (nt *NetworkBase) AllGlobalValues(ctrKey string, vals map[string]float32) {
	ctx := &nt.Ctx
	for di := uint32(0); di < nt.MaxData; di++ {
		for vv := GvRew; vv < GvCost; vv++ {
			key := fmt.Sprintf("%s  Di: %d\t%s", ctrKey, di, vv.String())
			vals[key] = GlbV(ctx, di, vv)
		}
		for vv := GvCost; vv <= GvCostRaw; vv++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNCosts; ui++ {
				key := fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, vv.String(), ui)
				vals[key] = GlbCostV(ctx, di, vv, ui)
			}
		}
		for vv := GvUSneg; vv <= GvUSnegRaw; vv++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNNegUSs; ui++ {
				key := fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, vv.String(), ui)
				vals[key] = GlbUSnegV(ctx, di, vv, ui)
			}
		}
		for vv := GvDrives; vv < GlobalVarsN; vv++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNPosUSs; ui++ {
				key := fmt.Sprintf("%s  Di: %d\t%s\t%d", ctrKey, di, vv.String(), ui)
				vals[key] = GlbUSposV(ctx, di, vv, ui)
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

// ConnectLayerNames establishes a pathway between two layers, referenced by name
// adding to the recv and send pathway lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) ConnectLayerNames(send, recv string, pat paths.Pattern, typ PathTypes) (rlay, slay *Layer, pj *Path, err error) {
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

// ConnectLayers establishes a pathway between two layers,
// adding to the recv and send pathway lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) ConnectLayers(send, recv *Layer, pat paths.Pattern, typ PathTypes) *Path {
	pj := &Path{}
	pj.Init(pj)
	pj.Connect(send, recv, pat, typ)
	recv.RcvPaths.Add(pj)
	send.SndPaths.Add(pj)
	return pj
}

// BidirConnectLayerNames establishes bidirectional pathways between two layers,
// referenced by name, with low = the lower layer that sends a Forward pathway
// to the high layer, and receives a Back pathway in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) BidirConnectLayerNames(low, high string, pat paths.Pattern) (lowlay, highlay *Layer, fwdpj, backpj *Path, err error) {
	lowlay, err = nt.LayByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, ForwardPath)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, BackPath)
	return
}

// BidirConnectLayers establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) BidirConnectLayers(low, high *Layer, pat paths.Pattern) (fwdpj, backpj *Path) {
	fwdpj = nt.ConnectLayers(low, high, pat, ForwardPath)
	backpj = nt.ConnectLayers(high, low, pat, BackPath)
	return
}

// BidirConnectLayersPy establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *NetworkBase) BidirConnectLayersPy(low, high *Layer, pat paths.Pattern) {
	nt.ConnectLayers(low, high, pat, ForwardPath)
	nt.ConnectLayers(high, low, pat, BackPath)
}

// LateralConnectLayer establishes a self-pathway within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayer(lay *Layer, pat paths.Pattern) *Path {
	pj := &Path{}
	return nt.LateralConnectLayerPath(lay, pat, pj)
}

// LateralConnectLayerPath makes lateral self-pathway using given pathway.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayerPath(lay *Layer, pat paths.Pattern, pj *Path) *Path {
	pj.Init(pj)
	pj.Connect(lay, lay, pat, LateralPath)
	lay.RcvPaths.Add(pj)
	lay.SndPaths.Add(pj)
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
	simCtx.NetIndexes.NData = uint32(maxData)
	simCtx.NetIndexes.MaxData = uint32(maxData)
}

// Build constructs the layer and pathway state based on the layer shapes
// and patterns of interconnectivity. Configures threading using heuristics based
// on final network size.  Must set UseGPUOrder properly prior to calling.
// Configures the given Context object used in the simulation with the memory
// access strides for this network -- must be set properly -- see SetCtxStrides.
func (nt *NetworkBase) Build(simCtx *Context) error { //types:add
	nt.UseGPUOrder = true // todo: set externally
	if nt.Rubicon.NPosUSs == 0 {
		nt.Rubicon.SetNUSs(simCtx, 1, 1)
	}
	nt.Rubicon.Update()
	ctx := &nt.Ctx
	*ctx = *simCtx
	ctx.NetIndexes.NetIndex = nt.NetIndex
	nt.FunTimes = make(map[string]*timer.Time)
	nt.LayClassMap = make(map[string][]string)
	maxData := int(nt.MaxData)
	emsg := ""
	totNeurons := 0
	totPaths := 0
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
		totPaths += ly.NSendPaths() // either way
		cls := strings.Split(ly.Class(), " ")
		for _, cl := range cls {
			ll := nt.LayClassMap[cl]
			ll = append(ll, ly.Name())
			nt.LayClassMap[cl] = ll
		}
	}
	nt.LayParams = make([]LayerParams, nLayers)
	nt.LayValues = make([]LayerValues, nLayers*maxData)
	nt.Pools = make([]Pool, totPools*maxData)
	nt.NNeurons = uint32(totNeurons)
	nneurv := uint32(totNeurons) * nt.MaxData * uint32(NeuronVarsN)
	nt.Neurons = make([]float32, nneurv)
	nneurav := uint32(totNeurons) * uint32(NeuronAvgVarsN)
	nt.NeuronAvgs = make([]float32, nneurav)
	nneuri := uint32(totNeurons) * uint32(NeuronIndexesN)
	nt.NeuronIxs = make([]uint32, nneuri)
	nt.Paths = make([]*Path, totPaths)
	nt.PathParams = make([]PathParams, totPaths)
	nt.Exts = make([]float32, totExts*maxData)

	if nt.UseGPUOrder {
		ctx.NeuronVars.SetVarOuter(totNeurons, maxData)
		ctx.NeuronAvgVars.SetVarOuter(totNeurons)
		ctx.NeuronIndexes.SetIndexOuter(totNeurons)
	} else {
		ctx.NeuronVars.SetNeuronOuter(maxData)
		ctx.NeuronAvgVars.SetNeuronOuter()
		ctx.NeuronIndexes.SetNeuronOuter()
	}

	totSynapses := 0
	totRecvCon := 0
	totSendCon := 0
	neurIndex := 0
	pathIndex := 0
	rpathIndex := 0
	poolIndex := 0
	extIndex := 0
	for li, ly := range nt.Layers {
		ly.Params = &nt.LayParams[li]
		ly.Params.LayType = LayerTypes(ly.Typ)
		ly.Values = nt.LayValues[li*maxData : (li+1)*maxData]
		if ly.IsOff() {
			continue
		}
		shp := ly.Shape()
		nn := shp.Len()
		ly.NNeurons = uint32(nn)
		ly.NeurStIndex = uint32(neurIndex)
		ly.MaxData = nt.MaxData
		np := ly.NSubPools() + 1
		npd := np * maxData
		ly.NPools = uint32(np)
		ly.Pools = nt.Pools[poolIndex : poolIndex+npd]
		ly.Params.Indexes.LayIndex = uint32(li)
		ly.Params.Indexes.MaxData = nt.MaxData
		ly.Params.Indexes.PoolSt = uint32(poolIndex)
		ly.Params.Indexes.NeurSt = uint32(neurIndex)
		ly.Params.Indexes.NeurN = uint32(nn)
		if shp.NumDims() == 2 {
			ly.Params.Indexes.ShpUnY = int32(shp.DimSize(0))
			ly.Params.Indexes.ShpUnX = int32(shp.DimSize(1))
			ly.Params.Indexes.ShpPlY = 1
			ly.Params.Indexes.ShpPlX = 1
		} else {
			ly.Params.Indexes.ShpPlY = int32(shp.DimSize(0))
			ly.Params.Indexes.ShpPlX = int32(shp.DimSize(1))
			ly.Params.Indexes.ShpUnY = int32(shp.DimSize(2))
			ly.Params.Indexes.ShpUnX = int32(shp.DimSize(3))
		}
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Values[di].LayIndex = uint32(li)
			ly.Values[di].DataIndex = uint32(di)
		}
		for pi := 0; pi < np; pi++ {
			for di := 0; di < maxData; di++ {
				ix := pi*int(ly.MaxData) + di
				pl := &ly.Pools[ix]
				pl.LayIndex = uint32(li)
				pl.DataIndex = uint32(di)
				pl.PoolIndex = uint32(poolIndex + ix)
			}
		}
		if ly.LayerType().IsExt() {
			ly.Exts = nt.Exts[extIndex : extIndex+nn*maxData]
			ly.Params.Indexes.ExtsSt = uint32(extIndex)
			extIndex += nn * maxData
		} else {
			ly.Exts = nil
			ly.Params.Indexes.ExtsSt = 0 // sticking with uint32 here -- otherwise could be -1
		}
		spaths := *ly.SendPaths()
		ly.Params.Indexes.SendSt = uint32(pathIndex)
		ly.Params.Indexes.SendN = uint32(len(spaths))
		for pi, pj := range spaths {
			pii := pathIndex + pi
			pj.Params = &nt.PathParams[pii]
			nt.Paths[pii] = pj
		}
		err := ly.Build() // also builds paths and sets SubPool indexes
		if err != nil {
			emsg += err.Error() + "\n"
		}
		// now collect total number of synapses after layer build
		for _, pj := range spaths {
			totSynapses += len(pj.SendConIndex)
			totSendCon += nn // sep vals for each send neuron per path
		}
		rpaths := *ly.RecvPaths()
		ly.Params.Indexes.RecvSt = uint32(rpathIndex)
		ly.Params.Indexes.RecvN = uint32(len(rpaths))
		totRecvCon += nn * len(rpaths)
		rpathIndex += len(rpaths)
		neurIndex += nn
		pathIndex += len(spaths)
		poolIndex += npd
	}
	if totSynapses > math.MaxUint32 {
		log.Fatalf("ERROR: total number of synapses is greater than uint32 capacity\n")
	}

	nt.NSyns = uint32(totSynapses)
	nSynFloat := totSynapses * int(SynapseVarsN)
	nt.Synapses = make([]float32, nSynFloat)
	nSynCaFloat := totSynapses * int(SynapseCaVarsN) * int(nt.MaxData)
	nt.SynapseCas = make([]float32, nSynCaFloat)
	nt.SynapseIxs = make([]uint32, totSynapses*int(SynapseIndexesN))
	nt.PathSendCon = make([]StartN, totSendCon)
	nt.PathRecvCon = make([]StartN, totRecvCon)
	nt.RecvPathIndexes = make([]uint32, rpathIndex)
	nt.RecvSynIndexes = make([]uint32, totSynapses)

	if nt.UseGPUOrder {
		ctx.SynapseVars.SetVarOuter(totSynapses)
		ctx.SynapseCaVars.SetVarOuter(totSynapses, maxData)
		ctx.SynapseIndexes.SetIndexOuter(totSynapses)
	} else {
		ctx.SynapseVars.SetSynapseOuter()
		ctx.SynapseCaVars.SetSynapseOuter(maxData)
		ctx.SynapseIndexes.SetSynapseOuter()
	}

	// distribute synapses, send
	syIndex := 0
	pjidx := 0
	sendConIndex := 0
	for _, ly := range nt.Layers {
		for _, pj := range ly.SndPaths {
			rlay := pj.Recv
			pj.Params.Indexes.RecvLay = uint32(rlay.Idx)
			pj.Params.Indexes.RecvNeurSt = uint32(rlay.NeurStIndex)
			pj.Params.Indexes.RecvNeurN = rlay.NNeurons
			pj.Params.Indexes.SendLay = uint32(ly.Idx)
			pj.Params.Indexes.SendNeurSt = uint32(ly.NeurStIndex)
			pj.Params.Indexes.SendNeurN = ly.NNeurons

			nsyn := len(pj.SendConIndex)
			pj.Params.Indexes.SendConSt = uint32(sendConIndex)
			pj.Params.Indexes.SynapseSt = uint32(syIndex)
			pj.SynStIndex = uint32(syIndex)
			pj.Params.Indexes.PathIndex = uint32(pjidx)
			pj.NSyns = uint32(nsyn)
			for sni := uint32(0); sni < ly.NNeurons; sni++ {
				si := ly.NeurStIndex + sni
				scon := pj.SendCon[sni]
				nt.PathSendCon[sendConIndex] = scon
				sendConIndex++
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					SetSynI(ctx, syni, SynSendIndex, uint32(si)) // network-global idx
					SetSynI(ctx, syni, SynRecvIndex, pj.SendConIndex[syi]+uint32(rlay.NeurStIndex))
					SetSynI(ctx, syni, SynPathIndex, uint32(pjidx))
					syIndex++
				}
			}
			pjidx++
		}
	}

	// update recv synapse / path info
	rpathIndex = 0
	recvConIndex := 0
	syIndex = 0
	for _, ly := range nt.Layers {
		for _, pj := range ly.RcvPaths {
			nt.RecvPathIndexes[rpathIndex] = pj.Params.Indexes.PathIndex
			pj.Params.Indexes.RecvConSt = uint32(recvConIndex)
			pj.Params.Indexes.RecvSynSt = uint32(syIndex)
			synSt := pj.Params.Indexes.SynapseSt
			for rni := uint32(0); rni < ly.NNeurons; rni++ {
				if len(pj.RecvCon) <= int(rni) {
					continue
				}
				rcon := pj.RecvCon[rni]
				nt.PathRecvCon[recvConIndex] = rcon
				recvConIndex++
				syIndexes := pj.RecvSynIndexes(rni)
				for _, ssi := range syIndexes {
					nt.RecvSynIndexes[syIndex] = ssi + synSt
					syIndex++
				}
			}
			rpathIndex++
		}
	}

	ctx.NetIndexes.MaxData = nt.MaxData
	ctx.NetIndexes.NLayers = uint32(nLayers)
	ctx.NetIndexes.NNeurons = nt.NNeurons
	ctx.NetIndexes.NPools = uint32(totPools)
	ctx.NetIndexes.NSyns = nt.NSyns
	ctx.NetIndexes.RubiconNPosUSs = nt.Rubicon.NPosUSs
	ctx.NetIndexes.RubiconNNegUSs = nt.Rubicon.NNegUSs
	ctx.SetGlobalStrides()

	nt.SetCtxStrides(simCtx)
	nt.BuildGlobals(simCtx)

	nt.Layout()
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// BuildPathGBuf builds the PathGBuf, PathGSyns,
// based on the MaxDelay values in thePathParams,
// which should have been configured by this point.
// Called by default in InitWts()
func (nt *NetworkBase) BuildPathGBuf() {
	nt.MaxDelay = 0
	npjneur := uint32(0)
	for _, ly := range nt.Layers {
		nneur := uint32(ly.NNeurons)
		for _, pj := range ly.RcvPaths {
			if pj.Params.Com.MaxDelay > nt.MaxDelay {
				nt.MaxDelay = pj.Params.Com.MaxDelay
			}
			npjneur += nneur
		}
	}
	mxlen := nt.MaxDelay + 1
	gbsz := npjneur * mxlen * nt.MaxData
	gsynsz := npjneur * nt.MaxData
	if uint32(cap(nt.PathGBuf)) >= gbsz {
		nt.PathGBuf = nt.PathGBuf[:gbsz]
	} else {
		nt.PathGBuf = make([]int32, gbsz)
	}
	if uint32(cap(nt.PathGSyns)) >= gsynsz {
		nt.PathGSyns = nt.PathGSyns[:gsynsz]
	} else {
		nt.PathGSyns = make([]float32, gsynsz)
	}

	gbi := uint32(0)
	gsi := uint32(0)
	for _, ly := range nt.Layers {
		nneur := uint32(ly.NNeurons)
		for _, pj := range ly.RcvPaths {
			gbs := nneur * mxlen * nt.MaxData
			pj.Params.Indexes.GBufSt = gbi
			pj.GBuf = nt.PathGBuf[gbi : gbi+gbs]
			gbi += gbs
			pj.Params.Indexes.GSynSt = gsi
			pj.GSyns = nt.PathGSyns[gsi : gsi+nneur*nt.MaxData]
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
	nt.LayValues = nil
	nt.Pools = nil
	nt.Neurons = nil
	nt.NeuronAvgs = nil
	nt.NeuronIxs = nil
	nt.Paths = nil
	nt.PathParams = nil
	nt.Synapses = nil
	nt.SynapseCas = nil
	nt.SynapseIxs = nil
	nt.PathSendCon = nil
	nt.PathRecvCon = nil
	nt.PathGBuf = nil
	nt.PathGSyns = nil
	nt.RecvPathIndexes = nil
	nt.Exts = nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Weights File

// SaveWtsJSON saves network weights (and any other state that adapts with learning)
// to a JSON-formatted file.  If filename has .gz extension, then file is gzip compressed.
func (nt *NetworkBase) SaveWtsJSON(filename core.Filename) error { //types:add
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
func (nt *NetworkBase) OpenWtsJSON(filename core.Filename) error { //types:add
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
	nt.GPU.SyncAllFromGPU()
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
func (nt *NetworkBase) OpenWtsCpp(filename core.Filename) error {
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
		for _, pj := range ly.SndPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
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
		varIndex, _ := ly.UnitVarIndex(nrnVar)
		nn := int(ly.NNeurons)
		for lni := 0; lni < nn; lni++ {
			(*vals)[i] = ly.UnitVal1D(varIndex, lni, di)
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
// todo: support r. s. pathway values
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

// SetRandSeed sets random seed and calls ResetRandSeed
func (nt *NetworkBase) SetRandSeed(seed int64) {
	nt.RandSeed = seed
	nt.ResetRandSeed()
}

// ResetRandSeed sets random seed to saved RandSeed, ensuring that the
// network-specific random seed generator has been created.
func (nt *NetworkBase) ResetRandSeed() {
	if nt.Rand.Rand == nil {
		nt.Rand.NewRand(nt.RandSeed)
	} else {
		nt.Rand.Seed(nt.RandSeed)
	}
}

// CheckSameSize checks if this network is the same size as given other,
// in terms of NNeurons, MaxData, and NSyns.  Returns error message if not.
func (nt *NetworkBase) CheckSameSize(on *NetworkBase) error {
	if nt.NNeurons != on.NNeurons {
		err := fmt.Errorf("CheckSameSize: dest NNeurons: %d != src NNeurons: %d", nt.NNeurons, on.NNeurons)
		return err
	}
	if nt.MaxData != on.MaxData {
		err := fmt.Errorf("CheckSameSize: dest MaxData: %d != src MaxData: %d", nt.MaxData, on.MaxData)
		return err
	}
	if nt.NSyns != on.NSyns {
		err := fmt.Errorf("CheckSameSize: dest NSyns: %d != src NSyns: %d", nt.NSyns, on.NSyns)
		return err
	}
	return nil
}

// CopyStateFrom copies entire network state from other network.
// Other network must have identical configuration, as this just
// does a literal copy of the state values.  This is checked
// and errors are returned (and logged).
// See also DiffFrom.
func (nt *NetworkBase) CopyStateFrom(on *NetworkBase) error {
	if err := nt.CheckSameSize(on); err != nil {
		slog.Error(err.Error())
		return err
	}
	copy(nt.Neurons, on.Neurons)
	copy(nt.NeuronAvgs, on.NeuronAvgs)
	copy(nt.Pools, on.Pools)
	copy(nt.LayValues, on.LayValues)
	copy(nt.Synapses, on.Synapses)
	copy(nt.SynapseCas, on.SynapseCas)
	return nil
}

// DiffFrom returns a string reporting differences between this network
// and given other, up to given max number of differences (0 = all),
// for each state value.
func (nt *NetworkBase) DiffFrom(ctx *Context, on *NetworkBase, maxDiff int) string {
	diffs := ""
	ndif := 0
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		for ni := uint32(0); ni < nt.NNeurons; ni++ {
			for nvar := Spike; nvar < NeuronVarsN; nvar++ {
				nv := nt.Neurons[ctx.NeuronVars.Index(ni, di, nvar)]
				ov := on.Neurons[ctx.NeuronVars.Index(ni, di, nvar)]
				if nv != ov {
					diffs += fmt.Sprintf("Neuron: di: %d\tni: %d\tvar: %s\tval: %g\toth: %g\n", di, ni, nvar.String(), nv, ov)
					ndif++
					if maxDiff > 0 && ndif >= maxDiff {
						return diffs
					}
				}
			}
		}
	}
	for ni := uint32(0); ni < nt.NNeurons; ni++ {
		for nvar := ActAvg; nvar < NeuronAvgVarsN; nvar++ {
			nv := nt.NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)]
			ov := on.NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)]
			if nv != ov {
				diffs += fmt.Sprintf("NeuronAvg: ni: %d\tvar: %s\tval: %g\toth: %g\n", ni, nvar.String(), nv, ov)
				ndif++
				if maxDiff > 0 && ndif >= maxDiff {
					return diffs
				}
			}
		}
	}
	for si := uint32(0); si < nt.NSyns; si++ {
		for svar := Wt; svar < SynapseVarsN; svar++ {
			sv := nt.Synapses[ctx.SynapseVars.Index(si, svar)]
			ov := on.Synapses[ctx.SynapseVars.Index(si, svar)]
			if sv != ov {
				diffs += fmt.Sprintf("Synapse: si: %d\tvar: %s\tval: %g\toth: %g\n", si, svar.String(), sv, ov)
				ndif++
				if maxDiff > 0 && ndif >= maxDiff {
					return diffs
				}
			}
		}
	}
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		for si := uint32(0); si < nt.NSyns; si++ {
			for svar := Tr; svar < SynapseCaVarsN; svar++ {
				sv := nt.Synapses[ctx.SynapseCaVars.Index(di, si, svar)]
				ov := on.Synapses[ctx.SynapseCaVars.Index(di, si, svar)]
				if sv != ov {
					diffs += fmt.Sprintf("SynapseCa: di: %d, si: %d\tvar: %s\tval: %g\toth: %g\n", di, si, svar.String(), sv, ov)
					ndif++
					if maxDiff > 0 && ndif >= maxDiff {
						return diffs
					}
				}
			}
		}
	}
	return diffs
}
