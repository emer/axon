// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/emer/emergent/emer"
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
	EmerNet     emer.Network          `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Nm          string                `desc:"overall name of network -- helps discriminate if there are multiple"`
	Layers      emer.Layers           `desc:"list of layers"`
	NThreads    int                   `desc:"number of threads to use for parallel threading -- typically only up to 4 is worthwile for large networks, with diminishing returns beyond 2, and 1 is best for small networks."`
	WtsFile     string                `desc:"filename of last weights file loaded or saved"`
	LayMap      map[string]emer.Layer `view:"-" desc:"map of name to layers -- layer names must be unique"`
	LayClassMap map[string][]string   `view:"-" desc:"map of layer classes -- made during Build"`
	MinPos      mat32.Vec3            `view:"-" desc:"minimum display position in network"`
	MaxPos      mat32.Vec3            `view:"-" desc:"maximum display position in network"`
	MetaData    map[string]string     `desc:"optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save"`

	// Implementation level code below:
	Neurons     []Neuron               `view:"-" desc:"entire network's allocation of neurons -- can be operated upon in parallel"`
	Prjns       []AxonPrjn             `view:"-" desc:"pointers to all projections in the network, via the AxonPrjn interface"`
	Threads     NetThreads             `desc:"threading config and implementation for CPU"`
	RecFunTimes bool                   `view:"-" desc:"record function timer information"`
	FunTimes    map[string]*timer.Time `view:"-" desc:"timers for each major function (step of processing)"`
	WaitGp      sync.WaitGroup         `view:"-" desc:"network-level wait group for synchronizing threaded layer calls"`
}

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name.
func (nt *NetworkBase) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
	nt.NThreads = 1
}

// emer.Network interface methods:
func (nt *NetworkBase) Name() string                  { return nt.Nm }
func (nt *NetworkBase) Label() string                 { return nt.Nm }
func (nt *NetworkBase) NLayers() int                  { return len(nt.Layers) }
func (nt *NetworkBase) Layer(idx int) emer.Layer      { return nt.Layers[idx] }
func (nt *NetworkBase) Bounds() (min, max mat32.Vec3) { min = nt.MinPos; max = nt.MaxPos; return }

// LayerByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *NetworkBase) LayerByName(name string) emer.Layer {
	if nt.LayMap == nil || len(nt.LayMap) != len(nt.Layers) {
		nt.MakeLayMap()
	}
	ly := nt.LayMap[name]
	return ly
}

// LayerByNameTry returns a layer by looking it up by name -- returns error message
// if layer is not found
func (nt *NetworkBase) LayerByNameTry(name string) (emer.Layer, error) {
	ly := nt.LayerByName(name)
	if ly == nil {
		err := fmt.Errorf("Layer named: %v not found in Network: %v\n", name, nt.Nm)
		// log.Println(err)
		return ly, err
	}
	return ly, nil
}

// MakeLayMap updates layer map based on current layers
func (nt *NetworkBase) MakeLayMap() {
	nt.LayMap = make(map[string]emer.Layer, len(nt.Layers))
	for _, ly := range nt.Layers {
		nt.LayMap[ly.Name()] = ly
	}
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
	return dedupe.DeDupe(nms)
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
		var lstly emer.Layer
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
					oly, err = nt.LayerByNameTry(rp.Other)
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
		rpjn := ly.RecvPrjns()
		for _, p := range *rpjn {
			if p.IsOff() {
				continue
			}
			pj := p.(AxonPrjn).AsAxon()
			str += fmt.Sprintf("\t%23s\t\tAbs:\t%g\tRel:\t%g\n", pj.Name(), pj.PrjnScale.Abs, pj.PrjnScale.Rel)
		}
	}
	return str
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *NetworkBase) AddLayerInit(ly emer.Layer, name string, shape []int, typ emer.LayerType) {
	if nt.EmerNet == nil {
		log.Printf("Network EmerNet is nil -- you MUST call InitName on network, passing a pointer to the network to initialize properly!")
		return
	}
	ly.InitName(ly, name, nt.EmerNet)
	ly.Config(shape, typ)
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
func (nt *NetworkBase) AddLayer(name string, shape []int, typ emer.LayerType) emer.Layer {
	ly := nt.EmerNet.NewLayer() // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *NetworkBase) AddLayer2D(name string, shapeY, shapeX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Axon networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *NetworkBase) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a projection between two layers, referenced by name
// adding to the recv and send projection lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) ConnectLayerNames(send, recv string, pat prjn.Pattern, typ emer.PrjnType) (rlay, slay emer.Layer, pj emer.Prjn, err error) {
	rlay, err = nt.LayerByNameTry(recv)
	if err != nil {
		return
	}
	slay, err = nt.LayerByNameTry(send)
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
func (nt *NetworkBase) ConnectLayers(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.ConnectLayersPrjn(send, recv, pat, typ, pj)
}

// ConnectLayersPrjn makes connection using given projection between two layers,
// adding given prjn to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) ConnectLayersPrjn(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(send, recv, pat, typ)
	recv.RecvPrjns().Add(pj)
	send.SendPrjns().Add(pj)
	return pj
}

// BidirConnectLayerNames establishes bidirectional projections between two layers,
// referenced by name, with low = the lower layer that sends a Forward projection
// to the high layer, and receives a Back projection in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) BidirConnectLayerNames(low, high string, pat prjn.Pattern) (lowlay, highlay emer.Layer, fwdpj, backpj emer.Prjn, err error) {
	lowlay, err = nt.LayerByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayerByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, emer.Forward)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, emer.Back)
	return
}

// BidirConnectLayers establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) BidirConnectLayers(low, high emer.Layer, pat prjn.Pattern) (fwdpj, backpj emer.Prjn) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, emer.Back)
	return
}

// BidirConnectLayersPy establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *NetworkBase) BidirConnectLayersPy(low, high emer.Layer, pat prjn.Pattern) {
	nt.ConnectLayers(low, high, pat, emer.Forward)
	nt.ConnectLayers(high, low, pat, emer.Back)
}

// LateralConnectLayer establishes a self-projection within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayer(lay emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.LateralConnectLayerPrjn(lay, pat, pj)
}

// LateralConnectLayerPrjn makes lateral self-projection using given projection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayerPrjn(lay emer.Layer, pat prjn.Pattern, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(lay, lay, pat, emer.Lateral)
	lay.RecvPrjns().Add(pj)
	lay.SendPrjns().Add(pj)
	return pj
}

// Build constructs the layer and projection state based on the layer shapes
// and patterns of interconnectivity
func (nt *NetworkBase) Build() error {
	// maxp := runtime.GOMAXPROCS(0)
	// if maxp > 4 {
	// 	maxp = 4
	// 	runtime.GOMAXPROCS(maxp)
	// }
	// mpi.Printf("Running on: %d GOMAXPROCS\n", maxp)

	nt.FunTimes = make(map[string]*timer.Time)

	nt.LayClassMap = make(map[string][]string)
	emsg := ""
	totNeurons := 0
	totPrjns := 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		totNeurons += ly.Shape().Len()
		totPrjns += len(*ly.SendPrjns())
		cls := strings.Split(ly.Class(), " ")
		for _, cl := range cls {
			ll := nt.LayClassMap[cl]
			ll = append(ll, ly.Name())
			nt.LayClassMap[cl] = ll
		}
	}
	nt.Neurons = make([]Neuron, totNeurons) // never grows
	nt.Prjns = make([]AxonPrjn, totPrjns)

	// todo: not currently useful:
	// nt.Threads.SetDefaults(totNeurons, totPrjns)
	// just use nthreads -- LVis testing shows similar scaling for all funs
	// except CycleNeuron which does keep scaling but in clusters you typically
	// can't actually steal more threads anyway, so just go with simple:
	nt.Threads.Set(2, nt.NThreads, nt.NThreads, nt.NThreads, nt.NThreads)
	nt.ThreadsAlloc()

	nidx := 0
	pidx := 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		nn := ly.Shape().Len()
		aly := ly.(AxonLayer).AsAxon()
		aly.Neurons = nt.Neurons[nidx : nidx+nn]
		aly.NeurStIdx = nidx
		err := ly.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
		sprjns := *ly.SendPrjns()
		for pi, spj := range sprjns {
			nt.Prjns[pidx+pi] = spj.(AxonPrjn)
		}
		nidx += nn
		pidx += len(sprjns)
	}
	nt.Layout()
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// DeleteAll deletes all layers, prepares network for re-configuring and building
func (nt *NetworkBase) DeleteAll() {
	nt.Layers = nil
	nt.LayMap = nil
	nt.FunTimes = nil
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
		ly, er := nt.LayerByNameTry(lw.Layer)
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
