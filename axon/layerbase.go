// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"fmt"
	"log"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
)

// LayerBase manages the structural elements of the layer, which are common
// to any Layer type. The main Layer then can just have the algorithm-specific code.
type LayerBase struct {
	AxonLay     AxonLayer         `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an AxonLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Network     emer.Network      `copy:"-" json:"-" xml:"-" view:"-" desc:"our parent network, in case we need to use it to find other layers etc -- set when added by network"`
	Nm          string            `desc:"Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name"`
	Cls         string            `desc:"Class is for applying parameter styles, can be space separated multple tags"`
	Off         bool              `desc:"inactivate this layer -- allows for easy experimentation"`
	Shp         etensor.Shape     `desc:"shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools"`
	Typ         emer.LayerType    `desc:"type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)"`
	Rel         relpos.Rel        `view:"inline" desc:"Spatial relationship to other layer, determines positioning"`
	Ps          mat32.Vec3        `desc:"position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis."`
	Idx         int               `view:"-" inactive:"-" desc:"a 0..n-1 index of the position of the layer within list of layers in the network. For Axon networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers."`
	NeurStIdx   int               `view:"-" inactive:"-" desc:"starting index of neurons for this layer within the global Network list"`
	RepIxs      []int             `view:"-" desc:"indexes of representative units in the layer, for computationally expensive stats or displays -- also set RepShp"`
	RepShp      etensor.Shape     `view:"-" desc:"shape of representative units in the layer -- if RepIxs is empty or .Shp is nil, use overall layer shape"`
	RcvPrjns    AxonPrjns         `desc:"list of receiving projections into this layer from other layers"`
	SndPrjns    AxonPrjns         `desc:"list of sending projections from this layer to other layers"`
	Neurons     []Neuron          `desc:"slice of neurons for this layer -- flat list of len = Shp.Len(). You must iterate over index and use pointer to modify values."`
	Pools       []Pool            `desc:"inhibition and other pooled, aggregate state variables -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
	BuildConfig map[string]string `desc:"configuration data set when the network is configured, that is used during the network Build() process via PostBuild method, after all the structure of the network has been fully constructed.  In particular, the Params is nil until Build, so setting anything specific in there (e.g., an index to another layer) must be done as a second pass.  Note that Params are all applied after Build and can set user-modifiable params, so this is for more special algorithm structural parameters set during ConfigNet() methods.,"`
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ly *LayerBase) InitName(lay emer.Layer, name string, net emer.Network) {
	ly.AxonLay = lay.(AxonLayer)
	ly.Nm = name
	ly.Network = net
	ly.BuildConfig = make(map[string]string)
}

// todo: remove from emer.Layer api
func (ly *LayerBase) Thread() int       { return 0 }
func (ly *LayerBase) SetThread(thr int) {}

func (ly *LayerBase) Name() string               { return ly.Nm }
func (ly *LayerBase) SetName(nm string)          { ly.Nm = nm }
func (ly *LayerBase) Label() string              { return ly.Nm }
func (ly *LayerBase) SetClass(cls string)        { ly.Cls = cls }
func (ly *LayerBase) LayerType() LayerTypes      { return LayerTypes(ly.Typ) }
func (ly *LayerBase) Class() string              { return ly.LayerType().String() + " " + ly.Cls }
func (ly *LayerBase) TypeName() string           { return "Layer" } // type category, for params..
func (ly *LayerBase) Type() emer.LayerType       { return ly.Typ }
func (ly *LayerBase) SetType(typ emer.LayerType) { ly.Typ = typ }
func (ly *LayerBase) IsOff() bool                { return ly.Off }
func (ly *LayerBase) SetOff(off bool) {
	ly.Off = off
	// a Prjn is off if either the sending or the receiving layer is off
	// or if the prjn has been set to Off directly
	for _, pj := range ly.RcvPrjns {
		pj.SetOff(pj.SendLay().IsOff() || off)
	}
	for _, pj := range ly.SndPrjns {
		pj.SetOff(pj.RecvLay().IsOff() || off)
	}
}
func (ly *LayerBase) Shape() *etensor.Shape      { return &ly.Shp }
func (ly *LayerBase) Is2D() bool                 { return ly.Shp.NumDims() == 2 }
func (ly *LayerBase) Is4D() bool                 { return ly.Shp.NumDims() == 4 }
func (ly *LayerBase) RelPos() relpos.Rel         { return ly.Rel }
func (ly *LayerBase) Pos() mat32.Vec3            { return ly.Ps }
func (ly *LayerBase) SetPos(pos mat32.Vec3)      { ly.Ps = pos }
func (ly *LayerBase) Index() int                 { return ly.Idx }
func (ly *LayerBase) SetIndex(idx int)           { ly.Idx = idx }
func (ly *LayerBase) RecvPrjns() *AxonPrjns      { return &ly.RcvPrjns }
func (ly *LayerBase) NRecvPrjns() int            { return len(ly.RcvPrjns) }
func (ly *LayerBase) RecvPrjn(idx int) emer.Prjn { return ly.RcvPrjns[idx] }
func (ly *LayerBase) SendPrjns() *AxonPrjns      { return &ly.SndPrjns }
func (ly *LayerBase) NSendPrjns() int            { return len(ly.SndPrjns) }
func (ly *LayerBase) SendPrjn(idx int) emer.Prjn { return ly.SndPrjns[idx] }
func (ly *LayerBase) RepIdxs() []int             { return ly.RepIxs }
func (ly *LayerBase) NeurStartIdx() int          { return ly.NeurStIdx }

func (ly *LayerBase) SendNameTry(sender string) (emer.Prjn, error) {
	return emer.SendNameTry(ly.AxonLay, sender)
}
func (ly *LayerBase) SendNameTypeTry(sender, typ string) (emer.Prjn, error) {
	return emer.SendNameTypeTry(ly.AxonLay, sender, typ)
}
func (ly *LayerBase) RecvNameTry(receiver string) (emer.Prjn, error) {
	return emer.RecvNameTry(ly.AxonLay, receiver)
}
func (ly *LayerBase) RecvNameTypeTry(receiver, typ string) (emer.Prjn, error) {
	return emer.RecvNameTypeTry(ly.AxonLay, receiver, typ)
}

func (ly *LayerBase) Idx4DFrom2D(x, y int) ([]int, bool) {
	lshp := ly.Shape()
	nux := lshp.Dim(3)
	nuy := lshp.Dim(2)
	ux := x % nux
	uy := y % nuy
	px := x / nux
	py := y / nuy
	idx := []int{py, px, uy, ux}
	if !lshp.IdxIsValid(idx) {
		return nil, false
	}
	return idx, true
}

func (ly *LayerBase) SetRelPos(rel relpos.Rel) {
	ly.Rel = rel
	if ly.Rel.Scale == 0 {
		ly.Rel.Defaults()
	}
}

func (ly *LayerBase) Size() mat32.Vec2 {
	if ly.Rel.Scale == 0 {
		ly.Rel.Defaults()
	}
	var sz mat32.Vec2
	switch {
	case ly.Is2D():
		sz = mat32.Vec2{float32(ly.Shp.Dim(1)), float32(ly.Shp.Dim(0))} // Y, X
	case ly.Is4D():
		// note: pool spacing is handled internally in display and does not affect overall size
		sz = mat32.Vec2{float32(ly.Shp.Dim(1) * ly.Shp.Dim(3)), float32(ly.Shp.Dim(0) * ly.Shp.Dim(2))} // Y, X
	default:
		sz = mat32.Vec2{float32(ly.Shp.Len()), 1}
	}
	return sz.MulScalar(ly.Rel.Scale)
}

// SetShape sets the layer shape and also uses default dim names
func (ly *LayerBase) SetShape(shape []int) {
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ly.Shp.SetShape(shape, nil, dnms) // row major default
}

// SetRepIdxsShape sets the RepIdxs, and RepShape and as list of dimension sizes
func (ly *LayerBase) SetRepIdxsShape(idxs, shape []int) {
	ly.RepIxs = idxs
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ly.RepShp.SetShape(shape, nil, dnms) // row major default
}

// RepShape returns the shape to use for representative units
func (ly *LayerBase) RepShape() *etensor.Shape {
	sz := len(ly.RepIxs)
	if sz == 0 {
		return &ly.Shp
	}
	if ly.RepShp.Len() < sz {
		ly.RepShp.SetShape([]int{sz}, nil, nil) // row major default
	}
	return &ly.RepShp
}

// NPools returns the number of unit sub-pools according to the shape parameters.
// Currently supported for a 4D shape, where the unit pools are the first 2 Y,X dims
// and then the units within the pools are the 2nd 2 Y,X dims
func (ly *LayerBase) NPools() int {
	if ly.Shp.NumDims() != 4 {
		return 0
	}
	return ly.Shp.Dim(0) * ly.Shp.Dim(1)
}

// Pool returns pool at given index
func (ly *LayerBase) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

// PoolTry returns pool at given index, returns error if index is out of range
func (ly *LayerBase) PoolTry(idx int) (*Pool, error) {
	np := len(ly.Pools)
	if idx < 0 || idx >= np {
		return nil, fmt.Errorf("Layer Pool index: %v out of range, N = %v", idx, np)
	}
	return &(ly.Pools[idx]), nil
}

// RecipToSendPrjn finds the reciprocal projection relative to the given sending projection
// found within the SendPrjns of this layer.  This is then a recv prjn within this layer:
//
//	S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
//
// returns false if not found.
func (ly *LayerBase) RecipToSendPrjn(spj emer.Prjn) (emer.Prjn, bool) {
	for _, rpj := range ly.RcvPrjns {
		if rpj.SendLay() == spj.RecvLay() {
			return rpj, true
		}
	}
	return nil, false
}

// Config configures the basic properties of the layer
func (ly *LayerBase) Config(shape []int, typ emer.LayerType) {
	ly.SetShape(shape)
	ly.Typ = typ
}

// ApplyParams applies given parameter style Sheet to this layer and its recv projections.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ly *LayerBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	app, err := pars.Apply(ly.AxonLay, setMsg) // essential to go through AxonLay
	if app {
		ly.AxonLay.UpdateParams()
		applied = true
	}
	if err != nil {
		rerr = err
	}
	for _, pj := range ly.RcvPrjns {
		app, err = pj.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ly *LayerBase) NonDefaultParams() string {
	nds := giv.StructNonDefFieldsStr(ly.AxonLay, ly.Nm)
	for _, pj := range ly.RcvPrjns {
		pnd := pj.NonDefaultParams()
		nds += pnd
	}
	return nds
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// SetBuildConfig sets named configuration parameter to given string value
// to be used in the PostBuild stage -- mainly for layer names that need to be
// looked up and turned into indexes, after entire network is built.
func (ly *LayerBase) SetBuildConfig(param, val string) {
	ly.BuildConfig[param] = val
}

// BuildConfigByName looks for given BuildConfig option by name,
// and reports & returns an error if not found.
func (ly *LayerBase) BuildConfigByName(nm string) (string, error) {
	cfg, ok := ly.BuildConfig[nm]
	if !ok {
		err := fmt.Errorf("Layer: %s does not have BuildConfig: %s set -- error in ConfigNet", ly.Name(), nm)
		log.Println(err)
		return cfg, err
	}
	return cfg, nil
}

// BuildConfigFindLayer looks for BuildConfig of given name
// and if found, looks for layer with corresponding name.
// if mustName is true, then an error is logged if the BuildConfig
// name does not exist.  An error is always logged if the layer name
// is not found.  -1 is returned in any case of not found.
func (ly *LayerBase) BuildConfigFindLayer(nm string, mustName bool) int32 {
	idx := int32(-1)
	if rnm, ok := ly.BuildConfig[nm]; ok {
		dly, err := ly.Network.LayerByNameTry(rnm)
		if err != nil {
			log.Println(err)
		} else {
			idx = int32(dly.Index())
		}
	} else {
		if mustName {
			err := fmt.Errorf("Layer: %s does not have BuildConfig: %s set -- error in ConfigNet", ly.Name(), nm)
			log.Println(err)
		}
	}
	return idx
}

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *LayerBase) BuildSubPools() {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shp.Shapes()
	spy := sh[0]
	spx := sh[1]
	pi := 1
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			soff := uint32(ly.Shp.Offset([]int{py, px, 0, 0}))
			eoff := uint32(ly.Shp.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1)
			pl := &ly.Pools[pi]
			pl.StIdx = soff
			pl.EdIdx = eoff
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				nrn.SubPool = uint32(pi)
			}
			pi++
		}
	}
}

// BuildPools builds the inhibitory pools structures -- nu = number of units in layer
func (ly *LayerBase) BuildPools(nu int) error {
	np := 1 + ly.NPools()
	ly.Pools = make([]Pool, np)
	lpl := &ly.Pools[0]
	lpl.StIdx = 0
	lpl.EdIdx = uint32(nu)
	if np > 1 {
		ly.BuildSubPools()
	}
	return nil
}

// BuildPrjns builds the projections, recv-side
func (ly *LayerBase) BuildPrjns() error {
	emsg := ""
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		err := pj.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// Build constructs the layer state, including calling Build on the projections
func (ly *LayerBase) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	// note: ly.Neurons are allocated by Network from global network pool
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.NeurIdx = uint32(ni)
		nrn.LayIdx = uint32(ly.Idx)
	}
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	ly.AxonLay.PostBuild()
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (ly *LayerBase) VarRange(varNm string) (min, max float32, err error) {
	sz := len(ly.Neurons)
	if sz == 0 {
		return
	}
	vidx := 0
	vidx, err = NeuronVarIdxByName(varNm)
	if err != nil {
		return
	}

	v0 := ly.Neurons[0].VarByIndex(vidx)
	min = v0
	max = v0
	for i := 1; i < sz; i++ {
		vl := ly.Neurons[i].VarByIndex(vidx)
		if vl < min {
			min = vl
		}
		if vl > max {
			max = vl
		}
	}
	return
}
