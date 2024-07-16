// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"strconv"

	"cogentcore.org/core/base/indent"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/emergent/v2/weights"
)

// LayerBase manages the structural elements of the layer, which are common
// to any Layer type.
// The Base does not have algorithm-specific methods and parameters, so it can be easily
// reused for different algorithms, and cleanly separates the algorithm-specific code.
// Any dependency on the algorithm-level Layer can be captured in the AxonLayer interface,
// accessed via the AxonLay field.
type LayerBase struct {

	// we need a pointer to ourselves as an AxonLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary.
	AxonLay AxonLayer `copier:"-" json:"-" xml:"-" display:"-"`

	// our parent network, in case we need to use it to find other layers etc -- set when added by network
	Network *Network `copier:"-" json:"-" xml:"-" display:"-"`

	// Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name
	Nm string

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string

	// inactivate this layer -- allows for easy experimentation
	Off bool

	// shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools
	Shp tensor.Shape

	// type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)
	Typ LayerTypes

	// Spatial relationship to other layer, determines positioning
	Rel relpos.Rel `table:"-" display:"inline"`

	// position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis.
	Ps math32.Vector3 `table:"-"`

	// a 0..n-1 index of the position of the layer within list of layers in the network. For Axon networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers.
	Idx int `display:"-" inactive:"-"`

	// number of neurons in the layer
	NNeurons uint32 `display:"-"`

	// starting index of neurons for this layer within the global Network list
	NeurStIndex uint32 `display:"-" inactive:"-"`

	// number of pools based on layer shape -- at least 1 for layer pool + 4D subpools
	NPools uint32 `display:"-"`

	// maximum amount of input data that can be processed in parallel in one pass of the network. Neuron, Pool, Values storage is allocated to hold this amount.
	MaxData uint32 `display:"-"`

	// indexes of representative units in the layer, for computationally expensive stats or displays -- also set RepShp
	RepIxs []int `display:"-"`

	// shape of representative units in the layer -- if RepIxs is empty or .Shp is nil, use overall layer shape
	RepShp tensor.Shape `display:"-"`

	// list of receiving pathways into this layer from other layers
	RcvPaths AxonPaths

	// list of sending pathways from this layer to other layers
	SndPaths AxonPaths

	// layer-level state values that are updated during computation -- one for each data parallel -- is a sub-slice of network full set
	Values []LayerValues

	// computes FS-FFFB inhibition and other pooled, aggregate state variables -- has at least 1 for entire layer (lpl = layer pool), and one for each sub-pool if shape supports that (4D) * 1 per data parallel (inner loop).  This is a sub-slice from overall Network Pools slice.  You must iterate over index and use pointer to modify values.
	Pools []Pool

	// external input values for this layer, allocated from network global Exts slice
	Exts []float32 `display:"-"`

	// configuration data set when the network is configured, that is used during the network Build() process via PostBuild method, after all the structure of the network has been fully constructed.  In particular, the Params is nil until Build, so setting anything specific in there (e.g., an index to another layer) must be done as a second pass.  Note that Params are all applied after Build and can set user-modifiable params, so this is for more special algorithm structural parameters set during ConfigNet() methods.,
	BuildConfig map[string]string `table:"-"`

	// default parameters that are applied prior to user-set parameters -- these are useful for specific layer functionality in specialized brain areas (e.g., Rubicon, BG etc) not associated with a layer type, which otherwise is used to hard-code initial default parameters -- typically just set to a literal map.
	DefParams params.Params `table:"-"`

	// provides a history of parameters applied to the layer
	ParamsHistory params.HistoryImpl `table:"-"`
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ly *LayerBase) InitName(lay emer.Layer, name string, net emer.Network) {
	ly.AxonLay = lay.(AxonLayer)
	ly.Nm = name
	ly.Network = net.(AxonNetwork).AsAxon()
	ly.BuildConfig = make(map[string]string)
}

// todo: remove from emer.Layer api
func (ly *LayerBase) Thread() int       { return 0 }
func (ly *LayerBase) SetThread(thr int) {}

func (ly *LayerBase) Name() string      { return ly.Nm }
func (ly *LayerBase) SetName(nm string) { ly.Nm = nm }
func (ly *LayerBase) Label() string     { return ly.Nm }
func (ly *Layer) AddClass(cls ...string) emer.Layer {
	ly.Cls = params.AddClass(ly.Cls, cls...)
	return ly
}
func (ly *LayerBase) LayerType() LayerTypes      { return LayerTypes(ly.Typ) }
func (ly *LayerBase) Class() string              { return ly.LayerType().String() + " " + ly.Cls }
func (ly *LayerBase) TypeName() string           { return "Layer" } // type category, for params..
func (ly *LayerBase) Type() emer.LayerType       { return emer.LayerType(ly.Typ) }
func (ly *LayerBase) SetType(typ emer.LayerType) { ly.Typ = LayerTypes(typ) }
func (ly *LayerBase) IsOff() bool                { return ly.Off }
func (ly *LayerBase) SetOff(off bool) {
	ly.Off = off
	// a Path is off if either the sending or the receiving layer is off
	// or if the path has been set to Off directly
	for _, pj := range ly.RcvPaths {
		pj.SetOff(pj.SendLay().IsOff() || off)
	}
	for _, pj := range ly.SndPaths {
		pj.SetOff(pj.RecvLay().IsOff() || off)
	}
}
func (ly *LayerBase) Shape() *tensor.Shape       { return &ly.Shp }
func (ly *LayerBase) Is2D() bool                 { return ly.Shp.NumDims() == 2 }
func (ly *LayerBase) Is4D() bool                 { return ly.Shp.NumDims() == 4 }
func (ly *LayerBase) RelPos() relpos.Rel         { return ly.Rel }
func (ly *LayerBase) Pos() math32.Vector3        { return ly.Ps }
func (ly *LayerBase) SetPos(pos math32.Vector3)  { ly.Ps = pos }
func (ly *LayerBase) Index() int                 { return ly.Idx }
func (ly *LayerBase) SetIndex(idx int)           { ly.Idx = idx }
func (ly *LayerBase) RecvPaths() *AxonPaths      { return &ly.RcvPaths }
func (ly *LayerBase) NRecvPaths() int            { return len(ly.RcvPaths) }
func (ly *LayerBase) RecvPath(idx int) emer.Path { return ly.RcvPaths[idx] }
func (ly *LayerBase) SendPaths() *AxonPaths      { return &ly.SndPaths }
func (ly *LayerBase) NSendPaths() int            { return len(ly.SndPaths) }
func (ly *LayerBase) SendPath(idx int) emer.Path { return ly.SndPaths[idx] }
func (ly *LayerBase) RepIndexes() []int          { return ly.RepIxs }
func (ly *LayerBase) NeurStartIndex() int        { return int(ly.NeurStIndex) }

func (ly *LayerBase) SendNameTry(sender string) (emer.Path, error) {
	return emer.SendNameTry(ly.AxonLay, sender)
}
func (ly *LayerBase) SendNameTypeTry(sender, typ string) (emer.Path, error) {
	return emer.SendNameTypeTry(ly.AxonLay, sender, typ)
}
func (ly *LayerBase) RecvNameTry(receiver string) (emer.Path, error) {
	return emer.RecvNameTry(ly.AxonLay, receiver)
}
func (ly *LayerBase) RecvNameTypeTry(receiver, typ string) (emer.Path, error) {
	return emer.RecvNameTypeTry(ly.AxonLay, receiver, typ)
}

func (ly *LayerBase) SendName(sender string) *Path {
	pj, err := ly.SendNameTry(sender)
	if err != nil {
		log.Println(err)
	}
	return pj.(*Path) // will panic on nil
}
func (ly *LayerBase) SendNameType(sender, typ string) *Path {
	pj, err := ly.SendNameTypeTry(sender, typ)
	if err != nil {
		log.Println(err)
	}
	return pj.(*Path) // will panic on nil
}
func (ly *LayerBase) RecvName(receiver string) *Path {
	pj, err := ly.RecvNameTry(receiver)
	if err != nil {
		log.Println(err)
	}
	return pj.(*Path) // will panic on nil
}
func (ly *LayerBase) RecvNameType(receiver, typ string) *Path {
	pj, err := ly.RecvNameTypeTry(receiver, typ)
	if err != nil {
		log.Println(err)
	}
	return pj.(*Path) // will panic on nil
}

func (ly *LayerBase) Index4DFrom2D(x, y int) ([]int, bool) {
	lshp := ly.Shape()
	nux := lshp.DimSize(3)
	nuy := lshp.DimSize(2)
	ux := x % nux
	uy := y % nuy
	px := x / nux
	py := y / nuy
	idx := []int{py, px, uy, ux}
	if !lshp.IndexIsValid(idx) {
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

// PlaceRightOf positions the layer to the right of the other layer,
// with given spacing, using default YAlign = Front alignment
func (ly *LayerBase) PlaceRightOf(other *Layer, space float32) {
	ly.Rel = relpos.NewRightOf(other.Name(), space)
}

// PlaceBehind positions the layer behind the other layer,
// with given spacing, using default XAlign = Left alignment
func (ly *LayerBase) PlaceBehind(other *Layer, space float32) {
	ly.Rel = relpos.NewBehind(other.Name(), space)
}

// PlaceAbove positions the layer above the other layer,
// using default XAlign = Left, YAlign = Front alignment
func (ly *LayerBase) PlaceAbove(other *Layer) {
	ly.Rel = relpos.NewAbove(other.Name())
}

func (ly *LayerBase) Size() math32.Vector2 {
	if ly.Rel.Scale == 0 {
		ly.Rel.Defaults()
	}
	var sz math32.Vector2
	switch {
	case ly.Is2D():
		sz = math32.Vec2(float32(ly.Shp.DimSize(1)), float32(ly.Shp.DimSize(0))) // Y, X
	case ly.Is4D():
		// note: pool spacing is handled internally in display and does not affect overall size
		sz = math32.Vec2(float32(ly.Shp.DimSize(1)*ly.Shp.DimSize(3)), float32(ly.Shp.DimSize(0)*ly.Shp.DimSize(2))) // Y, X
	default:
		sz = math32.Vec2(float32(ly.Shp.Len()), 1)
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
	ly.Shp.SetShape(shape, dnms...)
}

// SetRepIndexesShape sets the RepIndexes, and RepShape and as list of dimension sizes
func (ly *LayerBase) SetRepIndexesShape(idxs, shape []int) {
	ly.RepIxs = idxs
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ly.RepShp.SetShape(shape, dnms...)
}

// RepShape returns the shape to use for representative units
func (ly *LayerBase) RepShape() *tensor.Shape {
	sz := len(ly.RepIxs)
	if sz == 0 {
		return &ly.Shp
	}
	if ly.RepShp.Len() < sz {
		ly.RepShp.SetShape([]int{sz})
	}
	return &ly.RepShp
}

// NSubPools returns the number of sub-pools of neurons
// according to the shape parameters.  2D shapes have 0 sub pools.
// For a 4D shape, the pools are the first set of 2 Y,X dims
// and then the neurons within the pools are the 2nd set of 2 Y,X dims.
func (ly *LayerBase) NSubPools() int {
	if ly.Shp.NumDims() != 4 {
		return 0
	}
	return ly.Shp.DimSize(0) * ly.Shp.DimSize(1)
}

// Pool returns pool at given pool x data index
func (ly *LayerBase) Pool(pi, di uint32) *Pool {
	return &(ly.Pools[pi*ly.MaxData+di])
}

// SubPool returns subpool for given neuron, at data index
func (ly *LayerBase) SubPool(ctx *Context, ni, di uint32) *Pool {
	pi := NrnI(ctx, ni, NrnSubPool)
	return ly.Pool(pi, di)
}

// LayerValues returns LayerValues at given data index
func (ly *LayerBase) LayerValues(di uint32) *LayerValues {
	return &(ly.Values[di])
}

// RecipToSendPath finds the reciprocal pathway to
// the given sending pathway within the ly layer.
// i.e., where ly is instead the *receiving* layer from same other layer B
// that is the receiver of the spj pathway we're sending to.
//
//	ly = A,  other layer = B:
//
// spj: S=A -> R=B
// rpj: R=A <- S=B
//
// returns false if not found.
func (ly *LayerBase) RecipToSendPath(spj *Path) (*Path, bool) {
	for _, rpj := range ly.RcvPaths {
		if rpj.SendLay() == spj.RecvLay() { // B = sender of rpj, recv of spj
			return rpj, true
		}
	}
	return nil, false
}

// RecipToRecvPath finds the reciprocal pathway to
// the given recv pathway within the ly layer.
// i.e., where ly is instead the *sending* layer to same other layer B
// that is the sender of the rpj pathway we're receiving from.
//
//	ly = A, other layer = B:
//
// rpj: R=A <- S=B
// spj: S=A -> R=B
//
// returns false if not found.
func (ly *LayerBase) RecipToRecvPath(rpj *Path) (*Path, bool) {
	for _, spj := range ly.SndPaths {
		if spj.RecvLay() == rpj.SendLay() { // B = sender of rpj, recv of spj
			return spj, true
		}
	}
	return nil, false
}

// Config configures the basic properties of the layer
func (ly *LayerBase) Config(shape []int, typ emer.LayerType) {
	ly.SetShape(shape)
	ly.Typ = LayerTypes(typ)
}

// ParamsHistoryReset resets parameter application history
func (ly *LayerBase) ParamsHistoryReset() {
	ly.ParamsHistory.ParamsHistoryReset()
	for _, pj := range ly.RcvPaths {
		pj.ParamsHistoryReset()
	}
}

// ParamsApplied is just to satisfy History interface so reset can be applied
func (ly *LayerBase) ParamsApplied(sel *params.Sel) {
	ly.ParamsHistory.ParamsApplied(sel)
}

// ApplyParams applies given parameter style Sheet to this layer and its recv pathways.
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
	for _, pj := range ly.RcvPaths {
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

// ApplyDefParams applies DefParams default parameters if set
// Called by Layer.Defaults()
func (ly *LayerBase) ApplyDefParams() {
	if ly.DefParams == nil {
		return
	}
	err := ly.DefParams.Apply(ly.AxonLay, false)
	if err != nil {
		log.Printf("programmer error -- fix DefParams: %s\n", err)
	}
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ly *LayerBase) NonDefaultParams() string {
	// ndfs := reflectx.NonDefaultFields(ly.AxonLay.AsAxon().Params)
	nds := "non default field strings todo"
	//Str(ly.AxonLay.AsAxon().Params, ly.Nm)
	for _, pj := range ly.RcvPaths {
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
		dly, err := ly.Network.LayByNameTry(rnm)
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
func (ly *LayerBase) BuildSubPools(ctx *Context) {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shp.Sizes
	spy := sh[0]
	spx := sh[1]
	pi := uint32(1)
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			soff := uint32(ly.Shp.Offset([]int{py, px, 0, 0}))
			eoff := uint32(ly.Shp.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1)
			for di := uint32(0); di < ly.MaxData; di++ {
				pl := ly.Pool(pi, di)
				pl.StIndex = soff
				pl.EdIndex = eoff
			}
			pl := ly.Pool(pi, 0)
			for lni := pl.StIndex; lni < pl.EdIndex; lni++ {
				ni := ly.NeurStIndex + lni
				SetNrnI(ctx, ni, NrnSubPool, uint32(pi))
			}
			pi++
		}
	}
}

// BuildPools builds the inhibitory pools structures -- nu = number of units in layer
func (ly *LayerBase) BuildPools(ctx *Context, nn uint32) error {
	np := 1 + ly.NSubPools()
	for di := uint32(0); di < ly.MaxData; di++ {
		lpl := ly.Pool(0, di)
		lpl.StIndex = 0
		lpl.EdIndex = nn
		lpl.IsLayPool.SetBool(true)
	}
	if np > 1 {
		ly.BuildSubPools(ctx)
	}
	return nil
}

// BuildPaths builds the pathways, send-side
func (ly *LayerBase) BuildPaths(ctx *Context) error {
	emsg := ""
	for _, pj := range ly.SndPaths {
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

// Build constructs the layer state, including calling Build on the pathways
func (ly *LayerBase) Build() error {
	ctx := &ly.Network.Ctx
	nn := uint32(ly.Shp.Len())
	if nn == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		SetNrnI(ctx, ni, NrnNeurIndex, lni)
		SetNrnI(ctx, ni, NrnLayIndex, uint32(ly.Idx))
	}
	err := ly.BuildPools(ctx, nn)
	if err != nil {
		return err
	}
	err = ly.BuildPaths(ctx)
	ly.AxonLay.PostBuild()
	return err
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *LayerBase) UnitVarNames() []string {
	return NeuronVarNames
}

// UnitVarProps returns properties for variables
func (ly *LayerBase) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// UnitVarIndex returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *LayerBase) UnitVarIndex(varNm string) (int, error) {
	return NeuronVarIndexByName(varNm)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *LayerBase) UnitVarNum() int {
	return len(NeuronVarNames)
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *LayerBase) UnitVal1D(varIndex int, idx, di int) float32 {
	if idx < 0 || idx >= int(ly.NNeurons) {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= ly.UnitVarNum() {
		return math32.NaN()
	}
	if di < 0 || di >= int(ly.MaxData) {
		return math32.NaN()
	}
	ni := ly.NeurStIndex + uint32(idx)
	ctx := &ly.Network.Ctx
	nvars := ly.UnitVarNum()
	if varIndex >= nvars-NNeuronLayerVars {
		lvi := varIndex - (ly.UnitVarNum() - NNeuronLayerVars)
		switch lvi {
		case 0:
			return GlbV(ctx, uint32(di), GvDA)
		case 1:
			return GlbV(ctx, uint32(di), GvACh)
		case 2:
			return GlbV(ctx, uint32(di), GvNE)
		case 3:
			return GlbV(ctx, uint32(di), GvSer)
		case 4:
			pl := ly.SubPool(ctx, ni, uint32(di))
			return float32(pl.Gated)
		}
	} else if varIndex >= int(NeuronVarsN) {
		return NrnAvgV(ctx, ni, NeuronAvgVars(varIndex-int(NeuronVarsN)))
	} else {
		return NrnV(ctx, ni, uint32(di), NeuronVars(varIndex))
	}
	return math32.NaN()
}

// UnitValues fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *LayerBase) UnitValues(vals *[]float32, varNm string, di int) error {
	nn := ly.NNeurons
	if *vals == nil || cap(*vals) < int(nn) {
		*vals = make([]float32, nn)
	} else if len(*vals) < int(nn) {
		*vals = (*vals)[0:nn]
	}
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math32.NaN()
		for lni := uint32(0); lni < nn; lni++ {
			(*vals)[lni] = nan
		}
		return err
	}
	for lni := uint32(0); lni < nn; lni++ {
		(*vals)[lni] = ly.UnitVal1D(vidx, int(lni), di)
	}
	return nil
}

// UnitValuesTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *LayerBase) UnitValuesTensor(tsr tensor.Tensor, varNm string, di int) error {
	if tsr == nil {
		err := fmt.Errorf("axon.UnitValuesTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	nn := int(ly.NNeurons)
	tsr.SetShape(ly.Shp.Sizes, ly.Shp.Names...)
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for lni := 0; lni < nn; lni++ {
			tsr.SetFloat1D(lni, nan)
		}
		return err
	}
	for lni := 0; lni < nn; lni++ {
		v := ly.UnitVal1D(vidx, lni, di)
		if math32.IsNaN(v) {
			tsr.SetFloat1D(lni, math.NaN())
		} else {
			tsr.SetFloat1D(lni, float64(v))
		}
	}
	return nil
}

// UnitValuesRepTensor fills in values of given variable name on unit
// for a smaller subset of representative units in the layer, into given tensor.
// This is used for computationally intensive stats or displays that work
// much better with a smaller number of units.
// The set of representative units are defined by SetRepIndexes -- all units
// are used if no such subset has been defined.
// If tensor is not already big enough to hold the values, it is
// set to RepShape to hold all the values if subset is defined,
// otherwise it calls UnitValuesTensor and is identical to that.
// Returns error on invalid var name.
func (ly *LayerBase) UnitValuesRepTensor(tsr tensor.Tensor, varNm string, di int) error {
	nu := len(ly.RepIxs)
	if nu == 0 {
		return ly.UnitValuesTensor(tsr, varNm, di)
	}
	if tsr == nil {
		err := fmt.Errorf("axon.UnitValuesRepTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	if tsr.Len() != nu {
		rs := ly.RepShape()
		tsr.SetShape(rs.Sizes, rs.Names...)
	}
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for i, _ := range ly.RepIxs {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i, ui := range ly.RepIxs {
		v := ly.UnitVal1D(vidx, ui, di)
		if math32.IsNaN(v) {
			tsr.SetFloat1D(i, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *LayerBase) UnitValue(varNm string, idx []int, di int) float32 {
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	fidx := ly.Shp.Offset(idx)
	return ly.UnitVal1D(vidx, fidx, di)
}

// RecvPathValues fills in values of given synapse variable name,
// for pathway into given sending layer and neuron 1D index,
// for all receiving neurons in this layer,
// into given float32 slice (only resized if not big enough).
// pathType is the string representation of the path type -- used if non-empty,
// useful when there are multiple pathways between two layers.
// Returns error on invalid var name.
// If the receiving neuron is not connected to the given sending layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv path (vals always set to nan on path err).
func (ly *LayerBase) RecvPathValues(vals *[]float32, varNm string, sendLay emer.Layer, sendIndex1D int, pathType string) error {
	var err error
	nn := int(ly.NNeurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("sending layer is nil")
	}
	var pj emer.Path
	if pathType != "" {
		pj, err = sendLay.RecvNameTypeTry(ly.Nm, pathType)
		if pj == nil {
			pj, err = sendLay.RecvNameTry(ly.Nm)
		}
	} else {
		pj, err = sendLay.RecvNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	if pj.IsOff() {
		return fmt.Errorf("pathway is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pj.SynValue(varNm, sendIndex1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

// SendPathValues fills in values of given synapse variable name,
// for pathway into given receiving layer and neuron 1D index,
// for all sending neurons in this layer,
// into given float32 slice (only resized if not big enough).
// pathType is the string representation of the path type -- used if non-empty,
// useful when there are multiple pathways between two layers.
// Returns error on invalid var name.
// If the sending neuron is not connected to the given receiving layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv path (vals always set to nan on path err).
func (ly *LayerBase) SendPathValues(vals *[]float32, varNm string, recvLay emer.Layer, recvIndex1D int, pathType string) error {
	var err error
	nn := int(ly.NNeurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("receiving layer is nil")
	}
	var pj emer.Path
	if pathType != "" {
		pj, err = recvLay.SendNameTypeTry(ly.Nm, pathType)
		if pj == nil {
			pj, err = recvLay.SendNameTry(ly.Nm)
		}
	} else {
		pj, err = recvLay.SendNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	if pj.IsOff() {
		return fmt.Errorf("pathway is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pj.SynValue(varNm, si, recvIndex1D)
	}
	return nil
}

// VarRange returns the min / max values for given variable
// todo: support r. s. pathway values
func (ly *LayerBase) VarRange(varNm string) (min, max float32, err error) {
	ctx := &ly.Network.Ctx
	nn := ly.NNeurons
	if nn == 0 {
		return
	}
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		return
	}
	nvar := NeuronVars(vidx)

	v0 := NrnV(ctx, ly.NeurStIndex, 0, nvar)
	min = v0
	max = v0
	for lni := uint32(1); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		vl := NrnV(ctx, ni, 0, nvar)
		if vl < min {
			min = vl
		}
		if vl > max {
			max = vl
		}
	}
	return
}

////////////////////////////////////////////
//  Weight Saving

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWtsJSON(w io.Writer, depth int) {
	ctx := &ly.Network.Ctx
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Layer\": %q,\n", ly.Nm)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth)) // note: ActAvg all sync'd so using first one is fine
	w.Write([]byte(fmt.Sprintf("\"ActMAvg\": \"%g\",\n", ly.Values[0].ActAvg.ActMAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActPAvg\": \"%g\",\n", ly.Values[0].ActAvg.ActPAvg)))
	w.Write(indent.TabBytes(depth))
	// note: last one has no comma
	w.Write([]byte(fmt.Sprintf("\"GiMult\": \"%g\"\n", ly.Values[0].ActAvg.GiMult)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	if ly.Params.IsLearnTrgAvg() {
		w.Write([]byte(fmt.Sprintf("\"Units\": {\n")))
		depth++

		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"ActAvg\": [ ")))
		nn := ly.NNeurons
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			nrnActAvg := NrnAvgV(ctx, ni, ActAvg)
			w.Write([]byte(fmt.Sprintf("%g", nrnActAvg)))
			if lni < nn-1 {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte(" ],\n"))

		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"TrgAvg\": [ ")))
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			nrnTrgAvg := NrnAvgV(ctx, ni, TrgAvg)
			w.Write([]byte(fmt.Sprintf("%g", nrnTrgAvg)))
			if lni < nn-1 {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte(" ]\n"))

		depth--
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("},\n"))
		w.Write(indent.TabBytes(depth))
	}

	onps := make(emer.Paths, 0, len(ly.RcvPaths))
	for _, pj := range ly.RcvPaths {
		if !pj.IsOff() {
			onps = append(onps, pj)
		}
	}
	np := len(onps)
	if np == 0 {
		w.Write([]byte(fmt.Sprintf("\"Paths\": null\n")))
	} else {
		w.Write([]byte(fmt.Sprintf("\"Paths\": [\n")))
		depth++
		for pi, pj := range onps {
			pj.WriteWtsJSON(w, depth) // this leaves path unterminated
			if pi == np-1 {
				w.Write([]byte("\n"))
			} else {
				w.Write([]byte(",\n"))
			}
		}
		depth--
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(" ]\n"))
	}
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("}")) // note: leave unterminated as outer loop needs to add , or just \n depending
}

// ReadWtsJSON reads the weights from this layer from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one layer only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (ly *Layer) ReadWtsJSON(r io.Reader) error {
	lw, err := weights.LayReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	return ly.SetWts(lw)
}

// SetWts sets the weights for this layer from weights.Layer decoded values
func (ly *Layer) SetWts(lw *weights.Layer) error {
	if ly.IsOff() {
		return nil
	}
	ctx := &ly.Network.Ctx
	if lw.MetaData != nil {
		for di := uint32(0); di < ly.MaxData; di++ {
			vals := &ly.Values[di]
			if am, ok := lw.MetaData["ActMAvg"]; ok {
				pv, _ := strconv.ParseFloat(am, 32)
				vals.ActAvg.ActMAvg = float32(pv)
			}
			if ap, ok := lw.MetaData["ActPAvg"]; ok {
				pv, _ := strconv.ParseFloat(ap, 32)
				vals.ActAvg.ActPAvg = float32(pv)
			}
			if gi, ok := lw.MetaData["GiMult"]; ok {
				pv, _ := strconv.ParseFloat(gi, 32)
				vals.ActAvg.GiMult = float32(pv)
			}
		}
	}
	if lw.Units != nil {
		if ta, ok := lw.Units["ActAvg"]; ok {
			for lni := range ta {
				if lni > int(ly.NNeurons) {
					break
				}
				ni := ly.NeurStIndex + uint32(lni)
				SetNrnAvgV(ctx, ni, ActAvg, ta[lni])
			}
		}
		if ta, ok := lw.Units["TrgAvg"]; ok {
			for lni := range ta {
				if lni > int(ly.NNeurons) {
					break
				}
				ni := ly.NeurStIndex + uint32(lni)
				SetNrnAvgV(ctx, ni, TrgAvg, ta[lni])
			}
		}
	}
	var err error
	if len(lw.Paths) == ly.NRecvPaths() { // this is essential if multiple paths from same layer
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pj := ly.RcvPaths[pi]
			er := pj.SetWts(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pj, _ := ly.SendNameTry(pw.From)
			if pj != nil {
				er := pj.SetWts(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	ly.AvgDifFromTrgAvg(ctx) // update AvgPct based on loaded ActAvg values
	return err
}
