// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"fmt"
	"io"
	"log"
	"strconv"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/weights"
)

// note: layerbase.go has all the basic infrastructure;
// layer.go has the algorithm-specific code.
// Everything is defined on Layer type.

// axon.Layer implements the basic Axon spiking activation function,
// and manages learning in the pathways.
type Layer struct {
	emer.LayerBase

	// layer parameters.
	Params *LayerParams

	// our parent network, in case we need to use it to find
	// other layers etc; set when added by network.
	Network *Network `copier:"-" json:"-" xml:"-" display:"-"`

	// type of layer.
	Type LayerTypes

	// number of neurons in the layer.
	NNeurons uint32 `display:"-"`

	// starting index of neurons for this layer within the global Network list.
	NeurStIndex uint32 `display:"-" inactive:"-"`

	// number of pools based on layer shape; at least 1 for layer pool + 4D subpools.
	NPools uint32 `display:"-"`

	// maximum amount of input data that can be processed in parallel
	// in one pass of the network.
	// Neuron, Pool, Values storage is allocated to hold this amount.
	MaxData uint32 `display:"-"`

	// list of receiving pathways into this layer from other layers
	RecvPaths []*Path

	// list of sending pathways from this layer to other layers
	SendPaths []*Path

	// layer-level state values that are updated during computation.
	// There is one for each data parallel.
	// This is a sub-slice of network full set.
	Values []LayerValues

	// computes FS-FFFB inhibition and other pooled, aggregate state variables.
	// has at least 1 for entire layer (lpl = layer pool), and one for each
	// sub-pool if shape supports that (4D) * 1 per data parallel (inner loop).
	// This is a sub-slice from overall Network Pools slice.
	// You must iterate over index and use pointer to modify values.
	Pools []Pool

	// external input values for this layer, allocated from network
	// global Exts slice.
	Exts []float32 `display:"-"`

	// configuration data set when the network is configured,
	// that is used during the network Build() process via PostBuild method,
	// after all the structure of the network has been fully constructed.
	// In particular, the Params is nil until Build, so setting anything
	// specific in there (e.g., an index to another layer) must be done
	//as a second pass.  Note that Params are all applied after Build
	// and can set user-modifiable params, so this is for more special
	// algorithm structural parameters set during ConfigNet() methods.
	BuildConfig map[string]string `table:"-"`

	// default parameters that are applied prior to user-set parameters.
	// These are useful for specific layer functionality in specialized
	// brain areas (e.g., Rubicon, BG etc) not associated with a layer type,
	// which otherwise is used to hard-code initial default parameters.
	// Typically just set to a literal map.
	DefaultParams params.Params `table:"-"`
}

// emer.Layer interface methods

func (ly *Layer) StyleObject() any           { return ly.Params }
func (ly *Layer) TypeName() string           { return ly.Type.String() }
func (ly *Layer) NumRecvPaths() int          { return len(ly.RecvPaths) }
func (ly *Layer) RecvPath(idx int) emer.Path { return ly.RecvPaths[idx] }
func (ly *Layer) NumSendPaths() int          { return len(ly.SendPaths) }
func (ly *Layer) SendPath(idx int) emer.Path { return ly.SendPaths[idx] }

// todo: not standard:
func (ly *Layer) SetOff(off bool) {
	ly.Off = off
	// a Path is off if either the sending or the receiving layer is off
	// or if the path has been set to Off directly
	for _, pt := range ly.RecvPaths {
		pt.Off = pt.Send.Off || off
	}
	for _, pt := range ly.SendPaths {
		pt.Off = pt.Recv.Off || off
	}
}

// Pool returns pool at given pool x data index
func (ly *Layer) Pool(pi, di uint32) *Pool {
	return &(ly.Pools[pi*ly.MaxData+di])
}

// SubPool returns subpool for given neuron, at data index
func (ly *Layer) SubPool(ctx *Context, ni, di uint32) *Pool {
	pi := NrnI(ctx, ni, NrnSubPool)
	return ly.Pool(pi, di)
}

// LayerValues returns LayerValues at given data index
func (ly *Layer) LayerValues(di uint32) *LayerValues {
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
func (ly *Layer) RecipToSendPath(spj *Path) (*Path, bool) {
	for _, rpj := range ly.RecvPaths {
		if rpj.Send == spj.Recv { // B = sender of rpj, recv of spj
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
func (ly *Layer) RecipToRecvPath(rpj *Path) (*Path, bool) {
	for _, spj := range ly.SendPaths {
		if spj.Recv == rpj.Send { // B = sender of rpj, recv of spj
			return spj, true
		}
	}
	return nil, false
}

// ApplyDefaultParams applies DefaultParams default parameters if set
// Called by Layer.Defaults()
func (ly *Layer) ApplyDefaultParams() {
	if ly.DefaultParams == nil {
		return
	}
	err := ly.DefaultParams.Apply(ly.EmerLayer, false)
	if err != nil {
		log.Printf("programmer error -- fix DefaultParams: %s\n", err)
	}
}

// AllParams returns a listing of all parameters in the Layer
func (ly *Layer) AllParams() string {
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Name + "\n" + ly.Params.AllParams()
	for _, pt := range ly.RecvPaths {
		str += pt.AllParams()
	}
	return str
}

////////////////////////////////////////////////////////////////////////
//  Build

// SetBuildConfig sets named configuration parameter to given string value
// to be used in the PostBuild stage -- mainly for layer names that need to be
// looked up and turned into indexes, after entire network is built.
func (ly *Layer) SetBuildConfig(param, val string) {
	ly.BuildConfig[param] = val
}

// BuildConfigByName looks for given BuildConfig option by name,
// and reports & returns an error if not found.
func (ly *Layer) BuildConfigByName(nm string) (string, error) {
	cfg, ok := ly.BuildConfig[nm]
	if !ok {
		err := fmt.Errorf("Layer: %s does not have BuildConfig: %s set -- error in ConfigNet", ly.Name, nm)
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
func (ly *Layer) BuildConfigFindLayer(nm string, mustName bool) int32 {
	idx := int32(-1)
	if rnm, ok := ly.BuildConfig[nm]; ok {
		dly := ly.Network.LayerByName(rnm)
		if dly != nil {
			idx = int32(dly.Index)
		}
	} else {
		if mustName {
			err := fmt.Errorf("Layer: %s does not have BuildConfig: %s set -- error in ConfigNet", ly.Name, nm)
			log.Println(err)
		}
	}
	return idx
}

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools(ctx *Context) {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shape.Sizes
	spy := sh[0]
	spx := sh[1]
	pi := uint32(1)
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			soff := uint32(ly.Shape.Offset([]int{py, px, 0, 0}))
			eoff := uint32(ly.Shape.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1)
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
func (ly *Layer) BuildPools(ctx *Context, nn uint32) error {
	np := 1 + ly.NumPools()
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
func (ly *Layer) BuildPaths(ctx *Context) error {
	emsg := ""
	for _, pt := range ly.SendPaths {
		if pt.Off {
			continue
		}
		err := pt.Build()
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
func (ly *Layer) Build() error {
	ctx := &ly.Network.Ctx
	nn := uint32(ly.Shape.Len())
	if nn == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Name)
	}
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		SetNrnI(ctx, ni, NrnNeurIndex, lni)
		SetNrnI(ctx, ni, NrnLayIndex, uint32(ly.Index))
	}
	err := ly.BuildPools(ctx, nn)
	if err != nil {
		return err
	}
	err = ly.BuildPaths(ctx)
	ly.PostBuild()
	return err
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVarNames
}

// UnitVarProps returns properties for variables
func (ly *Layer) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// UnitVarIndex returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIndex(varNm string) (int, error) {
	return NeuronVarIndexByName(varNm)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return len(NeuronVarNames)
}

// UnitValue1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitValue1D(varIndex int, idx, di int) float32 {
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
func (ly *Layer) RecvPathValues(vals *[]float32, varNm string, sendLay emer.Layer, sendIndex1D int, pathType string) error {
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
	slay := sendLay.AsEmer()
	var pt emer.Path
	if pathType != "" {
		pt, err = slay.SendPathByRecvNameType(ly.Name, pathType)
		if pt == nil {
			pt, err = slay.SendPathByRecvName(ly.Name)
		}
	} else {
		pt, err = slay.SendPathByRecvName(ly.Name)
	}
	if pt == nil {
		return err
	}
	if pt.AsEmer().Off {
		return fmt.Errorf("pathway is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pt.AsEmer().SynValue(varNm, sendIndex1D, ri) // this will work with any variable -- slower, but necessary
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
func (ly *Layer) SendPathValues(vals *[]float32, varNm string, recvLay emer.Layer, recvIndex1D int, pathType string) error {
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
	rlay := recvLay.AsEmer()
	var pt emer.Path
	if pathType != "" {
		pt, err = rlay.RecvPathBySendNameType(ly.Name, pathType)
		if pt == nil {
			pt, err = rlay.RecvPathBySendName(ly.Name)
		}
	} else {
		pt, err = rlay.RecvPathBySendName(ly.Name)
	}
	if pt == nil {
		return err
	}
	if pt.AsEmer().Off {
		return fmt.Errorf("pathway is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pt.AsEmer().SynValue(varNm, si, recvIndex1D)
	}
	return nil
}

// VarRange returns the min / max values for given variable
// todo: support r. s. pathway values
// error occurs when variable name is not found.
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
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
//		Weights

// WriteWeightsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWeightsJSON(w io.Writer, depth int) {
	ly.MetaData = make(map[string]string)
	ly.MetaData["ActMAvg"] = fmt.Sprintf("%g", ly.Values[0].ActAvg.ActMAvg)
	ly.MetaData["ActPAvg"] = fmt.Sprintf("%g", ly.Values[0].ActAvg.ActPAvg)
	ly.MetaData["GiMult"] = fmt.Sprintf("%g", ly.Values[0].ActAvg.GiMult)

	if ly.Params.IsLearnTrgAvg() {
		ly.LayerBase.WriteWeightsJSONBase(w, depth, "ActAvg", "TrgAvg")
	} else {
		ly.LayerBase.WriteWeightsJSONBase(w, depth)
	}
}

// SetWeights sets the weights for this layer from weights.Layer decoded values
func (ly *Layer) SetWeights(lw *weights.Layer) error {
	if ly.Off {
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
	if len(lw.Paths) == ly.NumRecvPaths() { // this is essential if multiple paths from same layer
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pt := ly.RecvPaths[pi]
			er := pt.SetWeights(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pt, _ := ly.RecvPathBySendName(pw.From)
			if pt != nil {
				er := pt.SetWeights(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	ly.AvgDifFromTrgAvg(ctx) // update AvgPct based on loaded ActAvg values
	return err
}
