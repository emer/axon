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
	"math/rand"
	"strconv"
	"strings"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gosl/sltype"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// axon.Layer implements the basic Axon spiking activation function,
// and manages learning in the projections.
type Layer struct {
	LayerBase
	Params  LayerParams `desc:"all layer-level parameters -- these must remain constant once configured"`
	Neurons []Neuron    `desc:"slice of neurons for this layer -- flat list of len = Shp.Len(). You must iterate over index and use pointer to modify values."`
	Pools   []Pool      `desc:"inhibition and other pooled, aggregate state variables -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
	Vals    LayerVals   `desc:"layer-level state values that are updated during computation"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// Object returns the object with parameters to be set by emer.Params
func (ly *Layer) Object() interface{} {
	return &ly.Params
}

func (ly *Layer) Defaults() {
	ly.Params.LayType = ly.LayerType()
	ly.Params.Defaults()
	ly.Vals.ActAvg.GiMult = 1
	for _, pj := range ly.RcvPrjns {
		pj.Defaults()
	}
	switch ly.LayerType() {
	case InputLayer:
		ly.Params.Act.Clamp.Ge = 1.5
		ly.Params.Inhib.Layer.Gi = 0.9
		ly.Params.Inhib.Pool.Gi = 0.9
		ly.Params.Learn.TrgAvgAct.SubMean = 0
	case TargetLayer:
		ly.Params.Act.Clamp.Ge = 0.8
		ly.Params.Learn.TrgAvgAct.SubMean = 0
		// ly.Params.Learn.RLRate.SigmoidMin = 1
	case CTLayer:
		ly.Params.CTLayerDefaults()
	case PulvinarLayer:
		ly.Params.PulvLayerDefaults()
	case RWPredLayer:
		ly.Params.RWPredLayerDefaults()
	}
}

// Update is an interface for generically updating after edits
// this should be used only for the values on the struct itself.
// UpdateParams is used to update all parameters, including Prjn.
func (ly *Layer) Update() {
	if !ly.Is4D() && ly.Params.Inhib.Pool.On.IsTrue() {
		ly.Params.Inhib.Pool.On.SetBool(false)
	}
	ly.Params.Update()
}

// UpdateParams updates all params given any changes that might
// have been made to individual values including those in the
// receiving projections of this layer.
// This is not called Update because it is not just about the
// local values in the struct.
func (ly *Layer) UpdateParams() {
	ly.Update()
	for _, pj := range ly.RcvPrjns {
		pj.UpdateParams()
	}
}

// HasPoolInhib returns true if the layer is using pool-level inhibition (implies 4D too).
// This is the proper check for using pool-level target average activations, for example.
func (ly *Layer) HasPoolInhib() bool {
	return ly.Params.Inhib.Pool.On.IsTrue()
}

// AsAxon returns this layer as a axon.Layer -- all derived layers must redefine
// this to return the base Layer type, so that the AxonLayer interface does not
// need to include accessors to all the basic stuff
func (ly *Layer) AsAxon() *Layer {
	return ly
}

// LayerType returns axon specific cast of ly.Typ layer type
func (ly *Layer) LayerType() LayerTypes {
	return LayerTypes(ly.Typ)
}

func (ly *Layer) Class() string {
	return ly.LayerType().String() + " " + ly.Cls
}

// JsonToParams reformates json output to suitable params display output
func JsonToParams(b []byte) string {
	br := strings.Replace(string(b), `"`, ``, -1)
	br = strings.Replace(br, ",\n", "", -1)
	br = strings.Replace(br, "{\n", "{", -1)
	br = strings.Replace(br, "} ", "}\n  ", -1)
	br = strings.Replace(br, "\n }", " }", -1)
	br = strings.Replace(br, "\n  }\n", " }", -1)
	return br[1:] + "\n"
}

// AllParams returns a listing of all parameters in the Layer
func (ly *Layer) AllParams() string {
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Nm + "\n" + ly.Params.AllParams()
	for _, pj := range ly.RcvPrjns {
		str += pj.AllParams()
	}
	return str
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns properties for variables
func (ly *Layer) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	return NeuronVarIdxByName(varNm)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return len(NeuronVars)
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= ly.UnitVarNum() {
		return mat32.NaN()
	}
	if varIdx >= ly.UnitVarNum()-NNeuronLayerVars {
		lvi := varIdx - (ly.UnitVarNum() - NNeuronLayerVars)
		switch lvi {
		case 0:
			return ly.Vals.NeuroMod.DA
		case 1:
			return ly.Vals.NeuroMod.ACh
		case 2:
			return ly.Vals.NeuroMod.NE
		}
	} else {
		nrn := &ly.Neurons[idx]
		return nrn.VarByIndex(varIdx)
	}
	return mat32.NaN()
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitVals(vals *[]float32, varNm string) error {
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		nan := mat32.NaN()
		for i := range ly.Neurons {
			(*vals)[i] = nan
		}
		return err
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.AxonLay.UnitVal1D(vidx, i)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("axon.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for i := range ly.Neurons {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i := range ly.Neurons {
		v := ly.AxonLay.UnitVal1D(vidx, i)
		if mat32.IsNaN(v) {
			tsr.SetFloat1D(i, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitValsRepTensor fills in values of given variable name on unit
// for a smaller subset of representative units in the layer, into given tensor.
// This is used for computationally intensive stats or displays that work
// much better with a smaller number of units.
// The set of representative units are defined by SetRepIdxs -- all units
// are used if no such subset has been defined.
// If tensor is not already big enough to hold the values, it is
// set to RepShape to hold all the values if subset is defined,
// otherwise it calls UnitValsTensor and is identical to that.
// Returns error on invalid var name.
func (ly *Layer) UnitValsRepTensor(tsr etensor.Tensor, varNm string) error {
	nu := len(ly.RepIxs)
	if nu == 0 {
		return ly.UnitValsTensor(tsr, varNm)
	}
	if tsr == nil {
		err := fmt.Errorf("axon.UnitValsRepTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	if tsr.Len() != nu {
		rs := ly.RepShape()
		tsr.SetShape(rs.Shp, rs.Strd, rs.Nms)
	}
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for i, _ := range ly.RepIxs {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i, ui := range ly.RepIxs {
		v := ly.AxonLay.UnitVal1D(vidx, ui)
		if mat32.IsNaN(v) {
			tsr.SetFloat1D(i, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitVal(varNm string, idx []int) float32 {
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	fidx := ly.Shp.Offset(idx)
	return ly.AxonLay.UnitVal1D(vidx, fidx)
}

// RecvPrjnVals fills in values of given synapse variable name,
// for projection into given sending layer and neuron 1D index,
// for all receiving neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the receiving neuron is not connected to the given sending layer or neuron
// then the value is set to mat32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) RecvPrjnVals(vals *[]float32, varNm string, sendLay emer.Layer, sendIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := mat32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("sending layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = sendLay.RecvNameTypeTry(ly.Nm, prjnType)
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
		return fmt.Errorf("projection is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pj.SynVal(varNm, sendIdx1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

// SendPrjnVals fills in values of given synapse variable name,
// for projection into given receiving layer and neuron 1D index,
// for all sending neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the sending neuron is not connected to the given receiving layer or neuron
// then the value is set to mat32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) SendPrjnVals(vals *[]float32, varNm string, recvLay emer.Layer, recvIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := mat32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("receiving layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = recvLay.SendNameTypeTry(ly.Nm, prjnType)
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
		return fmt.Errorf("projection is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pj.SynVal(varNm, si, recvIdx1D)
	}
	return nil
}

// Pool returns pool at given index
func (ly *Layer) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

// PoolTry returns pool at given index, returns error if index is out of range
func (ly *Layer) PoolTry(idx int) (*Pool, error) {
	np := len(ly.Pools)
	if idx < 0 || idx >= np {
		return nil, fmt.Errorf("Layer Pool index: %v out of range, N = %v", idx, np)
	}
	return &(ly.Pools[idx]), nil
}

func (ly *Layer) SendNameTry(sender string) (emer.Prjn, error) {
	return emer.SendNameTry(ly, sender)
}
func (ly *Layer) SendNameTypeTry(sender, typ string) (emer.Prjn, error) {
	return emer.SendNameTypeTry(ly, sender, typ)
}
func (ly *Layer) RecvNameTry(receiver string) (emer.Prjn, error) {
	return emer.RecvNameTry(ly, receiver)
}
func (ly *Layer) RecvNameTypeTry(receiver, typ string) (emer.Prjn, error) {
	return emer.RecvNameTypeTry(ly, receiver, typ)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools() {
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
func (ly *Layer) BuildPools(nu int) error {
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
func (ly *Layer) BuildPrjns() error {
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
func (ly *Layer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	// note: ly.Neurons are allocated by Network from global network pool
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.LayIdx = uint32(ly.Idx)
	}
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	return err
}

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWtsJSON(w io.Writer, depth int) {
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Layer\": %q,\n", ly.Nm)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActMAvg\": \"%g\",\n", ly.Vals.ActAvg.ActMAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActPAvg\": \"%g\",\n", ly.Vals.ActAvg.ActPAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GiMult\": \"%g\"\n", ly.Vals.ActAvg.GiMult)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	if ly.IsLearnTrgAvg() {
		w.Write([]byte(fmt.Sprintf("\"Units\": {\n")))
		depth++

		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"ActAvg\": [ ")))
		nn := len(ly.Neurons)
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			w.Write([]byte(fmt.Sprintf("%g", nrn.ActAvg)))
			if ni < nn-1 {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte(" ],\n"))

		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"TrgAvg\": [ ")))
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			w.Write([]byte(fmt.Sprintf("%g", nrn.TrgAvg)))
			if ni < nn-1 {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte(" ]\n"))

		depth--
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("},\n"))
		w.Write(indent.TabBytes(depth))
	}

	onps := make(emer.Prjns, 0, len(ly.RcvPrjns))
	for _, pj := range ly.RcvPrjns {
		if !pj.IsOff() {
			onps = append(onps, pj)
		}
	}
	np := len(onps)
	if np == 0 {
		w.Write([]byte(fmt.Sprintf("\"Prjns\": null\n")))
	} else {
		w.Write([]byte(fmt.Sprintf("\"Prjns\": [\n")))
		depth++
		for pi, pj := range onps {
			pj.WriteWtsJSON(w, depth) // this leaves prjn unterminated
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
	if lw.MetaData != nil {
		if am, ok := lw.MetaData["ActMAvg"]; ok {
			pv, _ := strconv.ParseFloat(am, 32)
			ly.Vals.ActAvg.ActMAvg = float32(pv)
		}
		if ap, ok := lw.MetaData["ActPAvg"]; ok {
			pv, _ := strconv.ParseFloat(ap, 32)
			ly.Vals.ActAvg.ActPAvg = float32(pv)
		}
		if gi, ok := lw.MetaData["GiMult"]; ok {
			pv, _ := strconv.ParseFloat(gi, 32)
			ly.Vals.ActAvg.GiMult = float32(pv)
		}
	}
	if lw.Units != nil {
		if ta, ok := lw.Units["ActAvg"]; ok {
			for ni := range ta {
				if ni > len(ly.Neurons) {
					break
				}
				nrn := &ly.Neurons[ni]
				nrn.ActAvg = ta[ni]
			}
		}
		if ta, ok := lw.Units["TrgAvg"]; ok {
			for ni := range ta {
				if ni > len(ly.Neurons) {
					break
				}
				nrn := &ly.Neurons[ni]
				nrn.TrgAvg = ta[ni]
			}
		}
	}
	var err error
	rpjs := ly.RecvPrjns()
	if len(lw.Prjns) == len(*rpjs) { // this is essential if multiple prjns from same layer
		for pi := range lw.Prjns {
			pw := &lw.Prjns[pi]
			pj := (*rpjs)[pi]
			er := pj.SetWts(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Prjns {
			pw := &lw.Prjns[pi]
			pj, _ := ly.SendNameTry(pw.From)
			if pj != nil {
				er := pj.SetWts(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	ly.AvgDifFmTrgAvg() // update AvgPct based on loaded ActAvg values
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
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

// note: all basic computation can be performed on layer-level and prjn level

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes the weight values in the network, i.e., resetting learning
// Also calls InitActs
func (ly *Layer) InitWts() {
	ly.AxonLay.UpdateParams()
	ly.Vals.ActAvg.ActMAvg = ly.Params.Inhib.ActAvg.Nominal
	ly.Vals.ActAvg.ActPAvg = ly.Params.Inhib.ActAvg.Nominal
	ly.Vals.ActAvg.AvgMaxGeM = 1
	ly.Vals.ActAvg.AvgMaxGiM = 1
	ly.Vals.ActAvg.GiMult = 1
	ly.AxonLay.InitActAvg()
	ly.AxonLay.InitActs()
	ly.Vals.CorSim.Init()

	ly.AxonLay.InitGScale()

	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.InitWts()
	}
}

// InitActAvg initializes the running-average activation values that drive learning.
// and the longer time averaging values.
func (ly *Layer) InitActAvg() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Params.Learn.InitNeurCa(nrn)
	}
	strg := ly.Params.Learn.TrgAvgAct.TrgRange.Min
	rng := ly.Params.Learn.TrgAvgAct.TrgRange.Range()
	inc := float32(0)
	if ly.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		nNy := ly.Shp.Dim(2)
		nNx := ly.Shp.Dim(3)
		nn := nNy * nNx
		if nn > 1 {
			inc = rng / float32(nn-1)
		}
		np := len(ly.Pools)
		porder := make([]int, nn)
		for i := range porder {
			porder[i] = i
		}
		for pi := 1; pi < np; pi++ {
			pl := &ly.Pools[pi]
			if ly.Params.Learn.TrgAvgAct.Permute.IsTrue() {
				erand.PermuteInts(porder)
			}
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				vi := porder[ni-pl.StIdx]
				nrn.TrgAvg = strg + inc*float32(vi)
				nrn.AvgPct = nrn.TrgAvg
				nrn.ActAvg = ly.Params.Inhib.ActAvg.Nominal * nrn.TrgAvg
				nrn.AvgDif = 0
				nrn.DTrgAvg = 0
			}
		}
	} else {
		nn := len(ly.Neurons)
		if nn > 1 {
			inc = rng / float32(nn-1)
		}
		porder := make([]int, nn)
		for i := range porder {
			porder[i] = i
		}
		if ly.Params.Learn.TrgAvgAct.Permute.IsTrue() {
			erand.PermuteInts(porder)
		}
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			vi := porder[ni]
			nrn.TrgAvg = strg + inc*float32(vi)
			nrn.AvgPct = nrn.TrgAvg
			nrn.ActAvg = ly.Params.Inhib.ActAvg.Nominal * nrn.TrgAvg
			nrn.AvgDif = 0
			nrn.DTrgAvg = 0
		}
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs() {
	ly.Params.Act.Clamp.IsInput.SetBool(ly.AxonLay.IsInput())
	ly.Params.Act.Clamp.IsTarget.SetBool(ly.AxonLay.IsTarget())
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Params.Act.InitActs(nrn)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Init()
		pl.ActM.Init()
		pl.ActP.Init()
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsInput.IsTrue() {
			pl.Inhib.Clamped.SetBool(true)
		}
		// todo: need to dynamically update Target layers!
	}
	ly.InitPrjnGBuffs()
}

// InitPrjnGBuffs initializes the projection-level conductance buffers and
// conductance integration values for receiving projections in this layer.
func (ly *Layer) InitPrjnGBuffs() {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).InitGBuffs()
	}
}

// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
func (ly *Layer) InitWtSym() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		plp := p.(AxonPrjn)
		if plp.AsAxon().Params.SWt.Init.Sym.IsFalse() {
			continue
		}
		// key ordering constraint on which way weights are copied
		if p.RecvLay().Index() < p.SendLay().Index() {
			continue
		}
		rpj, has := ly.RecipToSendPrjn(p)
		if !has {
			continue
		}
		rlp := rpj.(AxonPrjn)
		if rlp.AsAxon().Params.SWt.Init.Sym.IsFalse() {
			continue
		}
		plp.InitWtSym(rlp)
	}
}

// InitExt initializes external input state -- called prior to apply ext
func (ly *Layer) InitExt() {
	msk := NeuronHasExt | NeuronHasTarg | NeuronHasCmpr
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Target = 0
		nrn.ClearFlag(msk)
	}
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Target (else Ext)
func (ly *Layer) ApplyExtFlags() (clrmsk, setmsk NeuronFlags, toTarg bool) {
	clrmsk = NeuronHasExt | NeuronHasTarg | NeuronHasCmpr
	toTarg = false
	if ly.Typ == emer.Target {
		setmsk = NeuronHasTarg
		toTarg = true
	} else if ly.Typ == emer.Compare {
		setmsk = NeuronHasCmpr
		toTarg = true
	} else {
		setmsk = NeuronHasExt
	}
	return
}

// ApplyExt applies external input in the form of an etensor.Float32.  If
// dimensionality of tensor matches that of layer, and is 2D or 4D, then each dimension
// is iterated separately, so any mismatch preserves dimensional structure.
// Otherwise, the flat 1D view of the tensor is used.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt(ext etensor.Tensor) {
	switch {
	case ext.NumDims() == 2 && ly.Shp.NumDims() == 4: // special case
		ly.ApplyExt2Dto4D(ext)
	case ext.NumDims() != ly.Shp.NumDims() || !(ext.NumDims() == 2 || ext.NumDims() == 4):
		ly.ApplyExt1DTsr(ext)
	case ext.NumDims() == 2:
		ly.ApplyExt2D(ext)
	case ext.NumDims() == 4:
		ly.ApplyExt4D(ext)
	}
}

// ApplyExt2D applies 2D tensor external input
func (ly *Layer) ApplyExt2D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	ymx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatVal(idx))
			i := ly.Shp.Offset(idx)
			nrn := &ly.Neurons[i]
			if nrn.IsOff() {
				continue
			}
			if toTarg {
				nrn.Target = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearFlag(clrmsk)
			nrn.SetFlag(setmsk)
		}
	}
}

// ApplyExt2Dto4D applies 2D tensor external input to a 4D layer
func (ly *Layer) ApplyExt2Dto4D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	lNy, lNx, _, _ := etensor.Prjn2DShape(&ly.Shp, false)

	ymx := ints.MinInt(ext.Dim(0), lNy)
	xmx := ints.MinInt(ext.Dim(1), lNx)
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatVal(idx))
			ui := etensor.Prjn2DIdx(&ly.Shp, false, y, x)
			nrn := &ly.Neurons[ui]
			if nrn.IsOff() {
				continue
			}
			if toTarg {
				nrn.Target = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearFlag(clrmsk)
			nrn.SetFlag(setmsk)
		}
	}
}

// ApplyExt4D applies 4D tensor external input
func (ly *Layer) ApplyExt4D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	ypmx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xpmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	ynmx := ints.MinInt(ext.Dim(2), ly.Shp.Dim(2))
	xnmx := ints.MinInt(ext.Dim(3), ly.Shp.Dim(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					vl := float32(ext.FloatVal(idx))
					i := ly.Shp.Offset(idx)
					nrn := &ly.Neurons[i]
					if nrn.IsOff() {
						continue
					}
					if toTarg {
						nrn.Target = vl
					} else {
						nrn.Ext = vl
					}
					nrn.ClearFlag(clrmsk)
					nrn.SetFlag(setmsk)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1DTsr(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(ext.Len(), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := float32(ext.FloatVal1D(i))
		if toTarg {
			nrn.Target = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearFlag(clrmsk)
		nrn.SetFlag(setmsk)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ext []float64) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := float32(ext[i])
		if toTarg {
			nrn.Target = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearFlag(clrmsk)
		nrn.SetFlag(setmsk)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ext []float32) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := ext[i]
		if toTarg {
			nrn.Target = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearFlag(clrmsk)
		nrn.SetFlag(setmsk)
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (ly *Layer) UpdateExtFlags() {
	clrmsk, setmsk, _ := ly.ApplyExtFlags()
	for i := range ly.Neurons {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		nrn.ClearFlag(clrmsk)
		nrn.SetFlag(setmsk)
	}
}

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (ly *Layer) NewState() {
	pl := &ly.Pools[0]
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActMAvg, pl.ActM.Avg, ly.Params.Act.Dt.LongAvgDt)
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActPAvg, pl.ActP.Avg, ly.Params.Act.Dt.LongAvgDt)

	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsTarget.IsTrue() {
			pl.Inhib.Clamped.SetBool(false)
		}
	}

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkPrv = nrn.CaSpkD
		nrn.SpkMax = 0
		nrn.SpkMaxCa = 0
	}
	ly.AxonLay.DecayState(ly.Params.Act.Decay.Act, ly.Params.Act.Decay.Glong)
}

// InitGScale computes the initial scaling factor for synaptic input conductances G,
// stored in GScale.Scale, based on sending layer initial activation.
func (ly *Layer) InitGScale() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(AxonPrjn).AsAxon()
		slay := p.SendLay().(AxonLayer).AsAxon()
		savg := slay.Params.Inhib.ActAvg.Nominal
		snu := len(slay.Neurons)
		ncon := pj.RecvConNAvgMax.Avg
		pj.Params.GScale.Scale = pj.Params.PrjnScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a prjn, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.Params.GScale == 0 {
		// 	continue
		// }
		if pj.Typ == emer.Inhib {
			totGiRel += pj.Params.PrjnScale.Rel
		} else {
			totGeRel += pj.Params.PrjnScale.Rel
		}
	}

	for _, p := range ly.RcvPrjns {
		pj := p.(AxonPrjn).AsAxon()
		if pj.Typ == emer.Inhib {
			if totGiRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PrjnScale.Rel / totGiRel
				pj.Params.GScale.Scale /= totGiRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0
			}
		} else {
			if totGeRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PrjnScale.Rel / totGeRel
				pj.Params.GScale.Scale /= totGeRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0
			}
		}
	}
}

// DecayState decays activation state by given proportion
// (default decay values are ly.Params.Act.Decay.Act, Glong)
func (ly *Layer) DecayState(decay, glong float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Act.DecayState(nrn, decay, glong)
		// ly.Params.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
		// Note: synapse-level Ca decay happens in DWt
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Decay(decay)
	}
	if glong != 0 { // clear pipeline of incoming spikes, assuming time has passed
		ly.InitPrjnGBuffs()
	}
}

// DecayCaLrnSpk decays neuron-level calcium learning and spiking variables
// by given factor, which is typically ly.Params.Act.Decay.Glong.
// Note: this is NOT called by default and is generally
// not useful, causing variability in these learning factors as a function
// of the decay parameter that then has impacts on learning rates etc.
// It is only here for reference or optional testing.
func (ly *Layer) DecayCaLrnSpk(decay float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Learn.DecayCaLrnSpk(nrn, decay)
	}
}

// DecayStatePool decays activation state by given proportion in given sub-pool index (0 based)
func (ly *Layer) DecayStatePool(pool int, decay, glong float32) {
	pi := int32(pool + 1) // 1 based
	pl := &ly.Pools[pi]
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Act.DecayState(nrn, decay, glong)
	}
	pl.Inhib.Decay(decay)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle
//  note: these are calls to LayerParams methods that have the core computation

// Prjns.PrjnGatherSpikes called first, at Network level for all prjns

// GiFmSpikes integrates new inhibitory conductances from Spikes
// at the layer and pool level.
// Called separately by Network.CycleImpl on all Layers
func (ly *Layer) GiFmSpikes(ctx *Context) {
	lpl := &ly.Pools[0]
	subPools := (len(ly.Pools) > 1)
	for ni := range ly.Neurons { // note: layer-level iterating across neurons
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		ly.Params.GeExtToPool(ctx, uint32(ni), nrn, pl, lpl, subPools) // todo: can this be done in send spike?
		// todo: deal with this in plus phase and new state methods.
		// if ly.Params.Act.Clamp.Add.IsFalse() && nrn.HasFlag(NeuronHasExt) {
		// 	pl.Inhib.Clamped.SetBool(true)
		// 	lpl.Inhib.Clamped.SetBool(true)
		// }
	}
	ly.Params.LayPoolGiFmSpikes(ctx, lpl, &ly.Vals)
	ly.PoolGiFmSpikes(ctx)
}

// PoolGiFmSpikes computes inhibition Gi from Spikes within relevant Pools
func (ly *Layer) PoolGiFmSpikes(ctx *Context) {
	np := len(ly.Pools)
	if np == 1 {
		return
	}
	lpl := &ly.Pools[0]
	lyInhib := ly.Params.Inhib.Layer.On.IsTrue()
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		ly.Params.SubPoolGiFmSpikes(ctx, pl, lpl, lyInhib, ly.Vals.ActAvg.GiMult)
	}
}

// NeuronGatherSpikes integrates G*Raw and G*Syn values for given neuron
// from the Prjn-level GSyn integrated values.
func (ly *Layer) NeuronGatherSpikes(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.NeuronGatherSpikesInit(ctx, ni, nrn)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.AsAxon()
		gv := pj.GVals[ni]
		pj.Params.NeuronGatherSpikesPrjn(ctx, gv, ni, nrn)
	}
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls NeuronGatherSpikes, GFmRawSyn, GiInteg
func (ly *Layer) GInteg(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, giMult float32, randctr *sltype.Uint2) {
	ly.NeuronGatherSpikes(ctx, ni, nrn)

	drvGe := float32(0)
	nonDrvPct := float32(0)
	if ly.LayerType() == PulvinarLayer {
		dly := ly.Network.Layer(int(ly.Params.Pulv.DriveLayIdx)).(AxonLayer).AsAxon()
		drvMax := dly.Vals.ActAvg.CaSpkP.Max
		nonDrvPct = ly.Params.Pulv.NonDrivePct(drvMax) // how much non-driver to keep
		dneur := dly.Neurons[ni]
		if dly.LayerType() == SuperLayer {
			drvGe = ly.Params.Pulv.DriveGe(dneur.Burst)
		} else {
			drvGe = ly.Params.Pulv.DriveGe(dneur.CaSpkP)
		}
	}

	saveVal := ly.Params.SpecialPreGs(ctx, ni, nrn, drvGe, nonDrvPct, randctr)

	ly.Params.GFmRawSyn(ctx, ni, nrn, randctr)
	ly.Params.GiInteg(ctx, ni, nrn, pl, giMult)

	ly.Params.SpecialPostGs(ctx, ni, nrn, randctr, saveVal)
}

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *Layer) SpikeFmG(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.SpikeFmG(ctx, ni, nrn)
}

// CycleNeuron does one cycle (msec) of updating at the neuron level
func (ly *Layer) CycleNeuron(ni uint32, nrn *Neuron, ctx *Context) {
	randctr := ctx.RandCtr.Uint2() // use local var so updates are local
	ly.AxonLay.GInteg(ctx, ni, nrn, &ly.Pools[nrn.SubPool], ly.Vals.ActAvg.GiMult, &randctr)
	ly.AxonLay.SpikeFmG(ctx, ni, nrn)
	ly.AxonLay.PostSpike(ctx, ni, nrn)
}

// PostSpike does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
func (ly *Layer) PostSpike(ctx *Context, ni uint32, nrn *Neuron) {
	switch ly.LayerType() {
	case RWDaLayer:
		rly := ly.Network.Layer(int(ly.Params.RWDa.RewLayIdx)).(AxonLayer).AsAxon()
		ply := ly.Network.Layer(int(ly.Params.RWDa.RWPredLayIdx)).(AxonLayer).AsAxon()
		rnrn := &(rly.Neurons[0])
		hasRew := false
		if rnrn.HasFlag(NeuronHasExt) {
			hasRew = true
		}
		ract := rnrn.Act
		pnrn := &(ply.Neurons[0])
		pact := pnrn.Act
		if hasRew {
			ly.Vals.NeuroMod.DA = ract - pact
		} else {
			ly.Vals.NeuroMod.DA = 0 // nothing
		}
	}

	ly.Params.PostSpike(ctx, ni, nrn, &ly.Vals)
}

// SendSpike sends spike to receivers for all neurons that spiked
// last step in Cycle, integrated the next time around.
func (ly *Layer) SendSpike(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Spike == 0 {
			continue
		}
		ly.Pools[nrn.SubPool].Inhib.FBsRaw += 1.0 // note: this is immediate..
		if nrn.SubPool > 0 {
			ly.Pools[0].Inhib.FBsRaw += 1.0
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.SendSpike(ni)
		}
	}
}

// CyclePost is called after the standard Cycle update
// still within layer Cycle call.
// This is the hook for specialized algorithms (deep, hip, bg etc)
// to do something special after Spike / Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (ly *Layer) CyclePost(ctx *Context) {
	switch ly.LayerType() {
	case RWDaLayer:
		rly := ly.Network.Layer(int(ly.Params.RWDa.RewLayIdx)).(AxonLayer).AsAxon()
		ply := ly.Network.Layer(int(ly.Params.RWDa.RWPredLayIdx)).(AxonLayer).AsAxon()
		rnrn := &(rly.Neurons[0])
		hasRew := false
		if rnrn.HasFlag(NeuronHasExt) {
			hasRew = true
		}
		ract := rnrn.Act
		pnrn := &(ply.Neurons[0])
		pact := pnrn.Act
		da := float32(0)
		if hasRew {
			da = ract - pact
		}
		ctx.NeuroMod.DA = da // updates global value that will be copied to layers next cycle.
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Phase-level

// AvgMaxVarByPool returns the average and maximum value of given variable
// for given pool index (0 = entire layer, 1.. are subpools for 4D only).
// Uses fast index-based variable access.
func (ly *Layer) AvgMaxVarByPool(varNm string, poolIdx int) minmax.AvgMax32 {
	var am minmax.AvgMax32
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		log.Printf("axon.Layer.AvgMaxVar: %s\n", err)
		return am
	}
	pl := &ly.Pools[poolIdx]
	am.Init()
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		vl := ly.UnitVal1D(vidx, int(ni))
		am.UpdateVal(vl, int(ni))
	}
	am.CalcAvg()
	return am
}

// AvgGeM computes the average and max GeM stats, updated in MinusPhase
func (ly *Layer) AvgGeM(ctx *Context) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActM = ly.AvgMaxVarByPool("ActM", pi)
		pl.GeM = ly.AvgMaxVarByPool("GeM", pi)
		pl.GiM = ly.AvgMaxVarByPool("GiM", pi)
	}
	lpl := &ly.Pools[0]
	ly.Vals.ActAvg.AvgMaxGeM += ly.Params.Act.Dt.LongAvgDt * (lpl.GeM.Max - ly.Vals.ActAvg.AvgMaxGeM)
	ly.Vals.ActAvg.AvgMaxGiM += ly.Params.Act.Dt.LongAvgDt * (lpl.GiM.Max - ly.Vals.ActAvg.AvgMaxGiM)
}

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ctx *Context) {
	ly.Vals.ActAvg.CaSpkPM = ly.AvgMaxVarByPool("CaSpkP", 0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActM = nrn.ActInt
		nrn.CaSpkPM = nrn.CaSpkP
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Target
			nrn.SetFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
			nrn.ActInt = ly.Params.Act.Init.Act // reset for plus phase
		}
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsTarget.IsTrue() {
			pl.Inhib.Clamped.SetBool(true)
		}
	}
	ly.AvgGeM(ctx)
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ctx *Context) {
	ly.Vals.ActAvg.CaSpkP = ly.AvgMaxVarByPool("CaSpkP", 0)
	ly.Vals.ActAvg.CaSpkD = ly.AvgMaxVarByPool("CaSpkD", 0)

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActP = nrn.ActInt
		mlr := ly.Params.Learn.RLRate.RLRateSigDeriv(nrn.CaSpkD, ly.Vals.ActAvg.CaSpkD.Max)
		dlr := ly.Params.Learn.RLRate.RLRateDiff(nrn.CaSpkP, nrn.CaSpkD)
		nrn.RLRate = mlr * dlr
		nrn.ActAvg += ly.Params.Act.Dt.LongAvgDt * (nrn.ActM - nrn.ActAvg)
		var tau float32
		ly.Params.Act.Sahp.NinfTauFmCa(nrn.SahpCa, &nrn.SahpN, &tau)
		nrn.SahpCa = ly.Params.Act.Sahp.CaInt(nrn.SahpCa, nrn.CaSpkD)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActP = ly.AvgMaxVarByPool("ActP", pi)
	}
	ly.AxonLay.CorSimFmActs()
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) TargToExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Target
			nrn.SetFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) ClearTargExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = 0
			nrn.ClearFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// SpkSt1 saves current activation state in SpkSt1 variables (using CaP)
func (ly *Layer) SpkSt1(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkSt1 = nrn.CaSpkP
	}
}

// SpkSt2 saves current activation state in SpkSt2 variables (using CaP)
func (ly *Layer) SpkSt2(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkSt2 = nrn.CaSpkP
	}
}

// CorSimFmActs computes the correlation similarity
// (centered cosine aka normalized dot product)
// in activation state between minus and plus phases.
func (ly *Layer) CorSimFmActs() {
	lpl := &ly.Pools[0]
	avgM := lpl.ActM.Avg
	avgP := lpl.ActP.Avg
	cosv := float32(0)
	ssm := float32(0)
	ssp := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ap := nrn.ActP - avgP // zero mean = correl
		am := nrn.ActM - avgM
		cosv += ap * am
		ssm += am * am
		ssp += ap * ap
	}

	dist := mat32.Sqrt(ssm * ssp)
	if dist != 0 {
		cosv /= dist
	}
	ly.Vals.CorSim.Cor = cosv

	ly.Params.Act.Dt.AvgVarUpdt(&ly.Vals.CorSim.Avg, &ly.Vals.CorSim.Var, ly.Vals.CorSim.Cor)
}

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == emer.Target
// Other Target layers include the TRCLayer in deep predictive learning.
// It is used in SynScale to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *Layer) IsTarget() bool {
	switch ly.LayerType() {
	case TargetLayer:
		return true
	case PulvinarLayer:
		return true
	default:
		return false
	}
}

// IsInput returns true if this layer is an Input layer.
// By default, returns true for layers of Type == emer.Input
// Used to prevent adapting of inhibition or TrgAvg values.
func (ly *Layer) IsInput() bool {
	switch ly.LayerType() {
	case InputLayer:
		return true
	default:
		return false
	}
}

// IsInputOrTarget returns true if this layer is either an Input
// or a Target layer.
func (ly *Layer) IsInputOrTarget() bool {
	return (ly.AxonLay.IsTarget() || ly.AxonLay.IsInput())
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

func (ly *Layer) IsLearnTrgAvg() bool {
	if ly.AxonLay.IsTarget() || ly.AxonLay.IsInput() || ly.Params.Learn.TrgAvgAct.On.IsFalse() {
		return false
	}
	return true
}

// DWtLayer does weight change at the layer level.
// does NOT call main projection-level DWt method.
// in base, only calls DTrgAvgFmErr
func (ly *Layer) DWtLayer(ctx *Context) {
	ly.DTrgAvgFmErr()
}

// DTrgAvgFmErr computes change in TrgAvg based on unit-wise error signal
// Called by DWtLayer at the layer level
func (ly *Layer) DTrgAvgFmErr() {
	if !ly.IsLearnTrgAvg() {
		return
	}
	lr := ly.Params.Learn.TrgAvgAct.ErrLRate
	if lr == 0 {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.DTrgAvg += lr * (nrn.CaSpkP - nrn.CaSpkD) // CaP - CaD almost as good in ra25 -- todo explore
	}
}

// DTrgSubMean subtracts the mean from DTrgAvg values
// Called by TrgAvgFmD
func (ly *Layer) DTrgSubMean() {
	submean := ly.Params.Learn.TrgAvgAct.SubMean
	if submean == 0 {
		return
	}
	if ly.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		np := len(ly.Pools)
		for pi := 1; pi < np; pi++ {
			pl := &ly.Pools[pi]
			nn := 0
			avg := float32(0)
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				avg += nrn.DTrgAvg
				nn++
			}
			if nn == 0 {
				continue
			}
			avg /= float32(nn)
			avg *= submean
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				nrn.DTrgAvg -= avg
			}
		}
	} else {
		nn := 0
		avg := float32(0)
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			avg += nrn.DTrgAvg
			nn++
		}
		if nn == 0 {
			return
		}
		avg /= float32(nn)
		avg *= submean
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			nrn.DTrgAvg -= avg
		}
	}
}

// TrgAvgFmD updates TrgAvg from DTrgAvg
// it is called by WtFmDWtLayer
func (ly *Layer) TrgAvgFmD() {
	if !ly.IsLearnTrgAvg() || ly.Params.Learn.TrgAvgAct.ErrLRate == 0 {
		return
	}
	ly.DTrgSubMean()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.TrgAvg = ly.Params.Learn.TrgAvgAct.TrgRange.ClipVal(nrn.TrgAvg + nrn.DTrgAvg)
		nrn.DTrgAvg = 0
	}
}

// WtFmDWtLayer does weight update at the layer level.
// does NOT call main projection-level WtFmDWt method.
// in base, only calls TrgAvgFmD
func (ly *Layer) WtFmDWtLayer(ctx *Context) {
	ly.TrgAvgFmD()
}

// SlowAdapt is the layer-level slow adaptation functions.
// Calls AdaptInhib and AvgDifFmTrgAvg for Synaptic Scaling.
// Does NOT call projection-level methods.
func (ly *Layer) SlowAdapt(ctx *Context) {
	ly.AdaptInhib(ctx)
	ly.AvgDifFmTrgAvg()
	// note: prjn level call happens at network level
}

// AdaptInhib adapts inhibition
func (ly *Layer) AdaptInhib(ctx *Context) {
	if ly.Params.Inhib.ActAvg.AdaptGi.IsFalse() || ly.AxonLay.IsInput() {
		return
	}
	ly.Params.Inhib.ActAvg.Adapt(&ly.Vals.ActAvg.GiMult, ly.Vals.ActAvg.ActMAvg)
}

// AvgDifFmTrgAvg updates neuron-level AvgDif values from AvgPct - TrgAvg
// which is then used for synaptic scaling of LWt values in Prjn SynScale.
func (ly *Layer) AvgDifFmTrgAvg() {
	sp := 0
	if len(ly.Pools) > 1 {
		sp = 1
	}
	np := len(ly.Pools)
	for pi := sp; pi < np; pi++ {
		pl := &ly.Pools[pi]
		plavg := float32(0)
		nn := 0
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			plavg += nrn.ActAvg
			nn++
		}
		if nn == 0 {
			continue
		}
		plavg /= float32(nn)
		pl.AvgDif.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			nrn.AvgPct = nrn.ActAvg / plavg
			nrn.AvgDif = nrn.AvgPct - nrn.TrgAvg
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int(ni))
		}
		pl.AvgDif.CalcAvg()
	}
	if sp == 1 { // update stats
		pl := &ly.Pools[0]
		pl.AvgDif.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int(ni))
		}
		pl.AvgDif.CalcAvg()
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (ly *Layer) SynFail(ctx *Context) {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).SynFail(ctx)
	}
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateMod(mod float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateSched(sched float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// prjn is for the prjns Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (ly *Layer) SetSubMean(trgAvg, prjn float32) {
	ly.Params.Learn.TrgAvgAct.SubMean = trgAvg
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().Params.Learn.Trace.SubMean = prjn
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Threading / Reports

// CostEst returns the estimated computational cost associated with this layer,
// separated by neuron-level and synapse-level, in arbitrary units where
// cost per synapse is 1.  Neuron-level computation is more expensive but
// there are typically many fewer neurons, so in larger networks, synaptic
// costs tend to dominate.  Neuron cost is estimated from TimerReport output
// for large networks.
func (ly *Layer) CostEst() (neur, syn, tot int) {
	perNeur := 300 // cost per neuron, relative to synapse which is 1
	neur = len(ly.Neurons) * perNeur
	syn = 0
	for _, pji := range ly.SndPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		ns := len(pj.Syns)
		syn += ns
	}
	tot = neur + syn
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Stats

// note: use float64 for stats as that is best for logging

// PctUnitErr returns the proportion of units where the thresholded value of
// Target (Target or Compare types) or ActP does not match that of ActM.
// If Act > ly.Params.Act.Clamp.ErrThr, effective activity = 1 else 0
// robust to noisy activations.
func (ly *Layer) PctUnitErr() float64 {
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	thr := ly.Params.Act.Clamp.ErrThr
	wrong := 0
	n := 0
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		trg := false
		if ly.Typ == emer.Compare || ly.Typ == emer.Target {
			if nrn.Target > thr {
				trg = true
			}
		} else {
			if nrn.ActP > thr {
				trg = true
			}
		}
		if nrn.ActM > thr {
			if !trg {
				wrong++
			}
		} else {
			if trg {
				wrong++
			}
		}
		n++
	}
	if n > 0 {
		return float64(wrong) / float64(n)
	}
	return 0
}

// LocalistErr2D decodes a 2D layer with Y axis = redundant units, X = localist units
// returning the indexes of the max activated localist value in the minus and plus phase
// activities, and whether these are the same or different (err = different)
func (ly *Layer) LocalistErr2D() (err bool, minusIdx, plusIdx int) {
	ydim := ly.Shp.Dim(0)
	xdim := ly.Shp.Dim(1)
	var maxM, maxP float32
	for xi := 0; xi < xdim; xi++ {
		var sumP, sumM float32
		for yi := 0; yi < ydim; yi++ {
			ni := yi*xdim + xi
			nrn := &ly.Neurons[ni]
			sumM += nrn.ActM
			sumP += nrn.ActP
		}
		if sumM > maxM {
			minusIdx = xi
			maxM = sumM
		}
		if sumP > maxP {
			plusIdx = xi
			maxP = sumP
		}
	}
	err = minusIdx != plusIdx
	return
}

// LocalistErr4D decodes a 4D layer with each pool representing a localist value.
// Returns the flat 1D indexes of the max activated localist value in the minus and plus phase
// activities, and whether these are the same or different (err = different)
func (ly *Layer) LocalistErr4D() (err bool, minusIdx, plusIdx int) {
	npool := ly.Shp.Dim(0) * ly.Shp.Dim(1)
	nun := ly.Shp.Dim(2) * ly.Shp.Dim(3)
	var maxM, maxP float32
	for xi := 0; xi < npool; xi++ {
		var sumP, sumM float32
		for yi := 0; yi < nun; yi++ {
			ni := xi*nun + yi
			nrn := &ly.Neurons[ni]
			sumM += nrn.ActM
			sumP += nrn.ActP
		}
		if sumM > maxM {
			minusIdx = xi
			maxM = sumM
		}
		if sumP > maxP {
			plusIdx = xi
			maxP = sumP
		}
	}
	err = minusIdx != plusIdx
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.ClearFlag(NeuronOff)
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(prop float32) int {
	ly.UnLesionNeurons()
	if prop > 1 {
		log.Printf("LesionNeurons got a proportion > 1 -- must be 0-1 as *proportion* (not percent) of neurons to lesion: %v\n", prop)
		return 0
	}
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	p := rand.Perm(nn)
	nl := int(prop * float32(nn))
	for i := 0; i < nl; i++ {
		nrn := &ly.Neurons[p[i]]
		nrn.SetFlag(NeuronOff)
	}
	return nl
}

//////////////////////////////////////////////////////////////////////////////////////
//  Layer props for gui

var LayerProps = ki.Props{
	"EnumType:Typ": KiT_LayerTypes, // uses our LayerTypes for GUI
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
