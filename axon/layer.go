// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/emer/emergent/edge"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/ki/bitflag"
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
	Act     ActParams       `view:"add-fields" desc:"Activation parameters and methods for computing activations"`
	Inhib   InhibParams     `view:"add-fields" desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	Learn   LearnNeurParams `view:"add-fields" desc:"Learning parameters and methods that operate at the neuron level"`
	Neurons []Neuron        `desc:"slice of neurons for this layer -- flat list of len = Shp.Len(). You must iterate over index and use pointer to modify values."`
	Pools   []Pool          `desc:"inhibition and other pooled, aggregate state variables -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
	ActAvg  ActAvgVals      `view:"inline" desc:"running-average activation levels used for Ge scaling and adaptive inhibition"`
	CorSim  CorSimStats     `desc:"correlation (centered cosine aka normalized dot product) similarity between ActM, ActP states"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

func (ly *Layer) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On = true
	ly.Inhib.Layer.Gi = 1.0
	ly.Inhib.Pool.Gi = 1.0
	ly.ActAvg.GiMult = 1
	for _, pj := range ly.RcvPrjns {
		pj.Defaults()
	}
	switch ly.Typ {
	case emer.Input:
		ly.Act.Clamp.Ge = 1.0
		ly.Inhib.Layer.Gi = 0.9
		ly.Inhib.Pool.Gi = 0.9
	case emer.Target:
		ly.Act.Clamp.Ge = 0.6
		ly.Learn.RLrate.SigmoidMin = 1
	}
}

// todo: why is this UpdateParams and not just Update()?

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	if !ly.Is4D() && ly.Inhib.Pool.On {
		ly.Inhib.Pool.On = false
	}
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	for _, pj := range ly.RcvPrjns {
		pj.UpdateParams()
	}
}

// HasPoolInhib returns true if the layer is using pool-level inhibition (implies 4D too).
// This is the proper check for using pool-level target average activations, for example.
func (ly *Layer) HasPoolInhib() bool {
	return ly.Inhib.Pool.On
}

// ActAvgVals are running-average activation levels used for Ge scaling and adaptive inhibition
type ActAvgVals struct {
	ActMAvg   float32         `inactive:"+" desc:"running-average minus-phase activity integrated at Dt.LongAvgTau -- used for adapting inhibition relative to target level"`
	ActPAvg   float32         `inactive:"+" desc:"running-average plus-phase activity integrated at Dt.LongAvgTau"`
	AvgMaxGeM float32         `inactive:"+" desc:"running-average max of minus-phase Ge value across the layer integrated at Dt.LongAvgTau -- used for adjusting the GScale.Scale relative to the GTarg.MaxGe value -- see Prjn PrjnScale"`
	AvgMaxGiM float32         `inactive:"+" desc:"running-average max of minus-phase Gi value across the layer integrated at Dt.LongAvgTau -- used for adjusting the GScale.Scale relative to the GTarg.MaxGi value -- see Prjn PrjnScale"`
	GiMult    float32         `inactive:"+" desc:"multiplier on inhibition -- adapted to maintain target activity level"`
	CaSpkPM   minmax.AvgMax32 `inactive:"+" desc:"maximum CaSpkP value in layer in the minus phase -- for monitoring network activity levels"`
	CaSpkP    minmax.AvgMax32 `inactive:"+" desc:"maximum CaSpkP value in layer -- for RLrate computation"`
}

// CorSimStats holds correlation similarity (centered cosine aka normalized dot product)
// statistics at the layer level
type CorSimStats struct {
	Cor float32 `inactive:"+" desc:"correlation (centered cosine aka normalized dot product) activation difference between ActP and ActM on this alpha-cycle for this layer -- computed by CorSimFmActs called by PlusPhase"`
	Avg float32 `inactive:"+" desc:"running average of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`
	Var float32 `inactive:"+" desc:"running variance of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`
}

func (cd *CorSimStats) Init() {
	cd.Cor = 0
	cd.Avg = 0
}

// AsAxon returns this layer as a axon.Layer -- all derived layers must redefine
// this to return the base Layer type, so that the AxonLayer interface does not
// need to include accessors to all the basic stuff
func (ly *Layer) AsAxon() *Layer {
	return ly
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
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Nm + "\n"
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	for _, pj := range ly.RcvPrjns {
		pstr := pj.AllParams()
		str += pstr
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
	nrn := &ly.Neurons[idx]
	return nrn.VarByIndex(varIdx)
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
		pj, err = sendLay.SendPrjns().RecvNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
		}
	} else {
		pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
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
		pj, err = recvLay.RecvPrjns().SendNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
		}
	} else {
		pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
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
			soff := ly.Shp.Offset([]int{py, px, 0, 0})
			eoff := ly.Shp.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1
			pl := &ly.Pools[pi]
			pl.StIdx = soff
			pl.EdIdx = eoff
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				nrn.SubPool = int32(pi)
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
	lpl.EdIdx = nu
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
	ly.Neurons = make([]Neuron, nu)
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
	w.Write([]byte(fmt.Sprintf("\"ActMAvg\": \"%g\",\n", ly.ActAvg.ActMAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActPAvg\": \"%g\",\n", ly.ActAvg.ActPAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"AvgMaxGeM\": \"%g\",\n", ly.ActAvg.AvgMaxGeM)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"AvgMaxGiM\": \"%g\",\n", ly.ActAvg.AvgMaxGiM)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GiMult\": \"%g\"\n", ly.ActAvg.GiMult)))
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
			ly.ActAvg.ActMAvg = float32(pv)
		}
		if ap, ok := lw.MetaData["ActPAvg"]; ok {
			pv, _ := strconv.ParseFloat(ap, 32)
			ly.ActAvg.ActPAvg = float32(pv)
		}
		if ap, ok := lw.MetaData["AvgMaxGeM"]; ok {
			pv, _ := strconv.ParseFloat(ap, 32)
			ly.ActAvg.AvgMaxGeM = float32(pv)
		}
		if ap, ok := lw.MetaData["AvgMaxGiM"]; ok {
			pv, _ := strconv.ParseFloat(ap, 32)
			ly.ActAvg.AvgMaxGiM = float32(pv)
		}
		if gi, ok := lw.MetaData["GiMult"]; ok {
			pv, _ := strconv.ParseFloat(gi, 32)
			ly.ActAvg.GiMult = float32(pv)
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
			pj := rpjs.SendName(pw.From)
			if pj != nil {
				er := pj.SetWts(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	ly.SynScale() // update AvgPct based on loaded ActAvg values
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
	ly.ActAvg.ActMAvg = ly.Inhib.ActAvg.Init
	ly.ActAvg.ActPAvg = ly.Inhib.ActAvg.Init
	ly.ActAvg.AvgMaxGeM = 1
	ly.ActAvg.AvgMaxGiM = 1
	ly.ActAvg.GiMult = 1
	ly.AxonLay.InitActAvg()
	ly.AxonLay.InitActs()
	ly.CorSim.Init()

	ly.AxonLay.InitGScale()

	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).InitWts()
	}
}

// InitActAvg initializes the running-average activation values that drive learning.
// and the longer time averaging values.
func (ly *Layer) InitActAvg() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Learn.InitNeurCa(nrn)
	}
	strg := ly.Learn.TrgAvgAct.TrgRange.Min
	rng := ly.Learn.TrgAvgAct.TrgRange.Range()
	inc := float32(0)
	if ly.HasPoolInhib() && ly.Learn.TrgAvgAct.Pool {
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
			if ly.Learn.TrgAvgAct.Permute {
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
				nrn.ActAvg = ly.Inhib.ActAvg.Init * nrn.TrgAvg
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
		if ly.Learn.TrgAvgAct.Permute {
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
			nrn.ActAvg = ly.Inhib.ActAvg.Init * nrn.TrgAvg
			nrn.AvgDif = 0
			nrn.DTrgAvg = 0
		}
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.InitActs(nrn)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Init()
		pl.ActM.Init()
		pl.ActP.Init()
	}
	ly.InitRecvGBuffs()
}

// InitRecvGBuffs initializes the receiving layer conductance buffers
func (ly *Layer) InitRecvGBuffs() {
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
		if !(plp.AsAxon().SWt.Init.Sym) {
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
		if !(rlp.AsAxon().SWt.Init.Sym) {
			continue
		}
		plp.InitWtSym(rlp)
	}
}

// InitExt initializes external input state -- called prior to apply ext
func (ly *Layer) InitExt() {
	msk := bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Targ = 0
		nrn.ClearMask(msk)
	}
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Targ (else Ext)
func (ly *Layer) ApplyExtFlags() (clrmsk, setmsk int32, toTarg bool) {
	clrmsk = bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	toTarg = false
	if ly.Typ == emer.Target {
		setmsk = bitflag.Mask32(int(NeurHasTarg))
		toTarg = true
	} else if ly.Typ == emer.Compare {
		setmsk = bitflag.Mask32(int(NeurHasCmpr))
		toTarg = true
	} else {
		setmsk = bitflag.Mask32(int(NeurHasExt))
	}
	return
}

// ApplyExt applies external input in the form of an etensor.Float32.  If
// dimensionality of tensor matches that of layer, and is 2D or 4D, then each dimension
// is iterated separately, so any mismatch preserves dimensional structure.
// Otherwise, the flat 1D view of the tensor is used.
// If the layer is a Target or Compare layer type, then it goes in Targ
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
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearMask(clrmsk)
			nrn.SetMask(setmsk)
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
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearMask(clrmsk)
			nrn.SetMask(setmsk)
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
						nrn.Targ = vl
					} else {
						nrn.Ext = vl
					}
					nrn.ClearMask(clrmsk)
					nrn.SetMask(setmsk)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Targ
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
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Targ
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
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Targ
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
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
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
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (ly *Layer) NewState() {
	pl := &ly.Pools[0]
	ly.Inhib.ActAvg.AvgFmAct(&ly.ActAvg.ActMAvg, pl.ActM.Avg, ly.Act.Dt.LongAvgDt)
	ly.Inhib.ActAvg.AvgFmAct(&ly.ActAvg.ActPAvg, pl.ActP.Avg, ly.Act.Dt.LongAvgDt)

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkPrv = nrn.CaSpkD
		nrn.SpkMax = 0
	}
	ly.AxonLay.DecayState(ly.Act.Decay.Act, ly.Act.Decay.Glong)
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
		savg := slay.Inhib.ActAvg.Init
		snu := len(slay.Neurons)
		ncon := pj.RConNAvgMax.Avg
		pj.GScale.Scale = pj.PrjnScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a prjn, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.GScale == 0 {
		// 	continue
		// }
		if pj.Typ == emer.Inhib {
			totGiRel += pj.PrjnScale.Rel
		} else {
			totGeRel += pj.PrjnScale.Rel
		}
	}

	for _, p := range ly.RcvPrjns {
		pj := p.(AxonPrjn).AsAxon()
		if pj.Typ == emer.Inhib {
			if totGiRel > 0 {
				pj.GScale.Rel = pj.PrjnScale.Rel / totGiRel
				pj.GScale.Scale /= totGiRel
			} else {
				pj.GScale.Rel = 0
				pj.GScale.Scale = 0
			}
		} else {
			if totGeRel > 0 {
				pj.GScale.Rel = pj.PrjnScale.Rel / totGeRel
				pj.GScale.Scale /= totGeRel
			} else {
				pj.GScale.Rel = 0
				pj.GScale.Scale = 0
			}
		}
		pj.GScale.Init()
	}
}

// DecayState decays activation state by given proportion
// (default decay values are ly.Act.Decay.Act, Glong)
func (ly *Layer) DecayState(decay, glong float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.DecayState(nrn, decay, glong)
		// ly.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
		// Note: synapse-level Ca decay happens in DWt
	}
	for pi := range ly.Pools { // decaying average act is essential for inhib
		pl := &ly.Pools[pi]
		pl.Inhib.Decay(decay)
	}
	if glong == 1 {
		ly.InitRecvGBuffs()
	}
}

// DecayCaLrnSpk decays neuron-level calcium learning and spiking variables
// by given factor, which is typically ly.Act.Decay.Glong.
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
		ly.Learn.DecayCaLrnSpk(nrn, decay)
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
		ly.Act.DecayState(nrn, decay, glong)
	}
	pl.Inhib.Decay(decay)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// Cycle does one cycle of updating
func (ly *Layer) Cycle(ltime *Time) {
	ly.AxonLay.GFmInc(ltime)
	ly.AxonLay.AvgMaxGe(ltime)
	ly.AxonLay.InhibFmGeAct(ltime)
	ly.AxonLay.ActFmG(ltime)
	ly.AxonLay.PostAct(ltime)
	ly.AxonLay.CyclePost(ltime)
	ly.AxonLay.SendSpike(ltime)
}

// GFmInc integrates new synaptic conductances from increments sent during last Spike
func (ly *Layer) GFmInc(ltime *Time) {
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.GFmIncNeur(ltime, nrn, 0) // no extra
	}
}

// RecvGInc calls RecvGInc on receiving projections to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *Layer) RecvGInc(ltime *Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).RecvGInc(ltime)
	}
}

// GFmIncNeur is the neuron-level code for GFmInc that integrates overall Ge, Gi values
// from their G*Raw accumulators.  Takes an extra increment to add to geRaw
func (ly *Layer) GFmIncNeur(ltime *Time, nrn *Neuron, geExt float32) {
	// note: GABAB integrated in ActFmG one timestep behind, b/c depends on integrated Gi inhib
	ly.Act.NMDAFmRaw(nrn, geExt)
	ly.Learn.LrnNMDAFmRaw(nrn, geExt)
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmRaw(nrn, nrn.GeRaw+geExt, nrn.Gnmda+nrn.Gvgcc)
	nrn.GeRaw = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
// This operates at the pool level so does not make sense to combine with GFmInc
func (ly *Layer) AvgMaxGe(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Ge.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.Inhib.Ge.UpdateVal(nrn.Ge, ni)
		}
		pl.Inhib.Ge.CalcAvg()
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib, ly.ActAvg.GiMult)
	ly.PoolInhibFmGeAct(ltime)
	ly.TopoGi(ltime)
	ly.InhibFmPool(ltime)
}

// PoolInhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) PoolInhibFmGeAct(ltime *Time) {
	np := len(ly.Pools)
	if np == 1 {
		return
	}
	lpl := &ly.Pools[0]
	lyInhib := ly.Inhib.Layer.On
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		ly.Inhib.Pool.Inhib(&pl.Inhib, ly.ActAvg.GiMult)
		if lyInhib {
			pl.Inhib.LayGi = lpl.Inhib.Gi
			pl.Inhib.Gi = mat32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // pool is max of layer
		} else {
			lpl.Inhib.Gi = mat32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // update layer from pool
		}
	}
	if !lyInhib {
		lpl.Inhib.GiOrig = lpl.Inhib.Gi // effective GiOrig
	}
}

// TopoGi computes topographic Gi inhibition
// todo: this does not work for 2D layers, and in general needs more testing
func (ly *Layer) TopoGi(ltime *Time) {
	if !ly.Inhib.Topo.On {
		return
	}
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	wd := ly.Inhib.Topo.Width
	wrap := ly.Inhib.Topo.Wrap

	ssq := ly.Inhib.Topo.Sigma * float32(wd)
	ssq *= ssq
	ff0 := ly.Inhib.Topo.FF0

	l4d := ly.Is4D()

	var clip bool
	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			var tge, tact, twt float32
			for iy := -wd; iy <= wd; iy++ {
				ty := py + iy
				if ty, clip = edge.Edge(ty, pyn, wrap); clip {
					continue
				}
				for ix := -wd; ix <= wd; ix++ {
					tx := px + ix
					if tx, clip = edge.Edge(tx, pxn, wrap); clip {
						continue
					}
					ds := float32(iy*iy + ix*ix)
					df := mat32.Sqrt(ds)
					di := int(mat32.Round(df))
					if di > wd {
						continue
					}
					wt := mat32.FastExp(-0.5 * ds / ssq)
					twt += wt
					ti := ty*pxn + tx
					if l4d {
						pl := &ly.Pools[ti+1]
						tge += wt * pl.Inhib.Ge.Avg
						tact += wt * pl.Inhib.Act.Avg
					} else {
						nrn := &ly.Neurons[ti]
						tge += wt * nrn.Ge
						tact += wt * nrn.Act
					}
				}
			}

			gi := ly.Inhib.Topo.GiFmGeAct(tge, tact, ff0*twt)
			pi := py*pxn + px
			if l4d {
				pl := &ly.Pools[pi+1]
				pl.Inhib.Gi += gi
				// } else {
				// 	nrn := &ly.Neurons[pi]
				// 	nrn.GiSelf = gi
			}
		}
	}
}

// InhibFmPool computes inhibition Gi from Pool-level aggregated inhibition, including self and syn
func (ly *Layer) InhibFmPool(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		nrn.Gi = pl.Inhib.Gi + ly.Inhib.Inhib.GiSyn(nrn.GiSyn+nrn.GiNoise)
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *Time) {
	intdt := ly.Act.Dt.IntDt
	if ltime.PlusPhase {
		intdt *= 3.0
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		ly.Learn.CaFmSpike(nrn)
		if ltime.Cycle >= ly.Act.Dt.MaxCycStart && nrn.CaSpkP > nrn.SpkMax {
			nrn.SpkMax = nrn.CaSpkP
		}
		nrn.ActInt += intdt * (nrn.Act - nrn.ActInt) // using reg act here now
		if !ltime.PlusPhase {
			nrn.GeM += ly.Act.Dt.IntDt * (nrn.Ge - nrn.GeM)
			nrn.GiM += ly.Act.Dt.IntDt * (nrn.GiSyn - nrn.GiM)
		}
		// note: this is here because it depends on Gi
		nrn.GABAB, nrn.GABABx = ly.Act.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi)
		nrn.GgabaB = ly.Act.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)
		if ly.Act.KNa.On {
			nrn.Gk += nrn.GgabaB // Gk was set by KNa
		} else {
			nrn.Gk = nrn.GgabaB
		}
	}
}

// PostAct does updates after activation (spiking) updated for all neurons,
// including the running-average activation used in driving inhibition,
// and synaptic-level calcium updates depending on spiking, NMDA
func (ly *Layer) PostAct(ltime *Time) {
	ly.AvgMaxAct(ltime)
	ly.AvgMaxCaSpkP(ltime)
	if !ltime.Testing {
		ly.AxonLay.SynCa(ltime)
	}
}

// AvgMaxAct updates the running-average activation used in driving inhibition
func (ly *Layer) AvgMaxAct(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		var avg, max float32
		maxi := 0
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			// in theory having quicker activation than Act will be useful, but maybe the delay is
			// not such a big deal, and otherwise it tracks pretty smoothly and closely
			// todo: need to come back and fix this with a combination of PV and SST inhibitory
			// neurons: general story that PV is fast onset but depleting, targets axon initial,
			// SST is slower and faciltating / not depleting, hits dendrites.
			// https://github.com/emer/axon/discussions/64
			avg += nrn.Act // SpkCaM, SpkCaP not clearly better, nor is spike..
			if nrn.Act > max {
				max = nrn.Act
				maxi = ni
			}
		}
		nn := pl.EdIdx - pl.StIdx
		pl.Inhib.Act.Sum = avg
		pl.Inhib.Act.N = nn
		if nn > 1 {
			avg /= float32(nn)
		}
		ly.Inhib.Inhib.AvgAct(&pl.Inhib.Act.Avg, avg)
		pl.Inhib.Act.Max = max
		pl.Inhib.Act.MaxIdx = maxi
	}
}

// AvgMaxCaSpkP computes the average and max CaSpkP in the layer.
// This is a useful direct measure of time-integrated neural activity
// used for learning and other functions, instead of Act, which is
// purely for visualization.
func (ly *Layer) AvgMaxCaSpkP(ltime *Time) {
	ly.ActAvg.CaSpkP.Init()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.ActAvg.CaSpkP.UpdateVal(nrn.CaSpkP, ni)
	}
	ly.ActAvg.CaSpkP.CalcAvg()
}

// SpikedAvgByPool returns the average across Spiked values by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) SpikedAvgByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	sum := float32(0)
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		sum += nrn.Spiked
	}
	return sum / float32(pl.EdIdx-pl.StIdx)
}

// SpikeAvgByPool returns the average across Spike values by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) SpikeAvgByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	sum := float32(0)
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		sum += nrn.Spike
	}
	return sum / float32(pl.EdIdx-pl.StIdx)
}

// MaxSpkMax returns the maximum SpkMax across the layer
func (ly *Layer) MaxSpkMax() float32 {
	mx := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.SpkMax > mx {
			mx = nrn.SpkMax
		}
	}
	return mx
}

// SpkMaxAvgByPool returns the average SpkMax value by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) SpkMaxAvgByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	sum := float32(0)
	cnt := 0
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		sum += nrn.SpkMax
		cnt++
	}
	if cnt > 0 {
		sum /= float32(cnt)
	}
	return sum
}

// SpkMaxMaxByPool returns the average SpkMax value by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) SpkMaxMaxByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	max := float32(0)
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.SpkMax > max {
			max = nrn.SpkMax
		}
	}
	return max
}

// AvgGeM computes the average and max GeM stats
func (ly *Layer) AvgGeM(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.GeM.Init()
		pl.GiM.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.GeM.UpdateVal(nrn.GeM, ni)
			pl.GiM.UpdateVal(nrn.GiM, ni)
		}
		pl.GeM.CalcAvg()
		pl.GiM.CalcAvg()
	}
	lpl := &ly.Pools[0]
	ly.ActAvg.AvgMaxGeM += ly.Act.Dt.LongAvgDt * (lpl.GeM.Max - ly.ActAvg.AvgMaxGeM)
	ly.ActAvg.AvgMaxGiM += ly.Act.Dt.LongAvgDt * (lpl.GiM.Max - ly.ActAvg.AvgMaxGiM)
}

// CyclePost is called after the standard Cycle update
// still within layer Cycle call.
// This is the hook for specialized algorithms (deep, hip, bg etc)
// to do something special after Spike / Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (ly *Layer) CyclePost(ltime *Time) {
}

// SendSpike sends spike to receivers -- last step in Cycle, integrated
// the next time around.
func (ly *Layer) SendSpike(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() || nrn.Spike == 0 {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.(AxonPrjn).SendSpike(ni)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Phase-level

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ltime *Time) {
	ly.ActAvg.CaSpkPM.Init()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.ActAvg.CaSpkPM.UpdateVal(nrn.CaSpkP, ni)
		nrn.ActM = nrn.ActInt
		if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Targ
			nrn.SetFlag(NeurHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
			nrn.ActInt = ly.Act.Init.Act // reset for plus phase
		}
	}
	ly.ActAvg.CaSpkPM.CalcAvg()
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActM.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.ActM.UpdateVal(nrn.ActM, ni)
		}
		pl.ActM.CalcAvg()
	}
	ly.AvgGeM(ltime)
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActP = nrn.ActInt
		mlr := ly.Learn.RLrate.RLrateSigDeriv(nrn.CaSpkP, ly.ActAvg.CaSpkP.Max)
		dlr := ly.Learn.RLrate.RLrateDiff(nrn.CaSpkP, nrn.CaSpkD)
		nrn.RLrate = mlr * dlr
		nrn.ActAvg += ly.Act.Dt.LongAvgDt * (nrn.ActM - nrn.ActAvg)
		nrn.SahpN, _ = ly.Act.Sahp.NinfTauFmCa(nrn.SahpCa)
		nrn.SahpCa = ly.Act.Sahp.CaInt(nrn.SahpCa, nrn.CaSpkD)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActP.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.ActP.UpdateVal(nrn.ActP, ni)
		}
		pl.ActP.CalcAvg()
	}
	ly.AxonLay.CorSimFmActs()
}

// TargToExt sets external input Ext from target values Targ
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) TargToExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Targ
			nrn.SetFlag(NeurHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Targ.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) ClearTargExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
			nrn.Ext = 0
			nrn.ClearFlag(NeurHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// SpkSt1 saves current activation state in SpkSt1 variables (using CaP)
func (ly *Layer) SpkSt1(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkSt1 = nrn.CaSpkP
	}
}

// SpkSt2 saves current activation state in SpkSt2 variables (using CaP)
func (ly *Layer) SpkSt2(ltime *Time) {
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
	ly.CorSim.Cor = cosv

	ly.Act.Dt.AvgVarUpdt(&ly.CorSim.Avg, &ly.CorSim.Var, ly.CorSim.Cor)
}

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == emer.Target
// Other Target layers include the TRCLayer in deep predictive learning.
// It is used in SynScale to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *Layer) IsTarget() bool {
	return ly.Typ == emer.Target
}

// IsInput returns true if this layer is an Input layer.
// By default, returns true for layers of Type == emer.Input
// Used to prevent adapting of inhibition or TrgAvg values.
func (ly *Layer) IsInput() bool {
	return ly.Typ == emer.Input
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

func (ly *Layer) IsLearnTrgAvg() bool {
	if ly.AxonLay.IsTarget() || ly.AxonLay.IsInput() || !ly.Learn.TrgAvgAct.On {
		return false
	}
	return true
}

// DTrgAvgFmErr computes change in TrgAvg based on unit-wise error signal
func (ly *Layer) DTrgAvgFmErr() {
	if !ly.IsLearnTrgAvg() {
		return
	}
	lr := ly.Learn.TrgAvgAct.ErrLrate
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

// DTrgSpkCaPubMean subtracts the mean from DTrgAvg values
// Called by TrgAvgFmD
func (ly *Layer) DTrgSpkCaPubMean() {
	submean := ly.Learn.TrgAvgAct.SubMean
	if submean == 0 {
		return
	}
	if ly.HasPoolInhib() && ly.Learn.TrgAvgAct.Pool {
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
func (ly *Layer) TrgAvgFmD() {
	if !ly.IsLearnTrgAvg() || ly.Learn.TrgAvgAct.ErrLrate == 0 {
		return
	}
	ly.DTrgSpkCaPubMean()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.TrgAvg = ly.Learn.TrgAvgAct.TrgRange.ClipVal(nrn.TrgAvg + nrn.DTrgAvg)
		nrn.DTrgAvg = 0
	}
}

// SynCa does cycle-level synaptic Ca updating for the Kinase learning mechanisms.
// Updates Ca, CaM, CaP, CaD cascaded at longer time scales, with CaP
// representing CaMKII LTP activity and CaD representing DAPK1 LTD activity.
func (ly *Layer) SynCa(ltime *Time) {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).SendSynCa(ltime)
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).RecvSynCa(ltime)
	}
}

// DWt computes the weight change (learning).
// calls DWt method on sending projections
func (ly *Layer) DWt(ltime *Time) {
	ly.DTrgAvgFmErr()
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).DWt(ltime)
	}
}

// DWtSubMean does mean subtraction for projections requiring it.
// calls DWtSubMean on the recv projections
func (ly *Layer) DWtSubMean(ltime *Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).DWtSubMean(ltime)
	}
}

// WtFmDWt updates the weights from delta-weight changes
// calls WtFmDWt on the sending projections
func (ly *Layer) WtFmDWt(ltime *Time) {
	ly.TrgAvgFmD()
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).WtFmDWt(ltime)
	}
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (ly *Layer) SlowAdapt(ltime *Time) {
	ly.GScaleAvgs(ltime)
	ly.AdaptInhib(ltime)
	ly.SynScale()
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).SlowAdapt(ltime)
	}
}

// GScaleAvgs updates conductance scale averages
func (ly *Layer) GScaleAvgs(ltime *Time) {
	var sum float32
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(AxonPrjn).AsAxon()
		sum += pj.GScale.AvgMax
	}
	if sum == 0 {
		return
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(AxonPrjn).AsAxon()
		pj.GScale.AvgMaxRel = pj.GScale.AvgMax / sum
	}
}

// AdaptInhib adapts inhibition
func (ly *Layer) AdaptInhib(ltime *Time) {
	if !ly.Inhib.ActAvg.AdaptGi || ly.AxonLay.IsInput() {
		return
	}
	ly.Inhib.ActAvg.Adapt(&ly.ActAvg.GiMult, ly.Inhib.ActAvg.Targ, ly.ActAvg.ActMAvg)
}

// SynScale performs synaptic scaling based on running average activation vs. targets
func (ly *Layer) SynScale() {
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
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), ni)
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
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), ni)
		}
		pl.AvgDif.CalcAvg()
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (ly *Layer) SynFail(ltime *Time) {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).SynFail(ltime)
	}
}

// LrateMod sets the Lrate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LrateSched).
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LrateMod(mod float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LrateMod(mod)
	}
}

// LrateSched sets the schedule-based learning rate multiplier.
// See also LrateMod.
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LrateSched(sched float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LrateSched(sched)
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
// Targ (Target or Compare types) or ActP does not match that of ActM.
// If Act > ly.Act.Clamp.ErrThr, effective activity = 1 else 0
// robust to noisy activations.
func (ly *Layer) PctUnitErr() float64 {
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	thr := ly.Act.Clamp.ErrThr
	wrong := 0
	n := 0
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		trg := false
		if ly.Typ == emer.Compare || ly.Typ == emer.Target {
			if nrn.Targ > thr {
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
		nrn.ClearFlag(NeurOff)
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
		nrn.SetFlag(NeurOff)
	}
	return nl
}

//////////////////////////////////////////////////////////////////////////////////////
//  Layer props for gui

var LayerProps = ki.Props{
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
