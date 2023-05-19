// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
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
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// index naming:
// lni = layer-based neuron index (0 = first neuron in layer)
// ni  = absolute whole network neuron index

// axon.Layer implements the basic Axon spiking activation function,
// and manages learning in the projections.
type Layer struct {
	LayerBase
	Params *LayerParams `desc:"all layer-level parameters -- these must remain constant once configured"`
	Vals   []LayerVals  `desc:"layer-level state values that are updated during computation -- one for each data parallel -- is a sub-slice of network full set"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// Object returns the object with parameters to be set by emer.Params
func (ly *Layer) Object() interface{} {
	return ly.Params
}

func (ly *Layer) Defaults() {
	if ly.Params != nil {
		ly.Params.LayType = ly.LayerType()
		ly.Params.Defaults()
		ly.Vals.ActAvg.GiMult = 1 // todo: for all
	}
	for _, pj := range ly.RcvPrjns { // must do prjn defaults first, then custom
		pj.Defaults()
	}
	if ly.Params == nil {
		return
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
		ly.Params.CTDefaults()
	case PTMaintLayer:
		ly.PTMaintDefaults()
	case PTPredLayer:
		ly.Params.PTPredDefaults()
	case PTNotMaintLayer:
		ly.PTNotMaintDefaults()
	case PulvinarLayer:
		ly.Params.PulvDefaults()

	case RewLayer:
		ly.Params.RWDefaults()
	case RWPredLayer:
		ly.Params.RWDefaults()
		ly.Params.RWPredDefaults()
	case RWDaLayer:
		ly.Params.RWDefaults()
	case TDPredLayer:
		ly.Params.TDDefaults()
		ly.Params.TDPredDefaults()
	case TDIntegLayer, TDDaLayer:
		ly.Params.TDDefaults()

	case LDTLayer:
		ly.LDTDefaults()
	case BLALayer:
		ly.BLADefaults()
	case CeMLayer:
		ly.CeMDefaults()
	case VSPatchLayer:
		ly.Params.VSPatchDefaults()
	case DrivesLayer:
		ly.Params.DrivesDefaults()
	case EffortLayer:
		ly.Params.EffortDefaults()
	case UrgencyLayer:
		ly.Params.UrgencyDefaults()
	case USLayer:
		ly.Params.USDefaults()
	case PVLayer:
		ly.Params.PVDefaults()

	case MatrixLayer:
		ly.MatrixDefaults()
	case GPLayer:
		ly.GPDefaults()
	case STNLayer:
		ly.STNDefaults()
	case BGThalLayer:
		ly.BGThalDefaults()
	case VSGatedLayer:
		ly.Params.VSGatedDefaults()
	}
	ly.ApplyDefParams()
	ly.UpdateParams()
}

// Update is an interface for generically updating after edits
// this should be used only for the values on the struct itself.
// UpdateParams is used to update all parameters, including Prjn.
func (ly *Layer) Update() {
	if ly.Params == nil {
		return
	}
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

// PostBuild performs special post-Build() configuration steps for specific algorithms,
// using configuration data set in BuildConfig during the ConfigNet process.
func (ly *Layer) PostBuild() {
	ly.Params.LayInhib.Idx1 = ly.BuildConfigFindLayer("LayInhib1Name", false) // optional
	ly.Params.LayInhib.Idx2 = ly.BuildConfigFindLayer("LayInhib2Name", false) // optional
	ly.Params.LayInhib.Idx3 = ly.BuildConfigFindLayer("LayInhib3Name", false) // optional
	ly.Params.LayInhib.Idx4 = ly.BuildConfigFindLayer("LayInhib4Name", false) // optional

	switch ly.LayerType() {
	case PulvinarLayer:
		ly.PulvPostBuild()

	case LDTLayer:
		ly.LDTPostBuild()
	case RWDaLayer:
		ly.RWDaPostBuild()
	case TDIntegLayer:
		ly.TDIntegPostBuild()
	case TDDaLayer:
		ly.TDDaPostBuild()

	case BLALayer:
		fallthrough
	case CeMLayer:
		fallthrough
	case USLayer:
		fallthrough
	case PVLayer:
		fallthrough
	case VSPatchLayer:
		ly.PVLVPostBuild()

	case MatrixLayer:
		ly.MatrixPostBuild()
	case GPLayer:
		ly.GPPostBuild()
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
	return NeuronVarNames
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
	return len(NeuronVarNames)
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= int(ly.NNeurons) {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= ly.UnitVarNum() {
		return mat32.NaN()
	}
	ctx := &ly.Network.Ctx
	if varIdx >= ly.UnitVarNum()-NNeuronLayerVars {
		lvi := varIdx - (ly.UnitVarNum() - NNeuronLayerVars)
		switch lvi {
		case 0:
			return ly.Vals[0].NeuroMod.DA
		case 1:
			return ly.Vals[0].NeuroMod.ACh
		case 2:
			return ly.Vals[0].NeuroMod.NE
		case 3:
			return ly.Vals[0].NeuroMod.Ser
		case 4:
			pl := ly.SubPool(ctx, uint32(idx), 0) // display uses data 0
			return float32(pl.Gated)
		}
	} else {
		return NrnV(ctx, uint32(idx)+ly.NeurStIdx, 0, NeuronVars(varIdx))
	}
	return mat32.NaN()
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitVals(vals *[]float32, varNm string) error {
	nn := ly.NNeurons
	if *vals == nil || cap(*vals) < int(nn) {
		*vals = make([]float32, nn)
	} else if len(*vals) < int(nn) {
		*vals = (*vals)[0:nn]
	}
	vidx, err := ly.UnitVarIdx(varNm)
	if err != nil {
		nan := mat32.NaN()
		for lni := uint32(0); lni < nn; lni++ {
			(*vals)[lni] = nan
		}
		return err
	}
	for lni := uint32(0); lni < nn; lni++ {
		(*vals)[lni] = ly.UnitVal1D(vidx, int(lni))
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
	nn := int(ly.NNeurons)
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	vidx, err := ly.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for lni := 0; lni < nn; lni++ {
			tsr.SetFloat1D(lni, nan)
		}
		return err
	}
	for lni := 0; lni < nn; lni++ {
		v := ly.UnitVal1D(vidx, lni)
		if mat32.IsNaN(v) {
			tsr.SetFloat1D(lni, math.NaN())
		} else {
			tsr.SetFloat1D(lni, float64(v))
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
	vidx, err := ly.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for i, _ := range ly.RepIxs {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i, ui := range ly.RepIxs {
		v := ly.UnitVal1D(vidx, ui)
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
	vidx, err := ly.UnitVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	fidx := ly.Shp.Offset(idx)
	return ly.UnitVal1D(vidx, fidx)
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
	nn := int(ly.NNeurons)
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
	nn := int(ly.NNeurons)
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
	w.Write([]byte(fmt.Sprintf("\"ActMAvg\": \"%g\",\n", ly.Vals[0].ActAvg.ActMAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActPAvg\": \"%g\",\n", ly.Vals[0].ActAvg.ActPAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GiMult\": \"%g\"\n", ly.Vals[0].ActAvg.GiMult)))
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
			nni := ly.NeurStIdx + lni
			nrnActAvg := NrnV(&ly.Network.Ctx, nni, 0, ActAvg)
			w.Write([]byte(fmt.Sprintf("%g", nrnActAvg)))
			if lni < nn-1 {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte(" ],\n"))

		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"TrgAvg\": [ ")))
		for lni := uint32(0); lni < nn; lni++ {
			nni := ly.NeurStIdx + lni
			nrnTrgAvg := NrnV(&ly.Network.Ctx, nni, 0, TrgAvg)
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
	ctx := &ly.Network.Ctx
	if lw.MetaData != nil {
		for di := uint32(0); di < ctx.NData; di++ {
			vals := &ly.Vals[di]
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
				ni := ly.NeurStIdx + uint32(lni)
				for di := uint32(0); di < ctx.NData; di++ {
					SetNrnV(ctx, ni, di, ActAvg, ta[lni])
				}
			}
		}
		if ta, ok := lw.Units["TrgAvg"]; ok {
			for lni := range ta {
				if lni > int(ly.NNeurons) {
					break
				}
				ni := ly.NeurStIdx + uint32(lni)
				for di := uint32(0); di < ctx.NData; di++ {
					SetNrnV(ctx, ni, di, TrgAvg, ta[lni])
				}
			}
		}
	}
	var err error
	if len(lw.Prjns) == ly.NRecvPrjns() { // this is essential if multiple prjns from same layer
		for pi := range lw.Prjns {
			pw := &lw.Prjns[pi]
			pj := ly.RcvPrjns[pi]
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

// note: all basic computation can be performed on layer-level and prjn level

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes the weight values in the network, i.e., resetting learning
// Also calls InitActs
func (ly *Layer) InitWts(ctx *Context, nt *Network) {
	ly.UpdateParams()
	ly.Params.Act.Dend.HasMod.SetBool(false)
	for di := uint32(0); di < ctx.NData; di++ {
		vals := &ly.Vals[di]
		vals.Init()
		vals.ActAvg.ActMAvg = ly.Params.Inhib.ActAvg.Nominal
		vals.ActAvg.ActPAvg = ly.Params.Inhib.ActAvg.Nominal
		ly.InitActAvg(ctx)
		ly.InitActs(ctx)
		ly.InitGScale(ctx)
	}
	for _, pj := range ly.SndPrjns {
		if pj.IsOff() {
			continue
		}
		pj.InitWts(ctx, nt)
	}
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		if pj.Params.Com.GType == ModulatoryG {
			ly.Params.Act.Dend.HasMod.SetBool(true)
			break
		}
	}

}

// InitActAvg initializes the running-average activation values
// that drive learning and the longer time averaging values.
func (ly *Layer) InitActAvg(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		nni := ly.NeurStIdx + lni
		for di := uint32(0); di < ctx.NData; di++ {
			ly.Params.Learn.InitNeurCa(ctx, nni, di)
		}
	}
	if ly.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		ly.InitActAvgPools(ctx)
	} else {
		ly.InitActAvgLayer(ctx)
	}
}

// InitActAvgLayer initializes the running-average activation values
// that drive learning and the longer time averaging values.
// version with just overall layer-level inhibition.
func (ly *Layer) InitActAvgLayer(ctx *Context) {
	strg := ly.Params.Learn.TrgAvgAct.TrgRange.Min
	rng := ly.Params.Learn.TrgAvgAct.TrgRange.Range()
	inc := float32(0)
	nn := ly.NNeurons
	if nn > 1 {
		inc = rng / float32(nn-1)
	}
	porder := make([]int, nn)
	for i := range porder {
		porder[i] = i
	}
	if ly.Params.Learn.TrgAvgAct.Permute.IsTrue() {
		erand.PermuteInts(porder, &ly.Network.Rand)
	}
	for lni := uint32(0); lni < nn; lni++ {
		nni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, nni) {
			continue
		}
		for di := uint32(0); di < ctx.NData; di++ {
			vi := porder[lni] // same for all datas
			trg := strg + inc*float32(vi)
			SetNrnV(ctx, nni, di, TrgAvg, trg)
			SetNrnV(ctx, nni, di, AvgPct, trg)
			SetNrnV(ctx, nni, di, ActAvg, ly.Params.Inhib.ActAvg.Nominal*trg)
			SetNrnV(ctx, nni, di, AvgDif, 0)
			SetNrnV(ctx, nni, di, DTrgAvg, 0)
		}
	}
}

// InitActAvgPools initializes the running-average activation values
// that drive learning and the longer time averaging values.
// version with pooled inhibition.
func (ly *Layer) InitActAvgPools(ctx *Context) {
	strg := ly.Params.Learn.TrgAvgAct.TrgRange.Min
	rng := ly.Params.Learn.TrgAvgAct.TrgRange.Range()
	inc := float32(0)
	nNy := ly.Shp.Dim(2)
	nNx := ly.Shp.Dim(3)
	nn := nNy * nNx
	if nn > 1 {
		inc = rng / float32(nn-1)
	}
	np := ly.NPools
	porder := make([]int, nn)
	for i := range porder {
		porder[i] = i
	}
	for pi := uint32(1); pi < np; pi++ {
		if ly.Params.Learn.TrgAvgAct.Permute.IsTrue() {
			erand.PermuteInts(porder, &ly.Network.Rand)
		}
		for di := uint32(0); di < ctx.NData; di++ {
			pl := ly.Pool(pi, di)
			for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
				nni := ly.NeurStIdx + lni
				if NrnIsOff(ctx, nni) {
					continue
				}
				vi := porder[lni-pl.StIdx]
				trg := strg + inc*float32(vi)
				SetNrnV(ctx, nni, di, TrgAvg, trg)
				SetNrnV(ctx, nni, di, AvgPct, trg)
				SetNrnV(ctx, nni, di, ActAvg, ly.Params.Inhib.ActAvg.Nominal*trg)
				SetNrnV(ctx, nni, di, AvgDif, 0)
				SetNrnV(ctx, nni, di, DTrgAvg, 0)
			}
		}
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs(ctx *Context) {
	ly.Params.Act.Clamp.IsInput.SetBool(ly.Params.IsInput())
	ly.Params.Act.Clamp.IsTarget.SetBool(ly.Params.IsTarget())
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		nni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, nni) {
			continue
		}
		for di := uint32(0); di < ctx.NData; di++ {
			ly.Params.Act.InitActs(ctx, nni, di, &ly.Network.Rand)
		}
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Init()
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsInput.IsTrue() {
			pl.Inhib.Clamped.SetBool(true)
		}
		// Target layers are dynamically updated
	}
	ly.InitPrjnGBuffs(ctx)
}

// InitPrjnGBuffs initializes the projection-level conductance buffers and
// conductance integration values for receiving projections in this layer.
func (ly *Layer) InitPrjnGBuffs(ctx *Context) {
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		pj.InitGBuffs()
	}
}

// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
func (ly *Layer) InitWtSym(ctx *Context) {
	for _, pj := range ly.SndPrjns {
		if pj.IsOff() {
			continue
		}
		if pj.Params.SWt.Init.Sym.IsFalse() {
			continue
		}
		// key ordering constraint on which way weights are copied
		if pj.Recv.Index() < pj.Send.Index() {
			continue
		}
		rpj, has := ly.RecipToSendPrjn(pj)
		if !has {
			continue
		}
		if rpj.Params.SWt.Init.Sym.IsFalse() {
			continue
		}
		pj.InitWtSym(rpj)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  ApplyExt

// InitExt initializes external input state.
// Should be called prior to ApplyExt on all layers receiving Ext input.
func (ly *Layer) InitExt(ctx *Context) {
	if !ly.LayerType().IsExt() {
		return
	}
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		if NrnIsOff(ctx, lni) {
			continue
		}
		for di := uint32(0); di < ctx.NData; di++ {
			ly.Params.InitExt(ctx, lni, di)
			ly.Exts[ly.ExtIdx(lni, di)] = -1 // missing by default
		}
	}
}

// ApplyExt applies external input in the form of an etensor.Float32 or 64.
// Negative values and NaNs are not valid, and will be interpreted as missing inputs.
// The given data index di is the data parallel index (0 < di < MaxData):
// must present inputs separately for each separate data parallel set.
// If dimensionality of tensor matches that of layer, and is 2D or 4D,
// then each dimension is iterated separately, so any mismatch preserves
// dimensional structure.
// Otherwise, the flat 1D view of the tensor is used.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext.
// Also sets the Exts values on layer, which are used for the GPU version,
// which requires calling the network ApplyExts() method -- is a no-op for CPU.
func (ly *Layer) ApplyExt(ctx *Context, di uint32, ext etensor.Tensor) {
	switch {
	case ext.NumDims() == 2 && ly.Shp.NumDims() == 4: // special case
		ly.ApplyExt2Dto4D(ctx, di, ext)
	case ext.NumDims() != ly.Shp.NumDims() || !(ext.NumDims() == 2 || ext.NumDims() == 4):
		ly.ApplyExt1DTsr(ctx, di, ext)
	case ext.NumDims() == 2:
		ly.ApplyExt2D(ctx, di, ext)
	case ext.NumDims() == 4:
		ly.ApplyExt4D(ctx, di, ext)
	}
}

// ApplyExtVal applies given external value to given neuron
// using clearMask, setMask, and toTarg from ApplyExtFlags.
// Also saves Val in Exts for potential use by GPU.
func (ly *Layer) ApplyExtVal(ctx *Context, lni, di uint32, val float32, clearMask, setMask NeuronFlags, toTarg bool) {
	ei := ly.ExtIdx(lni, di)
	if uint32(len(ly.Exts)) <= ei {
		log.Printf("Layer named: %s Type: %s does not have allocated Exts vals -- is likely not registered to receive external input in LayerTypes.IsExt() -- will not be presented to GPU", ly.Name(), ly.LayerType().String())
	} else {
		ly.Exts[ei] = val
	}
	if val < 0 {
		return
	}
	if toTarg {
		SetNrnV(ctx, lni, di, Target, val)
	} else {
		SetNrnV(ctx, lni, di, Ext, val)
	}
	NrnClearFlag(ctx, lni, clearMask)
	NrnSetFlag(ctx, lni, setMask)
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Target (else Ext)
func (ly *Layer) ApplyExtFlags() (clearMask, setMask NeuronFlags, toTarg bool) {
	ly.Params.ApplyExtFlags(&clearMask, &setMask, &toTarg)
	return
}

// ApplyExt2D applies 2D tensor external input
func (ly *Layer) ApplyExt2D(ctx *Context, di uint32, ext etensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	ymx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			val := float32(ext.FloatVal(idx))
			lni := uint32(ly.Shp.Offset(idx))
			if NrnIsOff(ctx, lni) {
				continue
			}
			ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
		}
	}
}

// ApplyExt2Dto4D applies 2D tensor external input to a 4D layer
func (ly *Layer) ApplyExt2Dto4D(ctx *Context, di uint32, ext etensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	lNy, lNx, _, _ := etensor.Prjn2DShape(&ly.Shp, false)

	ymx := ints.MinInt(ext.Dim(0), lNy)
	xmx := ints.MinInt(ext.Dim(1), lNx)
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			val := float32(ext.FloatVal(idx))
			lni := uint32(etensor.Prjn2DIdx(&ly.Shp, false, y, x))
			if NrnIsOff(ctx, lni) {
				continue
			}
			ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
		}
	}
}

// ApplyExt4D applies 4D tensor external input
func (ly *Layer) ApplyExt4D(ctx *Context, di uint32, ext etensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	ypmx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xpmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	ynmx := ints.MinInt(ext.Dim(2), ly.Shp.Dim(2))
	xnmx := ints.MinInt(ext.Dim(3), ly.Shp.Dim(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					val := float32(ext.FloatVal(idx))
					lni := uint32(ly.Shp.Offset(idx))
					if NrnIsOff(ctx, lni) {
						continue
					}
					ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1DTsr(ctx *Context, di uint32, ext etensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(ints.MinInt(ext.Len(), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		if NrnIsOff(ctx, lni) {
			continue
		}
		val := float32(ext.FloatVal1D(int(lni)))
		ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ctx *Context, di uint32, ext []float64) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(ints.MinInt(len(ext), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		if NrnIsOff(ctx, lni) {
			continue
		}
		val := float32(ext[lni])
		ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ctx *Context, di uint32, ext []float32) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(ints.MinInt(len(ext), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		if NrnIsOff(ctx, lni) {
			continue
		}
		val := ext[lni]
		ly.ApplyExtVal(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (ly *Layer) UpdateExtFlags(ctx *Context) {
	clearMask, setMask, _ := ly.ApplyExtFlags()
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		if NrnIsOff(ctx, lni) {
			continue
		}
		NrnClearFlag(ctx, lni, clearMask)
		NrnSetFlag(ctx, lni, setMask)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  InitGScale

// InitGScale computes the initial scaling factor for synaptic input conductances G,
// stored in GScale.Scale, based on sending layer initial activation.
func (ly *Layer) InitGScale(ctx *Context) {
	totGeRel := float32(0)
	totGiRel := float32(0)
	totGmRel := float32(0)
	totGmnRel := float32(0)
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		slay := pj.Send
		savg := slay.Params.Inhib.ActAvg.Nominal
		snu := slay.NNeurons
		ncon := pj.RecvConNAvgMax.Avg
		pj.Params.GScale.Scale = pj.Params.PrjnScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a prjn, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.Params.GScale == 0 {
		// 	continue
		// }
		switch pj.Params.Com.GType {
		case InhibitoryG:
			totGiRel += pj.Params.PrjnScale.Rel
		case ModulatoryG:
			totGmRel += pj.Params.PrjnScale.Rel
		case MaintG:
			totGmnRel += pj.Params.PrjnScale.Rel
		default:
			totGeRel += pj.Params.PrjnScale.Rel
		}
	}

	for _, pj := range ly.RcvPrjns {
		switch pj.Params.Com.GType {
		case InhibitoryG:
			if totGiRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PrjnScale.Rel / totGiRel
				pj.Params.GScale.Scale /= totGiRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0
			}
		case ModulatoryG:
			if totGmRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PrjnScale.Rel / totGmRel
				pj.Params.GScale.Scale /= totGmRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0

			}
		case MaintG:
			if totGmnRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PrjnScale.Rel / totGmnRel
				pj.Params.GScale.Scale /= totGmnRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0

			}
		default:
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
	neur = int(ly.NNeurons) * perNeur
	syn = 0
	for _, pj := range ly.SndPrjns {
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
// returns one result per data parallel index ([ctx.NData])
func (ly *Layer) PctUnitErr(ctx *Context) []float64 {
	nn := ly.NNeurons
	if nn == 0 {
		return nil
	}
	errs := make([]float64, ctx.NData)
	thr := ly.Params.Act.Clamp.ErrThr
	for di := uint32(0); di < ctx.NData; di++ {
		wrong := 0
		n := 0
		for lni := uint32(0); lni < nn; lni++ {
			if NrnIsOff(ctx, lni) {
				continue
			}
			trg := false
			if ly.Typ == CompareLayer || ly.Typ == TargetLayer {
				if NrnV(ctx, lni, di, Target) > thr {
					trg = true
				}
			} else {
				if NrnV(ctx, lni, di, ActP) > thr {
					trg = true
				}
			}
			if NrnV(ctx, lni, di, ActM) > thr {
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
			errs[di] = float64(wrong) / float64(n)
		}
	}
	return errs
}

// LocalistErr2D decodes a 2D layer with Y axis = redundant units, X = localist units
// returning the indexes of the max activated localist value in the minus and plus phase
// activities, and whether these are the same or different (err = different)
// returns one result per data parallel index ([ctx.NData])
func (ly *Layer) LocalistErr2D(ctx *Context) (err []bool, minusIdx, plusIdx []int) {
	err = make([]bool, ctx.NData)
	minusIdx = make([]int, ctx.NData)
	plusIdx = make([]int, ctx.NData)
	ydim := ly.Shp.Dim(0)
	xdim := ly.Shp.Dim(1)
	for di := uint32(0); di < ctx.NData; di++ {
		var maxM, maxP float32
		var mIdx, pIdx int
		for xi := 0; xi < xdim; xi++ {
			var sumP, sumM float32
			for yi := 0; yi < ydim; yi++ {
				lni := uint32(yi*xdim + xi)
				sumM += NrnV(ctx, lni, di, ActM)
				sumP += NrnV(ctx, lni, di, ActP)
			}
			if sumM > maxM {
				mIdx = xi
				maxM = sumM
			}
			if sumP > maxP {
				pIdx = xi
				maxP = sumP
			}
		}
		er := mIdx != pIdx
		err[di] = er
		minusIdx[di] = mIdx
		plusIdx[di] = pIdx
	}
	return
}

// LocalistErr4D decodes a 4D layer with each pool representing a localist value.
// Returns the flat 1D indexes of the max activated localist value in the minus and plus phase
// activities, and whether these are the same or different (err = different)
func (ly *Layer) LocalistErr4D(ctx *Context) (err []bool, minusIdx, plusIdx []int) {
	err = make([]bool, ctx.NData)
	minusIdx = make([]int, ctx.NData)
	plusIdx = make([]int, ctx.NData)
	npool := ly.Shp.Dim(0) * ly.Shp.Dim(1)
	nun := ly.Shp.Dim(2) * ly.Shp.Dim(3)
	for di := uint32(0); di < ctx.NData; di++ {
		var maxM, maxP float32
		var mIdx, pIdx int
		for xi := 0; xi < npool; xi++ {
			var sumP, sumM float32
			for yi := 0; yi < nun; yi++ {
				lni := uint32(xi*nun + yi)
				sumM += NrnV(ctx, lni, di, ActM)
				sumP += NrnV(ctx, lni, di, ActP)
			}
			if sumM > maxM {
				mIdx = xi
				maxM = sumM
			}
			if sumP > maxP {
				pIdx = xi
				maxP = sumP
			}
		}
		er := mIdx != pIdx
		err[di] = er
		minusIdx[di] = mIdx
		plusIdx[di] = pIdx
	}
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		NrnClearFlag(ctx, lni, NeuronOff)
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(ctx *Context, prop float32) int {
	ly.UnLesionNeurons(ctx)
	if prop > 1 {
		log.Printf("LesionNeurons got a proportion > 1 -- must be 0-1 as *proportion* (not percent) of neurons to lesion: %v\n", prop)
		return 0
	}
	nn := ly.NNeurons
	if nn == 0 {
		return 0
	}
	p := rand.Perm(int(nn))
	nl := int(prop * float32(nn))
	for lni := uint32(0); lni < nn; lni++ {
		nip := uint32(p[lni])
		if NrnIsOff(ctx, nip) {
			continue
		}
		NrnSetFlag(ctx, nip, NeuronOff)
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
