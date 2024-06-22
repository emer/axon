// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"log"
	"math/rand"
	"strings"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/params"
)

// index naming:
// lni = layer-based neuron index (0 = first neuron in layer)
// ni  = absolute whole network neuron index

// axon.Layer implements the basic Axon spiking activation function,
// and manages learning in the pathways.
type Layer struct {
	LayerBase

	// all layer-level parameters -- these must remain constant once configured
	Params *LayerParams
}

// Object returns the object with parameters to be set by emer.Params
func (ly *Layer) Object() any {
	return ly.Params
}

func (ly *Layer) Defaults() { //types:add
	ctx := &ly.Network.Ctx
	if ly.Params != nil {
		ly.Params.LayType = ly.LayerType()
		ly.Params.Defaults()
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Values[di].ActAvg.GiMult = 1
		}
		ly.Params.Learn.CaLearn.Dt.PDTauForNCycles(int(ctx.ThetaCycles))
		ly.Params.Learn.CaSpk.Dt.PDTauForNCycles(int(ctx.ThetaCycles))
	}
	for _, pj := range ly.RcvPaths { // must do path defaults first, then custom
		pj.Defaults()
	}
	if ly.Params == nil {
		return
	}
	switch ly.LayerType() {
	case InputLayer:
		ly.Params.Acts.Clamp.Ge = 1.5
		ly.Params.Inhib.Layer.Gi = 0.9
		ly.Params.Inhib.Pool.Gi = 0.9
		ly.Params.Learn.TrgAvgAct.SubMean = 0
	case TargetLayer:
		ly.Params.Acts.Clamp.Ge = 0.8
		ly.Params.Learn.TrgAvgAct.SubMean = 0
		// ly.Params.Learn.RLRate.SigmoidMin = 1

	case CTLayer:
		ly.Params.CTDefaults()
	case PTMaintLayer:
		ly.PTMaintDefaults()
	case PTPredLayer:
		ly.Params.PTPredDefaults()
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
	ly.Params.CT.DecayForNCycles(int(ctx.ThetaCycles))
	ly.ApplyDefParams()
	ly.UpdateParams()
}

// Update is an interface for generically updating after edits
// this should be used only for the values on the struct itself.
// UpdateParams is used to update all parameters, including Path.
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
// receiving pathways of this layer.
// This is not called Update because it is not just about the
// local values in the struct.
func (ly *Layer) UpdateParams() {
	ly.Update()
	for _, pj := range ly.RcvPaths {
		pj.UpdateParams()
	}
}

// SetParam sets parameter at given path to given value.
// returns error if path not found or value cannot be set.
func (ly *Layer) SetParam(path, val string) error {
	return params.SetParam(ly.Params, path, val)
}

// PostBuild performs special post-Build() configuration steps for specific algorithms,
// using configuration data set in BuildConfig during the ConfigNet process.
func (ly *Layer) PostBuild() {
	ly.Params.LayInhib.Index1 = ly.BuildConfigFindLayer("LayInhib1Name", false) // optional
	ly.Params.LayInhib.Index2 = ly.BuildConfigFindLayer("LayInhib2Name", false) // optional
	ly.Params.LayInhib.Index3 = ly.BuildConfigFindLayer("LayInhib3Name", false) // optional
	ly.Params.LayInhib.Index4 = ly.BuildConfigFindLayer("LayInhib4Name", false) // optional

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
		ly.RubiconPostBuild()

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
	for _, pj := range ly.RcvPaths {
		str += pj.AllParams()
	}
	return str
}

// note: all basic computation can be performed on layer-level and path level

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes the weight values in the network, i.e., resetting learning
// Also calls InitActs
func (ly *Layer) InitWts(ctx *Context, nt *Network) { //types:add
	ly.UpdateParams()
	ly.Params.Acts.Dend.HasMod.SetBool(false)
	for di := uint32(0); di < ly.MaxData; di++ {
		vals := &ly.Values[di]
		vals.Init()
		vals.ActAvg.ActMAvg = ly.Params.Inhib.ActAvg.Nominal
		vals.ActAvg.ActPAvg = ly.Params.Inhib.ActAvg.Nominal
	}
	ly.InitActAvg(ctx)
	ly.InitActs(ctx)
	ly.InitGScale(ctx)
	for _, pj := range ly.SndPaths {
		if pj.IsOff() {
			continue
		}
		pj.InitWts(ctx, nt)
	}
	for _, pj := range ly.RcvPaths {
		if pj.IsOff() {
			continue
		}
		if pj.Params.Com.GType == ModulatoryG {
			ly.Params.Acts.Dend.HasMod.SetBool(true)
			break
		}
	}

}

// InitActAvg initializes the running-average activation values
// that drive learning and the longer time averaging values.
func (ly *Layer) InitActAvg(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Params.Learn.InitNeurCa(ctx, ni, di)
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
	tmax := ly.Params.Learn.TrgAvgAct.TrgRange.Max
	gibinit := ly.Params.Learn.TrgAvgAct.GiBaseInit
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
		randx.PermuteInts(porder, &ly.Network.Rand)
	}
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		vi := porder[lni] // same for all datas
		trg := strg + inc*float32(vi)
		SetNrnAvgV(ctx, ni, TrgAvg, trg)
		SetNrnAvgV(ctx, ni, AvgPct, trg)
		SetNrnAvgV(ctx, ni, ActAvg, ly.Params.Inhib.ActAvg.Nominal*trg)
		SetNrnAvgV(ctx, ni, AvgDif, 0)
		SetNrnAvgV(ctx, ni, DTrgAvg, 0)
		SetNrnAvgV(ctx, ni, GeBase, ly.Params.Acts.Init.GetGeBase(&ly.Network.Rand))
		SetNrnAvgV(ctx, ni, GiBase, ly.Params.Acts.Init.GetGiBase(&ly.Network.Rand))
		if gibinit > 0 {
			gib := gibinit * (tmax - trg)
			SetNrnAvgV(ctx, ni, GiBase, gib)
		}
	}
}

// InitActAvgPools initializes the running-average activation values
// that drive learning and the longer time averaging values.
// version with pooled inhibition.
func (ly *Layer) InitActAvgPools(ctx *Context) {
	strg := ly.Params.Learn.TrgAvgAct.TrgRange.Min
	rng := ly.Params.Learn.TrgAvgAct.TrgRange.Range()
	tmax := ly.Params.Learn.TrgAvgAct.TrgRange.Max
	gibinit := ly.Params.Learn.TrgAvgAct.GiBaseInit
	inc := float32(0)
	nNy := ly.Shp.DimSize(2)
	nNx := ly.Shp.DimSize(3)
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
			randx.PermuteInts(porder, &ly.Network.Rand)
		}
		pl := ly.Pool(pi, 0) // only using for idxs
		for lni := pl.StIndex; lni < pl.EdIndex; lni++ {
			ni := ly.NeurStIndex + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			vi := porder[lni-pl.StIndex]
			trg := strg + inc*float32(vi)
			SetNrnAvgV(ctx, ni, TrgAvg, trg)
			SetNrnAvgV(ctx, ni, AvgPct, trg)
			SetNrnAvgV(ctx, ni, ActAvg, ly.Params.Inhib.ActAvg.Nominal*trg)
			SetNrnAvgV(ctx, ni, AvgDif, 0)
			SetNrnAvgV(ctx, ni, DTrgAvg, 0)
			SetNrnAvgV(ctx, ni, GeBase, ly.Params.Acts.Init.GetGeBase(&ly.Network.Rand))
			SetNrnAvgV(ctx, ni, GiBase, ly.Params.Acts.Init.GetGiBase(&ly.Network.Rand))
			if gibinit > 0 {
				gib := gibinit * (tmax - trg)
				SetNrnAvgV(ctx, ni, GiBase, gib)
			}
		}
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs(ctx *Context) { //types:add
	ly.Params.Acts.Clamp.IsInput.SetBool(ly.Params.IsInput())
	ly.Params.Acts.Clamp.IsTarget.SetBool(ly.Params.IsTarget())
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Params.Acts.InitActs(ctx, ni, di)
		}
	}
	np := ly.NPools
	for pi := uint32(0); pi < np; pi++ {
		for di := uint32(0); di < ly.MaxData; di++ {
			pl := ly.Pool(pi, di)
			pl.Init()
			if ly.Params.Acts.Clamp.Add.IsFalse() && ly.Params.Acts.Clamp.IsInput.IsTrue() {
				pl.Inhib.Clamped.SetBool(true)
			}
			// Target layers are dynamically updated
		}
	}
	ly.InitPathGBuffs(ctx)
}

// InitPathGBuffs initializes the pathway-level conductance buffers and
// conductance integration values for receiving pathways in this layer.
func (ly *Layer) InitPathGBuffs(ctx *Context) {
	for _, pj := range ly.RcvPaths {
		if pj.IsOff() {
			continue
		}
		pj.InitGBuffs()
	}
}

// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
func (ly *Layer) InitWtSym(ctx *Context) {
	for _, pj := range ly.SndPaths {
		if pj.IsOff() {
			continue
		}
		if pj.Params.SWts.Init.Sym.IsFalse() {
			continue
		}
		// key ordering constraint on which way weights are copied
		if pj.Recv.Index() < pj.Send.Index() {
			continue
		}
		rpj, has := ly.RecipToSendPath(pj)
		if !has {
			continue
		}
		if rpj.Params.SWts.Init.Sym.IsFalse() {
			continue
		}
		pj.InitWtSym(ctx, rpj)
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
		ni := ly.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ly.MaxData; di++ {
			ly.Params.InitExt(ctx, ni, di)
			ei := ly.Params.Indexes.ExtIndex(lni, di)
			ly.Exts[ei] = -1 // missing by default
		}
	}
}

// ApplyExt applies external input in the form of an tensor.Float32 or 64.
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
func (ly *Layer) ApplyExt(ctx *Context, di uint32, ext tensor.Tensor) {
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
func (ly *Layer) ApplyExtValue(ctx *Context, lni, di uint32, val float32, clearMask, setMask NeuronFlags, toTarg bool) {
	ni := ly.NeurStIndex + lni
	if NrnIsOff(ctx, ni) {
		return
	}
	ei := ly.Params.Indexes.ExtIndex(lni, di)
	if uint32(len(ly.Exts)) <= ei {
		log.Printf("Layer named: %s Type: %s does not have allocated Exts vals -- is likely not registered to receive external input in LayerTypes.IsExt() -- will not be presented to GPU", ly.Name(), ly.LayerType().String())
	} else {
		ly.Exts[ei] = val
	}
	if val < 0 {
		return
	}
	if toTarg {
		SetNrnV(ctx, ni, di, Target, val)
	} else {
		SetNrnV(ctx, ni, di, Ext, val)
	}
	NrnClearFlag(ctx, ni, di, clearMask)
	NrnSetFlag(ctx, ni, di, setMask)
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Target (else Ext)
func (ly *Layer) ApplyExtFlags() (clearMask, setMask NeuronFlags, toTarg bool) {
	ly.Params.ApplyExtFlags(&clearMask, &setMask, &toTarg)
	return
}

// ApplyExt2D applies 2D tensor external input
func (ly *Layer) ApplyExt2D(ctx *Context, di uint32, ext tensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	ymx := min(ext.DimSize(0), ly.Shp.DimSize(0))
	xmx := min(ext.DimSize(1), ly.Shp.DimSize(1))
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			val := float32(ext.Float(idx))
			lni := uint32(ly.Shp.Offset(idx))
			ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
		}
	}
}

// ApplyExt2Dto4D applies 2D tensor external input to a 4D layer
func (ly *Layer) ApplyExt2Dto4D(ctx *Context, di uint32, ext tensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	lNy, lNx, _, _ := tensor.Projection2DShape(&ly.Shp, false)

	ymx := min(ext.DimSize(0), lNy)
	xmx := min(ext.DimSize(1), lNx)
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			val := float32(ext.Float(idx))
			lni := uint32(tensor.Projection2DIndex(&ly.Shp, false, y, x))
			ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
		}
	}
}

// ApplyExt4D applies 4D tensor external input
func (ly *Layer) ApplyExt4D(ctx *Context, di uint32, ext tensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	ypmx := min(ext.DimSize(0), ly.Shp.DimSize(0))
	xpmx := min(ext.DimSize(1), ly.Shp.DimSize(1))
	ynmx := min(ext.DimSize(2), ly.Shp.DimSize(2))
	xnmx := min(ext.DimSize(3), ly.Shp.DimSize(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					val := float32(ext.Float(idx))
					lni := uint32(ly.Shp.Offset(idx))
					ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1DTsr(ctx *Context, di uint32, ext tensor.Tensor) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(min(ext.Len(), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		val := float32(ext.Float1D(int(lni)))
		ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ctx *Context, di uint32, ext []float64) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(min(len(ext), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		val := float32(ext[lni])
		ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Target
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ctx *Context, di uint32, ext []float32) {
	clearMask, setMask, toTarg := ly.ApplyExtFlags()
	mx := uint32(min(len(ext), int(ly.NNeurons)))
	for lni := uint32(0); lni < mx; lni++ {
		val := ext[lni]
		ly.ApplyExtValue(ctx, lni, di, val, clearMask, setMask, toTarg)
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (ly *Layer) UpdateExtFlags(ctx *Context) {
	clearMask, setMask, _ := ly.ApplyExtFlags()
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
			NrnClearFlag(ctx, ni, di, clearMask)
			NrnSetFlag(ctx, ni, di, setMask)
		}
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
	for _, pj := range ly.RcvPaths {
		if pj.IsOff() {
			continue
		}
		slay := pj.Send
		savg := slay.Params.Inhib.ActAvg.Nominal
		snu := slay.NNeurons
		ncon := pj.RecvConNAvgMax.Avg
		pj.Params.GScale.Scale = pj.Params.PathScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a path, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.Params.GScale == 0 {
		// 	continue
		// }
		switch pj.Params.Com.GType {
		case InhibitoryG:
			totGiRel += pj.Params.PathScale.Rel
		case ModulatoryG:
			totGmRel += pj.Params.PathScale.Rel
		case MaintG:
			totGmnRel += pj.Params.PathScale.Rel
		default:
			totGeRel += pj.Params.PathScale.Rel
		}
	}

	for _, pj := range ly.RcvPaths {
		switch pj.Params.Com.GType {
		case InhibitoryG:
			if totGiRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PathScale.Rel / totGiRel
				pj.Params.GScale.Scale /= totGiRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0
			}
		case ModulatoryG:
			if totGmRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PathScale.Rel / totGmRel
				pj.Params.GScale.Scale /= totGmRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0

			}
		case MaintG:
			if totGmnRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PathScale.Rel / totGmnRel
				pj.Params.GScale.Scale /= totGmnRel
			} else {
				pj.Params.GScale.Rel = 0
				pj.Params.GScale.Scale = 0

			}
		default:
			if totGeRel > 0 {
				pj.Params.GScale.Rel = pj.Params.PathScale.Rel / totGeRel
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
	for _, pj := range ly.SndPaths {
		syn += int(pj.NSyns)
	}
	tot = neur + syn
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Stats

// note: use float64 for stats as that is best for logging

// PctUnitErr returns the proportion of units where the thresholded value of
// Target (Target or Compare types) or ActP does not match that of ActM.
// If Act > ly.Params.Acts.Clamp.ErrThr, effective activity = 1 else 0
// robust to noisy activations.
// returns one result per data parallel index ([ctx.NetIndexes.NData])
func (ly *Layer) PctUnitErr(ctx *Context) []float64 {
	nn := ly.NNeurons
	if nn == 0 {
		return nil
	}
	errs := make([]float64, ctx.NetIndexes.NData)
	thr := ly.Params.Acts.Clamp.ErrThr
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		wrong := 0
		n := 0
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			trg := false
			if ly.Typ == CompareLayer || ly.Typ == TargetLayer {
				if NrnV(ctx, ni, di, Target) > thr {
					trg = true
				}
			} else {
				if NrnV(ctx, ni, di, ActP) > thr {
					trg = true
				}
			}
			if NrnV(ctx, ni, di, ActM) > thr {
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
// returns one result per data parallel index ([ctx.NetIndexes.NData])
func (ly *Layer) LocalistErr2D(ctx *Context) (err []bool, minusIndex, plusIndex []int) {
	err = make([]bool, ctx.NetIndexes.NData)
	minusIndex = make([]int, ctx.NetIndexes.NData)
	plusIndex = make([]int, ctx.NetIndexes.NData)
	ydim := ly.Shp.DimSize(0)
	xdim := ly.Shp.DimSize(1)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		var maxM, maxP float32
		var mIndex, pIndex int
		for xi := 0; xi < xdim; xi++ {
			var sumP, sumM float32
			for yi := 0; yi < ydim; yi++ {
				lni := uint32(yi*xdim + xi)
				ni := ly.NeurStIndex + lni
				sumM += NrnV(ctx, ni, di, ActM)
				sumP += NrnV(ctx, ni, di, ActP)
			}
			if sumM > maxM {
				mIndex = xi
				maxM = sumM
			}
			if sumP > maxP {
				pIndex = xi
				maxP = sumP
			}
		}
		er := mIndex != pIndex
		err[di] = er
		minusIndex[di] = mIndex
		plusIndex[di] = pIndex
	}
	return
}

// LocalistErr4D decodes a 4D layer with each pool representing a localist value.
// Returns the flat 1D indexes of the max activated localist value in the minus and plus phase
// activities, and whether these are the same or different (err = different)
func (ly *Layer) LocalistErr4D(ctx *Context) (err []bool, minusIndex, plusIndex []int) {
	err = make([]bool, ctx.NetIndexes.NData)
	minusIndex = make([]int, ctx.NetIndexes.NData)
	plusIndex = make([]int, ctx.NetIndexes.NData)
	npool := ly.Shp.DimSize(0) * ly.Shp.DimSize(1)
	nun := ly.Shp.DimSize(2) * ly.Shp.DimSize(3)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		var maxM, maxP float32
		var mIndex, pIndex int
		for xi := 0; xi < npool; xi++ {
			var sumP, sumM float32
			for yi := 0; yi < nun; yi++ {
				lni := uint32(xi*nun + yi)
				ni := ly.NeurStIndex + lni
				sumM += NrnV(ctx, ni, di, ActM)
				sumP += NrnV(ctx, ni, di, ActP)
			}
			if sumM > maxM {
				mIndex = xi
				maxM = sumM
			}
			if sumP > maxP {
				pIndex = xi
				maxP = sumP
			}
		}
		er := mIndex != pIndex
		err[di] = er
		minusIndex[di] = mIndex
		plusIndex[di] = pIndex
	}
	return
}

// TestValues returns a map of key vals for testing
// ctrKey is a key of counters to contextualize values.
func (ly *Layer) TestValues(ctrKey string, vals map[string]float32) {
	for pi := uint32(0); pi < ly.NPools; pi++ {
		for di := uint32(0); di < ly.MaxData; di++ {
			pl := ly.Pool(pi, di)
			key := fmt.Sprintf("%s  Lay: %s\tPool: %d\tDi: %d", ctrKey, ly.Nm, pi, di)
			pl.TestValues(key, vals)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons() { //types:add
	ctx := &ly.Network.Ctx
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		for di := uint32(0); di < ly.MaxData; di++ {
			NrnClearFlag(ctx, ni, di, NeuronOff)
		}
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(prop float32) int { //types:add
	ctx := &ly.Network.Ctx
	ly.UnLesionNeurons()
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
		ni := ly.NeurStIndex + nip
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ly.MaxData; di++ {
			NrnSetFlag(ctx, ni, di, NeuronOff)
		}
	}
	return nl
}

func (ly *Layer) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(ly.Defaults).SetIcon(icons.Reset)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(ly.InitWts).SetIcon(icons.Reset)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(ly.InitActs).SetIcon(icons.Reset)
	})
	tree.Add(p, func(w *core.Separator) {})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(ly.LesionNeurons).SetIcon(icons.Cut)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(ly.UnLesionNeurons).SetIcon(icons.Cut)
	})
}
