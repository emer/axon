// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
)

// TRNLayer copies inhibition from pools in CT and Pulv layers, and from other
// TRNLayers, and pools this inhibition using the Max operation
type TRNLayer struct {
	axon.Layer
	ILayers emer.LayNames `desc:"layers that we receive inhibition from"`
}

var KiT_TRNLayer = kit.Types.AddType(&TRNLayer{}, LayerProps)

func (ly *TRNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = TRN
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *TRNLayer) InitActs() {
	ly.Layer.InitActs()
}

///////////////////////////////////////////////////////////////////////////////////////
// PulvALayer -- attention Pulv

// SendAttnParams parameters for sending attention
type SendAttnParams struct {
	Thr    float32       `desc:"threshold on layer-wide max activation (or average act for pooled 4D layers) for sending attention (below this, sends attn = 1)"`
	ToLays emer.LayNames `desc:"list of layers to send attentional modulation to"`
}

func (ti *SendAttnParams) Defaults() {
	ti.Thr = 0.1
}

func (ti *SendAttnParams) Update() {
}

// PulvAttnLayer is the thalamic relay cell layer for Attention in DeepAxon.
type PulvAttnLayer struct {
	axon.Layer                // access as .Layer
	SendAttn   SendAttnParams `view:"inline" desc:"sending attention parameters"`
}

var KiT_PulvAttnLayer = kit.Types.AddType(&PulvAttnLayer{}, LayerProps)

func (ly *PulvAttnLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0.5
	ly.Act.Decay.Glong = 1
	ly.Act.Decay.AHP = 0
	ly.Act.GABAB.Gbar = 0.005 // output layer settings
	ly.Act.NMDA.Gbar = 0.01
	ly.SendAttn.Defaults()
	ly.Typ = Pulv
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *PulvAttnLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.SendAttn.Update()
}

func (ly *PulvAttnLayer) Class() string {
	return "PulvA " + ly.Cls
}

func (ly *PulvAttnLayer) IsTarget() bool {
	return false // We are not
}

// CyclePost is called at end of Cycle
// We use it to send Attn
func (ly *PulvAttnLayer) CyclePost(ctime *axon.Time) {
	ly.AttnFmAct(ctime)
	ly.SendAttnLays(ctime)
}

// AttnFmAct computes our attention signal from activations
func (ly *PulvAttnLayer) AttnFmAct(ctime *axon.Time) {
	/*
		pyn := ly.Shp.Dim(0)
		pxn := ly.Shp.Dim(1)

		if ly.Is4D() {
			var amax float32
			for py := 0; py < pyn; py++ {
				for px := 0; px < pxn; px++ {
					pi := py*pxn + px
					pl := &ly.Pools[pi+1]
					act := pl.Inhib.Act.Avg
					if act > amax {
						amax = act
					}
				}
			}
			for py := 0; py < pyn; py++ {
				for px := 0; px < pxn; px++ {
					pi := py*pxn + px
					pl := &ly.Pools[pi+1]
					act := pl.Inhib.Act.Avg
					attn := float32(1)
					if amax >= ly.SendAttn.Thr {
						attn = act / amax
					}
					for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
						nrn := &ly.Neurons[ni]
						nrn.Attn = attn
					}
				}
			}
		} else { // 2D
			lpl := &ly.Pools[0]
			amax := lpl.Inhib.Act.Max
			for py := 0; py < pyn; py++ {
				for px := 0; px < pxn; px++ {
					ni := py*pxn + px
					nrn := &ly.Neurons[ni]
					act := nrn.Act
					attn := float32(1)
					if amax >= ly.SendAttn.Thr {
						attn = act / amax
					}
					nrn.Attn = attn
				}
			}
		}
	*/
}

// SendAttnLays sends attention signal to all layers
func (ly *PulvAttnLayer) SendAttnLays(ctime *axon.Time) {
	for _, nm := range ly.SendAttn.ToLays {
		tlyi := ly.Network.LayerByName(nm)
		if tlyi == nil {
			continue
		}
		tly := tlyi.(axon.AxonLayer).AsAxon()
		ly.SendAttnLay(tly, ctime)
	}
}

// SendAttnLay sends attention signal to given layer
func (ly *PulvAttnLayer) SendAttnLay(tly *axon.Layer, ctime *axon.Time) {
	yn := ly.Shp.Dim(0)
	xn := ly.Shp.Dim(1)

	txn := tly.Shp.Dim(1)

	pyn := ints.MinInt(yn, tly.Shp.Dim(0))
	pxn := ints.MinInt(xn, tly.Shp.Dim(1))

	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			pi := py*xn + px
			var attn float32
			if ly.Is4D() {
				pl := &ly.Pools[pi+1]
				nrn := &ly.Neurons[pl.StIdx]
				attn = nrn.Attn
			} else {
				nrn := &ly.Neurons[pi]
				attn = nrn.Attn
			}
			ti := py*txn + px
			if tly.Is4D() {
				pl := &tly.Pools[ti+1]
				for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
					nrn := &tly.Neurons[ni]
					nrn.Attn = attn
				}
			} else {
				nrn := &tly.Neurons[ti]
				nrn.Attn = attn
			}
		}
	}
}
