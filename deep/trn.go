// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/efuns"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
)

// TRNLayer copies inhibition from pools in CT and TRC layers, and from other
// TRNLayers, and pools this inhibition using the Max operation
type TRNLayer struct {
	axon.Layer
	ILayers emer.LayNames `desc:"layers that we receive inhibition from"`
}

var KiT_TRNLayer = kit.Types.AddType(&TRNLayer{}, axon.LayerProps)

func (ly *TRNLayer) Defaults() {
	ly.Layer.Defaults()
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *TRNLayer) InitActs() {
	ly.Layer.InitActs()
}

///////////////////////////////////////////////////////////////////////////////////////
// TRCALayer -- attention TRC

// TopoDrive provides for topographic gaussian activation integrating over neighborhood.
type TopoDrive struct {
	On      bool      `desc:"use topographic inhibition"`
	Width   int       `desc:"half-width of topographic inhibition within layer"`
	Wrap    bool      `desc:"wrap-around coordinates -- otherwise clipped at edges"`
	Sigma   float32   `desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Gain    float32   `desc:"overall inhibition multiplier for topographic inhibition (generally <= 1)"`
	SendThr float32   `desc:"threshold on layer-wide max activation (or average act for pooled 4D layers) for sending attention (below this, sends attn = 1)"`
	Wts     []float32 `inactive:"+" desc:"gaussian weights as function of distance, precomputed.  index 0 = dist 1"`
}

func (ti *TopoDrive) Defaults() {
	ti.Width = 4
	ti.Wrap = false
	ti.Sigma = 0.5
	ti.Gain = 1
	ti.SendThr = 0.5
	ti.Update()
}

func (ti *TopoDrive) Update() {
	if len(ti.Wts) != ti.Width {
		ti.Wts = make([]float32, ti.Width)
	}
	sig := float32(ti.Width) * ti.Sigma
	for i := range ti.Wts {
		ti.Wts[i] = ti.Gain * efuns.Gauss1DNoNorm(float32(i+1), sig)
	}
}

// TRCALayer is the thalamic relay cell layer for Attention in DeepAxon.
type TRCALayer struct {
	axon.Layer               // access as .Layer
	TopoDrive  TopoDrive     `view:"inline" desc:"topographic parameters for integrating driver inputs"`
	Driver     string        `desc:"name of Layer that drives topographic attentional input to this layer"`
	SendTo     emer.LayNames `desc:"list of layers to send attentional modulation to"`
}

var KiT_TRCALayer = kit.Types.AddType(&TRCALayer{}, LayerProps)

func (ly *TRCALayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0.5
	ly.Act.Decay.Glong = 1
	ly.Act.Decay.KNa = 0
	ly.Act.GABAB.Gbar = 0.005 // output layer settings
	ly.Act.NMDA.Gbar = 0.01
	ly.TopoDrive.Defaults()
	ly.TopoDrive.On = true
	ly.Typ = TRC
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *TRCALayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.TopoDrive.Update()
}

func (ly *TRCALayer) Class() string {
	return "TRCA " + ly.Cls
}

func (ly *TRCALayer) IsTarget() bool {
	return false // We are not
}

// DriverLayer returns the driver layer for given Driver
func (ly *TRCALayer) DriverLayer(drv string) (*axon.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(drv)
	if err != nil {
		err = fmt.Errorf("TRCALayer %s: Driver Layer: %v", ly.Name(), err)
		log.Println(err)
		return nil, err
	}
	return tly.(axon.AxonLayer).AsAxon(), nil
}

// TopoGePos returns position-specific Ge contribution from driver layer, or false if not valid
func (ly *TRCALayer) TopoGePos(dly *axon.Layer, py, px, widx int, sum *float32, n *int) {
	pyn := dly.Shp.Dim(0)
	pxn := dly.Shp.Dim(1)
	if py < 0 || py >= pyn {
		if !ly.TopoDrive.Wrap {
			return
		}
		if py < 0 {
			py += pyn
		} else {
			py -= pyn
		}
	}
	if px < 0 || px >= pxn {
		if !ly.TopoDrive.Wrap {
			return
		}
		if px < 0 {
			px += pxn
		} else {
			px -= pxn
		}
	}
	pi := py*pxn + px
	var g float32
	if dly.Is4D() {
		pl := &dly.Pools[pi+1]
		g = pl.Inhib.Act.Avg
	} else {
		nr := &dly.Neurons[pi+1]
		g = nr.Act
	}
	g *= ly.TopoDrive.Wts[widx]
	*sum += g
	(*n)++
}

func (ly *TRCALayer) GeFmDriverNeuron(tni int, drvGe float32, cyc int) {
	if tni >= len(ly.Neurons) {
		return
	}
	nrn := &ly.Neurons[tni]
	if nrn.IsOff() {
		return
	}
	actm := nrn.ActM
	geRaw := nrn.GeRaw + drvGe
	nrn.NMDA = ly.Act.NMDA.NMDA(nrn.NMDA, geRaw, nrn.NMDASyn)
	nrn.Gnmda = ly.Act.NMDA.Gnmda(nrn.NMDA, nrn.VmDend)
	ly.Act.GeFmRaw(nrn, geRaw, cyc, actm)
	nrn.GeRaw = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}

// GeFmDrivers computes excitatory conductance from driver neurons
func (ly *TRCALayer) GeFmDrivers(ltime *axon.Time) {
	cyc := ltime.Cycle

	yn := ly.Shp.Dim(0)
	xn := ly.Shp.Dim(1)
	wd := ly.TopoDrive.Width

	dly, err := ly.DriverLayer(ly.Driver)
	if err != nil {
		return
	}
	// sly, issuper := dly.AxonLay.(*SuperLayer)

	pyn := ints.MinInt(yn, dly.Shp.Dim(0))
	pxn := ints.MinInt(xn, dly.Shp.Dim(1))

	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			sum := float32(0)
			n := 0
			for iy := 1; iy <= wd; iy++ {
				for ix := 1; ix <= wd; ix++ {
					widx := ints.MinInt(iy-1, ix-1)
					ly.TopoGePos(dly, py+iy, px+ix, widx, &sum, &n)
					ly.TopoGePos(dly, py-iy, px+ix, widx, &sum, &n)
					ly.TopoGePos(dly, py+iy, px-ix, widx, &sum, &n)
					ly.TopoGePos(dly, py-iy, px-ix, widx, &sum, &n)
				}
			}
			if n == 0 {
				continue
			}
			sum /= float32(n)
			pi := py*xn + px
			if ly.Is4D() {
				pl := &ly.Pools[pi+1]
				for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
					ly.GeFmDriverNeuron(ni, sum, cyc)
				}
			} else {
				ly.GeFmDriverNeuron(pi, sum, cyc)
			}
		}
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *TRCALayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	// ly.GFmIncNeur(ltime) // regular
	ly.GeFmDrivers(ltime)
}

// CyclePost is called at end of Cycle
// We use it to send Attn
func (ly *TRCALayer) CyclePost(ltime *axon.Time) {
	ly.AttnFmAct(ltime)
	ly.SendAttn(ltime)
}

// AttnFmAct computes our attention signal from activations
func (ly *TRCALayer) AttnFmAct(ltime *axon.Time) {
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
				if amax >= ly.TopoDrive.SendThr {
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
				if amax >= ly.TopoDrive.SendThr {
					attn = act / amax
				}
				nrn.Attn = attn
			}
		}
	}
}

// SendAttn sends attention signal to SendTo layers
func (ly *TRCALayer) SendAttn(ltime *axon.Time) {
	for _, nm := range ly.SendTo {
		tlyi := ly.Network.LayerByName(nm)
		if tlyi == nil {
			continue
		}
		tly := tlyi.(axon.AxonLayer).AsAxon()
		ly.SendAttnLay(tly, ltime)
	}
}

// SendAttnLay sends attention signal to given layer
func (ly *TRCALayer) SendAttnLay(tly *axon.Layer, ltime *axon.Time) {
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
