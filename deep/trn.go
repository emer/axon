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
	"github.com/goki/mat32"
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

// TopoAct provides for topographic gaussian activation integrating over neighborhood.
type TopoAct struct {
	On    bool      `desc:"use topographic inhibition"`
	Width int       `desc:"half-width of topographic inhibition within layer"`
	Sigma float32   `desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Gain  float32   `desc:"overall inhibition multiplier for topographic inhibition (generally <= 1)"`
	Wts   []float32 `inactive:"+" desc:"gaussian weights as function of distance, precomputed.  index 0 = dist 1"`
}

func (ti *TopoAct) Defaults() {
	ti.Width = 4
	ti.Sigma = 0.5
	ti.Gain = 1
	ti.Update()
}

func (ti *TopoAct) Update() {
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
	axon.Layer         // access as .Layer
	TopoAct    TopoAct `desc:"topographic inhibition parameters for pool-level inhibition (only used for layers with pools)"`
	Driver     string  `desc:"name of SuperLayer that sends 5IB Burst driver inputs to this layer"`
}

var KiT_TRCALayer = kit.Types.AddType(&TRCALayer{}, LayerProps)

func (ly *TRCALayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0.5
	ly.Act.Decay.Glong = 1
	ly.Act.Decay.KNa = 0
	ly.Act.GABAB.Gbar = 0.005 // output layer settings
	ly.Act.NMDA.Gbar = 0.01
	ly.TopoAct.Defaults()
	ly.Typ = TRC
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *TRCALayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.TopoAct.Update()
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

// TopoGePos returns position-specific Ge contribution
func (ly *TRCALayer) TopoGePos(py, px, d int) float32 {
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	if py < 0 || py >= pyn {
		return 0
	}
	if px < 0 || px >= pxn {
		return 0
	}
	pi := py*pxn + px
	pl := ly.Pools[pi+1]
	g := ly.TopoAct.Wts[d]
	return g * pl.Inhib.GiOrig
}

// TopoGe computes topographic Ge from pools
func (ly *TRCALayer) TopoGe(ltime *axon.Time) {
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	wd := ly.TopoAct.Width

	// dly, err := ly.DriverLayer(ly.Driver)
	// if err != nil {
	// 	return
	// }
	// sly, issuper := dly.AxonLay.(*SuperLayer)

	laymax := float32(0)
	np := len(ly.Pools)
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		laymax = mat32.Max(laymax, pl.Inhib.GiOrig)
	}

	// laymax *= ly.TopoAct.LayGi

	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			max := laymax
			for iy := 1; iy <= wd; iy++ {
				for ix := 1; ix <= wd; ix++ {
					max = mat32.Max(max, ly.TopoGePos(py+iy, px+ix, ints.MinInt(iy-1, ix-1)))
					max = mat32.Max(max, ly.TopoGePos(py-iy, px+ix, ints.MinInt(iy-1, ix-1)))
					max = mat32.Max(max, ly.TopoGePos(py+iy, px-ix, ints.MinInt(iy-1, ix-1)))
					max = mat32.Max(max, ly.TopoGePos(py-iy, px-ix, ints.MinInt(iy-1, ix-1)))
				}
			}
			pi := py*pxn + px
			pl := &ly.Pools[pi+1]
			pl.Inhib.Gi = mat32.Max(max, pl.Inhib.Gi)
		}
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
// func (ly *TRCALayer) InhibFmGeAct(ltime *axon.Time) {
// 	lpl := &ly.Pools[0]
// 	ly.Inhib.Layer.Inhib(&lpl.Inhib, ly.ActAvg.GiMult)
// 	ly.PoolInhibFmGeAct(ltime)
// 	if ly.Is4D() && ly.TopoAct.On {
// 		ly.TopoGi(ltime)
// 	}
// 	ly.InhibFmPool(ltime)
// }
