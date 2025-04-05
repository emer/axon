// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lvis

import (
	"image"

	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/colorspace"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/fffb"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/vfilter"
)

// ColorVis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type ColorVis struct {
	Img        *V1Img                     `desc:"image that we operate upon -- one image often shared among multiple filters"`
	DoG        dog.Filter                 `desc:"LGN DoG filter parameters"`
	DoGNames   []string                   `desc:"names of the dog gain sets -- for naming output data"`
	DoGGains   []float32                  `desc:"overall gain factors, to compensate for diffs in OnGains"`
	DoGOnGains []float32                  `desc:"OnGain factors -- 1 = perfect balance, otherwise has relative imbalance for capturing main effects"`
	Geom       vfilter.Geom               `inactive:"+" view:"inline" desc:"geometry of input, output"`
	KWTA       kwta.KWTA                  `desc:"kwta parameters"`
	DoGTsr     tensor.Float32             `view:"no-inline" desc:"DoG filter tensor -- has 3 filters (on, off, net)"`
	DoGTab     table.Table                `view:"no-inline" desc:"DoG filter table (view only)"`
	KwtaTsr    tensor.Float32             `view:"no-inline" desc:"kwta output tensor"`
	OutAll     tensor.Float32             `view:"no-inline" desc:"output from 3 dogs with different tuning -- this is what goes into input layer"`
	OutTsrs    map[string]*tensor.Float32 `view:"no-inline" desc:"DoG filter output tensors"`
	Inhibs     fffb.Inhibs                `view:"no-inline" desc:"inhibition values for KWTA"`
}

func (vi *ColorVis) Defaults(bord_ex, sz, spc int, img *V1Img) {
	vi.Img = img
	vi.DoGNames = []string{"Bal"} // , "On", "Off"} // balanced, gain toward On, gain toward Off
	vi.DoGGains = []float32{8, 4.1, 4.4}
	vi.DoGOnGains = []float32{1, 1.2, 0.833}
	vi.DoG.Defaults()
	vi.DoG.SetSize(sz, spc)
	vi.DoG.OnSig = .5 // no spatial component, just pure contrast
	vi.DoG.OffSig = .5
	vi.DoG.Gain = 8
	vi.DoG.OnGain = 1
	vi.KWTA.Defaults()
	vi.KWTA.LayFFFB.Gi = 0
	vi.KWTA.PoolFFFB.Gi = 1.2
	vi.KWTA.XX1.Gain = 80
	vi.KWTA.XX1.NVar = 0.01

	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{sz/2 + bord_ex, sz/2 + bord_ex}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.DoG.ToTensor(&vi.DoGTsr)
	vi.DoGTab.Init()
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	// vi.DoGTab.Cols[1].SetMetaData("max", "0.2")
	// vi.DoGTab.Cols[1].SetMetaData("min", "-0.2")
	vi.OutTsrs = make(map[string]*tensor.Float32)
}

// OutTsr gets output tensor of given name, creating if not yet made
func (vi *ColorVis) OutTsr(name string) *tensor.Float32 {
	if vi.OutTsrs == nil {
		vi.OutTsrs = make(map[string]*tensor.Float32)
	}
	tsr, ok := vi.OutTsrs[name]
	if !ok {
		tsr = &tensor.Float32{}
		vi.OutTsrs[name] = tsr
		// tsr.SetMetaData("grid-fill", "1")
	}
	return tsr
}

// ColorDoG runs color contrast DoG filtering on input image
// must have valid Img in place to start.
func (vi *ColorVis) ColorDoG() {
	rimg := vi.Img.LMS.SubSpace(int(colorspace.LC)).(*tensor.Float32)
	gimg := vi.Img.LMS.SubSpace(int(colorspace.MC)).(*tensor.Float32)
	// rimg.SetMetaData("grid-fill", "1")
	// gimg.SetMetaData("grid-fill", "1")
	vi.OutTsrs["Red"] = rimg
	vi.OutTsrs["Green"] = gimg

	bimg := vi.Img.LMS.SubSpace(int(colorspace.SC)).(*tensor.Float32)
	yimg := vi.Img.LMS.SubSpace(int(colorspace.LMC)).(*tensor.Float32)
	// bimg.SetMetaData("grid-fill", "1")
	// yimg.SetMetaData("grid-fill", "1")
	vi.OutTsrs["Blue"] = bimg
	vi.OutTsrs["Yellow"] = yimg

	/*
		// for display purposes only:
		byimg := vi.ImgLMS.SubSpace(int(colorspace.SvLMC)).(*tensor.Float32)
		rgimg := vi.ImgLMS.SubSpace(int(colorspace.LvMC)).(*tensor.Float32)
		byimg.SetMetaData("grid-fill", "1")
		rgimg.SetMetaData("grid-fill", "1")
		vi.OutTsrs["Blue-Yellow"] = byimg
		vi.OutTsrs["Red-Green"] = rgimg
	*/

	for i, nm := range vi.DoGNames {
		vi.DoGFilter(nm, vi.DoGGains[i], vi.DoGOnGains[i])
	}
}

// DoGFilter runs filtering for given gain factors
func (vi *ColorVis) DoGFilter(name string, gain, onGain float32) {
	dogOn := vi.DoG.FilterTensor(&vi.DoGTsr, dog.On)
	dogOff := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Off)

	rgtsr := vi.OutTsr("DoG_" + name + "_Red-Green")
	rimg := vi.OutTsr("Red")
	gimg := vi.OutTsr("Green")
	vfilter.ConvDiff(&vi.Geom, dogOn, dogOff, rimg, gimg, rgtsr, gain, onGain)

	bytsr := vi.OutTsr("DoG_" + name + "_Blue-Yellow")
	bimg := vi.OutTsr("Blue")
	yimg := vi.OutTsr("Yellow")
	vfilter.ConvDiff(&vi.Geom, dogOn, dogOff, bimg, yimg, bytsr, gain, onGain)
}

// AggAll aggregates the different DoG components into OutAll
func (vi *ColorVis) AggAll() {
	otsr := vi.OutTsr("DoG_" + vi.DoGNames[0] + "_Red-Green")
	ny := otsr.DimSize(1)
	nx := otsr.DimSize(2)
	oshp := []int{ny, nx, 2, 2 * len(vi.DoGNames)}
	vi.OutAll.SetShapeSizes(oshp...)
	// vi.OutAll.SetMetaData("grid-fill", "1")
	for i, nm := range vi.DoGNames {
		rgtsr := vi.OutTsr("DoG_" + nm + "_Red-Green")
		bytsr := vi.OutTsr("DoG_" + nm + "_Blue-Yellow")
		vfilter.OuterAgg(i*2, 0, rgtsr, &vi.OutAll)
		vfilter.OuterAgg(i*2+1, 0, bytsr, &vi.OutAll)
	}
	if vi.KWTA.On {
		vi.KWTA.KWTAPool(&vi.OutAll, &vi.KwtaTsr, &vi.Inhibs, nil)
	} else {
		vi.KwtaTsr.CopyFrom(&vi.OutAll)
	}
}

// Filter is overall method to run filters on image set by SetImage*
func (vi *ColorVis) Filter() {
	vi.ColorDoG()
	vi.AggAll()
}
