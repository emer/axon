// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lvis

import (
	"image"

	"cogentcore.org/core/core"
	"cogentcore.org/lab/tensor"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/colorspace"
	"github.com/emer/v1vision/fffb"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1complex"
	"github.com/emer/v1vision/vfilter"
)

// Img manages conversion of a bitmap image into tensor formats for
// subsequent processing by filters.
type V1Img struct {
	File core.Filename  `desc:"name of image file to operate on"`
	Size image.Point    `desc:"target image size to use -- images will be rescaled to this size"`
	Img  image.Image    `view:"-" desc:"current input image"`
	Tsr  tensor.Float32 `view:"no-inline" desc:"input image as an RGB tensor"`
	LMS  tensor.Float32 `view:"no-inline" desc:"LMS components + opponents tensor version of image"`
}

func (vi *V1Img) Defaults() {
	vi.Size = image.Point{128, 128}
	// vi.Tsr.SetMetaData("image", "+")
	// vi.Tsr.SetMetaData("min", "0")
}

// SetImage sets current image for processing
// and converts to a float32 tensor for processing
func (vi *V1Img) SetImage(img image.Image, filtsz int) {
	vi.Img = img
	isz := vi.Img.Bounds().Size()
	if isz != vi.Size {
		vi.Img = transform.Resize(vi.Img, vi.Size.X, vi.Size.Y, transform.Linear)
	}
	vfilter.RGBToTensor(vi.Img, &vi.Tsr, filtsz, false) // pad for filt, bot zero
	// vfilter.WrapPadRGB(&vi.Tsr, filtsz)
	vfilter.FadePadRGB(&vi.Tsr, filtsz)
	colorspace.RGBTensorToLMSComps(&vi.LMS, &vi.Tsr)
	// vi.Tsr.SetMetaData("image", "+")
	// vi.Tsr.SetMetaData("min", "0")
}

// V1sOut contains output tensors for V1 Simple filtering, one per opponnent
type V1sOut struct {
	Tsr      tensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output tensor"`
	ExtGiTsr tensor.Float32 `view:"no-inline" desc:"V1 simple extra Gi from neighbor inhibition tensor"`
	KwtaTsr  tensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, kwta output tensor"`
	PoolTsr  tensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor"`
}

// Vis encapsulates specific visual processing pipeline for V1 filtering
type Vis struct {
	Color         bool                          `desc:"if true, do full color filtering -- else Black/White only"`
	SepColor      bool                          `desc:"record separate rows in V1s summary for each color -- otherwise just records the max across all colors"`
	ColorGain     float32                       `def:"8" desc:"extra gain for color channels -- lower contrast in general"`
	Img           *V1Img                        `desc:"image that we operate upon -- one image often shared among multiple filters"`
	V1sGabor      gabor.Filter                  `view:"inline" desc:"V1 simple gabor filter parameters"`
	V1sGeom       vfilter.Geom                  `inactive:"+" view:"inline" desc:"geometry of input, output for V1 simple-cell processing"`
	V1sNeighInhib kwta.NeighInhib               `desc:"neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code"`
	V1sKWTA       kwta.KWTA                     `desc:"kwta parameters for V1s"`
	V1sGaborTsr   tensor.Float32                `view:"no-inline" desc:"V1 simple gabor filter tensor"`
	V1s           [colorspace.OpponentsN]V1sOut `view:"inline" desc:"V1 simple gabor filter output, per channel"`
	V1sMaxTsr     tensor.Float32                `view:"no-inline" desc:"max over V1 simple gabor filters output tensor"`
	V1sPoolTsr    tensor.Float32                `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor"`
	V1sUnPoolTsr  tensor.Float32                `view:"no-inline" desc:"V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor"`
	V1sAngOnlyTsr tensor.Float32                `view:"no-inline" desc:"V1 simple gabor filter output, angle-only features tensor"`
	V1sAngPoolTsr tensor.Float32                `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor"`
	V1cLenSumTsr  tensor.Float32                `view:"no-inline" desc:"V1 complex length sum filter output tensor"`
	V1cEndStopTsr tensor.Float32                `view:"no-inline" desc:"V1 complex end stop filter output tensor"`
	V1AllTsr      tensor.Float32                `view:"no-inline" desc:"Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total (9 if SepColor)"`
	V1sInhibs     fffb.Inhibs                   `view:"no-inline" desc:"inhibition values for V1s KWTA"`
}

// Defaults sets default values: high: sz = 12, spc = 4, med: sz = 24, spc = 8
func (vi *Vis) Defaults(bord_ex, sz, spc int, img *V1Img) {
	vi.Img = img
	vi.Color = true
	vi.SepColor = false
	vi.ColorGain = 8
	vi.V1sGabor.Defaults()
	vi.V1sGabor.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.V1sGeom.Set(image.Point{sz/2 + bord_ex, sz/2 + bord_ex}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	// values from lvis models
	vi.V1sKWTA.LayFFFB.Gi = 1.5
	vi.V1sKWTA.XX1.Gain = 80
	vi.V1sKWTA.XX1.NVar = 0.01
	vi.V1sGabor.ToTensor(&vi.V1sGaborTsr)
}

func (vi *Vis) V1SimpleImg(v1s *V1sOut, img *tensor.Float32, gain float32) {
	vfilter.Conv(&vi.V1sGeom, &vi.V1sGaborTsr, img, &v1s.Tsr, gain*vi.V1sGabor.Gain)
	if vi.V1sNeighInhib.On {
		vi.V1sNeighInhib.Inhib4(&v1s.Tsr, &v1s.ExtGiTsr)
	} else {
		v1s.ExtGiTsr.SetZeros()
	}
	if vi.V1sKWTA.On {
		vi.V1sKWTA.KWTAPool(&v1s.Tsr, &v1s.KwtaTsr, &vi.V1sInhibs, &v1s.ExtGiTsr)
	} else {
		v1s.KwtaTsr.CopyFrom(&v1s.Tsr)
	}
}

// V1Simple runs all V1Simple Gabor filtering, depending on Color
func (vi *Vis) V1Simple() {
	grey := vi.Img.LMS.SubSpace(int(colorspace.GREY)).(*tensor.Float32)
	wbout := &vi.V1s[colorspace.WhiteBlack]
	vi.V1SimpleImg(wbout, grey, 1)
	tensor.SetShapeFrom(&vi.V1sMaxTsr, &wbout.KwtaTsr)
	vi.V1sMaxTsr.CopyFrom(&wbout.KwtaTsr)
	if vi.Color {
		rgout := &vi.V1s[colorspace.RedGreen]
		rgimg := vi.Img.LMS.SubSpace(int(colorspace.LvMC)).(*tensor.Float32)
		vi.V1SimpleImg(rgout, rgimg, vi.ColorGain)
		byout := &vi.V1s[colorspace.BlueYellow]
		byimg := vi.Img.LMS.SubSpace(int(colorspace.SvLMC)).(*tensor.Float32)
		vi.V1SimpleImg(byout, byimg, vi.ColorGain)
		for i, vl := range vi.V1sMaxTsr.Values {
			rg := rgout.KwtaTsr.Values[i]
			by := byout.KwtaTsr.Values[i]
			if rg > vl {
				vl = rg
			}
			if by > vl {
				vl = by
			}
			vi.V1sMaxTsr.Values[i] = vl
		}
	}
}

// it computes Angle-only, max-pooled version of V1Simple inputs.
func (vi *Vis) V1Complex() {
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sMaxTsr, &vi.V1sPoolTsr)
	vfilter.MaxReduceFilterY(&vi.V1sMaxTsr, &vi.V1sAngOnlyTsr)
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sAngOnlyTsr, &vi.V1sAngPoolTsr)
	v1complex.LenSum4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr)
	v1complex.EndStop4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr, &vi.V1cEndStopTsr)
}

// V1All aggregates all the relevant simple and complex features
// into the V1AllTsr which is used for input to a network
func (vi *Vis) V1All() {
	ny := vi.V1sPoolTsr.DimSize(0)
	nx := vi.V1sPoolTsr.DimSize(1)
	nang := vi.V1sPoolTsr.DimSize(3)
	nrows := 5
	if vi.Color && vi.SepColor {
		nrows += 4
	}
	oshp := []int{ny, nx, nrows, nang}
	vi.V1AllTsr.SetShapeSizes(oshp...)
	// 1 length-sum
	vfilter.FeatAgg([]int{0}, 0, &vi.V1cLenSumTsr, &vi.V1AllTsr)
	// 2 end-stop
	vfilter.FeatAgg([]int{0, 1}, 1, &vi.V1cEndStopTsr, &vi.V1AllTsr)
	// 2 pooled simple cell
	if vi.Color && vi.SepColor {
		rgout := &vi.V1s[colorspace.RedGreen]
		byout := &vi.V1s[colorspace.BlueYellow]
		vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &rgout.KwtaTsr, &rgout.PoolTsr)
		vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &byout.KwtaTsr, &byout.PoolTsr)
		vfilter.FeatAgg([]int{0, 1}, 5, &rgout.PoolTsr, &vi.V1AllTsr)
		vfilter.FeatAgg([]int{0, 1}, 7, &byout.PoolTsr, &vi.V1AllTsr)
	} else {
		vfilter.FeatAgg([]int{0, 1}, 3, &vi.V1sPoolTsr, &vi.V1AllTsr)
	}
}

// Filter is overall method to run filters on image set by SetImage*
func (vi *Vis) Filter() {
	vi.V1Simple()
	vi.V1Complex()
	vi.V1All()
}
