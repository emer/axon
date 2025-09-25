// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/transform"

	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensor/tmath"
	"cogentcore.org/lab/tensorcore"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/vfilter"
)

// Vis does DoG filtering on images
type Vis struct {
	// if true, and input image is larger than target image size, central region is clipped out as the input -- otherwise image is sized to target size
	ClipToFit bool

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom vfilter.Geom `edit:"-" display:"inline"`

	// target image size to use -- images will be rescaled to this size
	ImageSize image.Point

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGFilter tensor.Float32 `display:"no-inline"`

	// current input image
	Image image.Image `display:"-" set:"-"`

	// input image as tensor
	ImageTsr tensor.Float32 `display:"no-inline"`

	// DoG filter output tensor
	OutTsr tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ClipToFit = true
	vi.DoG.Defaults()
	sz := 16
	spc := 2
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImageSize = image.Point{24, 24}
	vi.Geom.SetSize(vi.ImageSize.Add(vi.Geom.Border.Mul(2)))
	vi.DoG.ToTensor(&vi.DoGFilter)
	tensorcore.AddGridStylerTo(&vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
}

// SetImage sets current image for processing
func (vi *Vis) SetImage(img image.Image) {
	vi.Image = img
	insz := vi.Geom.In
	ibd := img.Bounds()
	isz := ibd.Size()
	if vi.ClipToFit && isz.X > insz.X && isz.Y > insz.Y {
		st := isz.Sub(insz).Div(2).Add(ibd.Min)
		ed := st.Add(insz)
		vi.Image = imagex.AsRGBA(img).SubImage(image.Rectangle{Min: st, Max: ed})
		vfilter.RGBToGrey(vi.Image, &vi.ImageTsr, 0, false) // pad for filt, bot zero
	} else {
		if isz != vi.ImageSize {
			vi.Image = transform.Resize(vi.Image, vi.ImageSize.X, vi.ImageSize.Y, transform.Linear)
			vfilter.RGBToGrey(vi.Image, &vi.ImageTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
			vfilter.WrapPad(&vi.ImageTsr, vi.Geom.FiltRt.X)
		}
	}
}

// LGNDoG runs DoG filtering on input image
// must have valid Image in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGFilter, dog.Net)
	vfilter.Conv1(&vi.Geom, flt, &vi.ImageTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	n := vi.OutTsr.Len()
	for i := range n {
		vi.OutTsr.SetFloat1D(math.Log(vi.OutTsr.Float1D(i)+1), i)
	}
	mx := stats.Max(tensor.As1D(&vi.OutTsr))
	tmath.DivOut(&vi.OutTsr, mx, &vi.OutTsr)
}

// Filter is overall method to run filters on given image
func (vi *Vis) Filter(img image.Image) {
	vi.SetImage(img)
	vi.LGNDoG()
}
