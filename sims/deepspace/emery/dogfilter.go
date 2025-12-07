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
	"github.com/emer/v1vision/motion"
	"github.com/emer/v1vision/v1vision"
)

// Vis does DoG filtering on images
type Vis struct {
	// if true, and input image is larger than target image size,
	// central region is clipped out as the input.
	// otherwise image is sized to target size
	ClipToFit bool

	// NFrames is number of frames to render for motion.
	NFrames int

	// LGN DoG filter parameters
	DoG dog.Filter

	// Motion filter parameters.
	Motion motion.Params

	// geometry of input, output
	Geom v1vision.Geom `edit:"-" display:"inline"`

	// target image size to use -- images will be rescaled to this size
	ImageSize image.Point

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGFilter tensor.Float32 `display:"no-inline"`

	// current input image
	Image image.Image `display:"-" set:"-"`

	// input image as tensor
	ImageTsr tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ClipToFit = true
	vi.DoG.Defaults()
	vi.Motion.Defaults()
	vi.NFrames = 16
	vi.Motion.SlowTau = 4
	vi.Motion.FastTau = 2
	vi.Motion.FullGain = 3
	vi.Motion.IntegTau = 6
	sz := 8
	spc := 2
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImageSize = image.Point{64, 64} // note: must coordinate with camera size!
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
	if isz == vi.ImageSize {
		vi.Image = img
		v1vision.RGBToGrey(vi.Image, &vi.ImageTsr, 0, false) // pad for filt, bot zero
	} else if vi.ClipToFit {
		st := isz.Sub(insz).Div(2).Add(ibd.Min)
		ed := st.Add(insz)
		vi.Image = imagex.AsRGBA(img).SubImage(image.Rectangle{Min: st, Max: ed})
		v1vision.RGBToGrey(vi.Image, &vi.ImageTsr, 0, false) // pad for filt, bot zero
	} else {
		vi.Image = transform.Resize(vi.Image, vi.ImageSize.X, vi.ImageSize.Y, transform.Linear)
		v1vision.RGBToGrey(vi.Image, &vi.ImageTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
		v1vision.WrapPad(&vi.ImageTsr, vi.Geom.FiltRt.X)
	}
}

// LGNDoG runs DoG filtering on input image
// must have valid Image in place to start.
func (vi *Vis) LGNDoG(out *tensor.Float32) {
	flt := vi.DoG.FilterTensor(&vi.DoGFilter, dog.Net)
	v1vision.Conv1(&vi.Geom, flt, &vi.ImageTsr, out, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	n := out.Len()
	for i := range n {
		out.SetFloat1D(math.Log(out.Float1D(i)+1), i)
	}
	mx := stats.Max(tensor.As1D(out))
	tmath.DivOut(out, mx, out)
}

// FilterImage runs filters on given image, integrating results for motion.
func (vi *Vis) FilterImage(img image.Image, dout, slow, fast, star, insta, full *tensor.Float32, visNorm *float32) {
	vi.SetImage(img)
	vi.LGNDoG(dout)
	ve := vi.Motion.IntegrateFrame(slow, fast, dout)
	vi.Motion.StarMotion(star, slow, fast)
	vi.Motion.FullField(insta, full, star, ve, visNorm)
}
