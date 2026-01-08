// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objrec

import (
	"fmt"
	"image"

	"cogentcore.org/core/base/slicesx"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/paint"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/v1vision/v1std"
	"github.com/emer/v1vision/v1vision"
	"github.com/emer/v1vision/vxform"
)

// TrialState contains the state for a given trial.
// Trials are processed data-parallel per Step().
type TrialState struct {
	// LED number that was drawn
	LED int `edit:"-"`

	// current -- prev transforms
	XForm vxform.XForm

	// DrawImage is the image as drawn by the LED drawer.
	DrawImage image.Image

	// DrawImageTsr is the image as drawn by the LED drawer.
	DrawImageTsr tensor.Float32

	// XFormImage is the transformed image, from XForm.
	XFormImage image.Image
}

func (st *TrialState) String() string {
	return fmt.Sprintf("Obj: %02d, %s", st.LED, st.XForm.String())
}

// LEDEnv generates images of old-school "LED" style "letters"
// composed of a set of horizontal and vertical elements.
// All possible such combinations of 3 out of 6 line segments are created.
type LEDEnv struct {

	// name of this environment
	Name string

	// NData is the number of steps to process in data-parallel.
	NData int

	// Trials has NData state per trial for last Step()
	Trials []TrialState

	// draws LEDs onto image
	Draw LEDraw

	// V1c does all the V1 processing.
	V1c v1std.V1cGrey `new-window:"+"`

	// Image manages setting image.
	Image v1std.Image

	// number of output units per LED item -- spiking benefits from replication
	NOutPer int

	// minimum LED number to draw (0-19)
	MinLED int `min:"0" max:"19"`

	// maximum LED number to draw (0-19)
	MaxLED int `min:"0" max:"19"`

	// random transform parameters
	XFormRand vxform.Rand

	// CurLED one-hot output tensor, dims: [NData][4][5][NOutPer][1]
	Output tensor.Float32

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `display:"-"`

	// random seed: set this to control sequence
	RandSeed int64 `edit:"-"`
}

func (ev *LEDEnv) Trial(di int) *TrialState {
	return &ev.Trials[di]
}

func (ev *LEDEnv) Label() string { return ev.Name }

func (ev *LEDEnv) State(element string) tensor.Values {
	switch element {
	case "Image":
		// todo:
		// v1vision.RGBToGrey(paint.RenderToImage(ev.Draw.Paint), &ev.OrigImg, 0, false) // pad for filt, bot zero
		// return &ev.OrigImg
	case "V1":
		return ev.V1c.Output
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *LEDEnv) Defaults() {
	ev.Draw.Defaults()
	ev.V1c.Defaults()
	ev.Image.Defaults()
	ev.Image.Size = image.Point{40, 40}
	ev.V1c.SetSize(6, 2) // V1mF16 typically = 12, no border, spc = 4 -- using 1/2 that here
	ev.NOutPer = 5
	ev.XFormRand.TransX.Set(-0.25, 0.25)
	ev.XFormRand.TransY.Set(-0.25, 0.25)
	ev.XFormRand.Scale.Set(0.7, 1)
	ev.XFormRand.Rot.Set(-3.6, 3.6)
}

func (ev *LEDEnv) Config(ndata int, netGPU *gpu.GPU) {
	ev.NData = ndata
	ev.Trials = slicesx.SetLength(ev.Trials, ndata)
	v1vision.ComputeGPU = netGPU
	ev.V1c.Config(ndata, ev.Image.Size)
	ev.Output.SetShapeSizes(ndata, 4, 5, ev.NOutPer, 1)
	ev.RandSeed = 73
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RandSeed)
	} else {
		ev.Rand.Seed(ev.RandSeed)
	}
}

func (ev *LEDEnv) Init(run int) {
	ev.Draw.Init()
	ev.RandSeed = int64(73 + run)
	ev.Rand.Seed(ev.RandSeed)
}

func (ev *LEDEnv) Step() bool {
	imgs := make([]image.Image, ev.NData)
	for di := range ev.NData {
		st := ev.Trial(di)
		ev.DrawRandLED(di, st)
		imgs[di] = st.XFormImage
	}
	ev.V1c.RunImages(&ev.Image, imgs...)
	return true
}

func (ev *LEDEnv) Action(element string, input tensor.Values) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*LEDEnv)(nil)

func (ev *LEDEnv) String() string {
	return ev.TrialName(0)
}

// TrialName returns the string rep of the LED env state
func (ev *LEDEnv) TrialName(di int) string {
	st := ev.Trial(di)
	return st.String()
}

// SetOutput sets the output LED bit for given data item
func (ev *LEDEnv) SetOutput(out, di int) {
	ot := ev.Output.SubSpace(di).(*tensor.Float32)
	ot.SetZeros()
	si := ev.NOutPer * out
	for i := 0; i < ev.NOutPer; i++ {
		ot.SetFloat1D(1, si+i)
	}
}

// OutErr scores the output activity of network, returning the index of
// item with max overall activity, and 1 if that is error, 0 if correct.
// also returns a top-two error: if 2nd most active output was correct.
func (ev *LEDEnv) OutErr(tsr *tensor.Float64, di, corLED int) (maxi int, err, err2 float64) {
	ot := ev.Output.SubSpace(di).(*tensor.Float32)
	nc := ot.Len() / ev.NOutPer
	maxi = 0
	maxv := 0.0
	for i := 0; i < nc; i++ {
		si := ev.NOutPer * i
		sum := 0.0
		for j := 0; j < ev.NOutPer; j++ {
			sum += tsr.Float1D(si + j)
		}
		if sum > maxv {
			maxi = i
			maxv = sum
		}
	}
	err = 1.0
	if maxi == corLED {
		err = 0
	}
	maxv2 := 0.0
	maxi2 := 0
	for i := 0; i < nc; i++ {
		if i == maxi { // skip top
			continue
		}
		si := ev.NOutPer * i
		sum := 0.0
		for j := 0; j < ev.NOutPer; j++ {
			sum += tsr.Float1D(si + j)
		}
		if sum > maxv2 {
			maxi2 = i
			maxv2 = sum
		}
	}
	err2 = err
	if maxi2 == corLED {
		err2 = 0
	}
	return
}

// DrawRandLED picks a new random LED and draws it
func (ev *LEDEnv) DrawRandLED(di int, st *TrialState) {
	rng := 1 + ev.MaxLED - ev.MinLED
	led := ev.MinLED + ev.Rand.Intn(rng)
	ev.DrawLED(led)
	st.LED = led
	st.DrawImage = paint.RenderToImage(ev.Draw.Paint)
	ev.SetOutput(led, di)
	ev.XFormRand.Gen(&st.XForm, &ev.Rand)
	st.XFormImage = st.XForm.Image(st.DrawImage)
}

// DrawLED draw specified LED
func (ev *LEDEnv) DrawLED(led int) {
	ev.Draw.Clear()
	ev.Draw.DrawLED(led)
}
