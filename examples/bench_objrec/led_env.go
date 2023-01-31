// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/emer/vision/vfilter"
	"github.com/emer/vision/vxform"
)

// LEDEnv generates images of old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDEnv struct {
	Nm        string          `desc:"name of this environment"`
	Dsc       string          `desc:"description of this environment"`
	Draw      LEDraw          `desc:"draws LEDs onto image"`
	Vis       Vis             `desc:"visual processing params"`
	NOutPer   int             `desc:"number of output units per LED item -- spiking benefits from replication"`
	MinLED    int             `min:"0" max:"19" desc:"minimum LED number to draw (0-19)"`
	MaxLED    int             `min:"0" max:"19" desc:"maximum LED number to draw (0-19)"`
	CurLED    int             `inactive:"+" desc:"current LED number that was drawn"`
	PrvLED    int             `inactive:"+" desc:"previous LED number that was drawn"`
	XFormRand vxform.Rand     `desc:"random transform parameters"`
	XForm     vxform.XForm    `desc:"current -- prev transforms"`
	Run       env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch     env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial     env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
	OrigImg   etensor.Float32 `desc:"original image prior to random transforms"`
	Output    etensor.Float32 `desc:"CurLED one-hot output tensor"`
}

func (ev *LEDEnv) Name() string { return ev.Nm }
func (ev *LEDEnv) Desc() string { return ev.Dsc }

func (ev *LEDEnv) Validate() error {
	return nil
}

func (ev *LEDEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *LEDEnv) States() env.Elements {
	isz := ev.Draw.ImgSize
	sz := ev.Vis.V1AllTsr.Shapes()
	nms := ev.Vis.V1AllTsr.DimNames()
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"V1", sz, nms},
		{"Output", []int{4, 5}, []string{"Y", "X"}},
	}
	return els
}

func (ev *LEDEnv) State(element string) etensor.Tensor {
	switch element {
	case "Image":
		vfilter.RGBToGrey(ev.Draw.Image, &ev.OrigImg, 0, false) // pad for filt, bot zero
		return &ev.OrigImg
	case "V1":
		return &ev.Vis.V1AllTsr
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *LEDEnv) Actions() env.Elements {
	return nil
}

func (ev *LEDEnv) Defaults() {
	ev.Draw.Defaults()
	ev.Vis.Defaults()
	ev.NOutPer = 5
	ev.XFormRand.TransX.Set(-0.25, 0.25)
	ev.XFormRand.TransY.Set(-0.25, 0.25)
	ev.XFormRand.Scale.Set(0.7, 1)
	ev.XFormRand.Rot.Set(-3.6, 3.6)
}

func (ev *LEDEnv) Init(run int) {
	ev.Draw.Init()
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Output.SetShape([]int{4, 5, ev.NOutPer, 1}, nil, []string{"Y", "X", "N", "1"})
}

func (ev *LEDEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}
	ev.DrawRndLED()
	ev.FilterImg()
	// debug only:
	// vfilter.RGBToGrey(ev.Draw.Image, &ev.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoObject renders specific object (LED number)
func (ev *LEDEnv) DoObject(objno int) {
	ev.DrawLED(objno)
	ev.FilterImg()
}

func (ev *LEDEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *LEDEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*LEDEnv)(nil)

// String returns the string rep of the LED env state
func (ev *LEDEnv) String() string {
	return fmt.Sprintf("Obj: %02d, %s", ev.CurLED, ev.XForm.String())
}

// SetOutput sets the output LED bit
func (ev *LEDEnv) SetOutput(out int) {
	ev.Output.SetZeros()
	si := ev.NOutPer * out
	for i := 0; i < ev.NOutPer; i++ {
		ev.Output.SetFloat1D(si+i, 1)
	}
}

// OutErr scores the output activity of network, returning the index of
// item with max overall activity, and 1 if that is error, 0 if correct.
// also returns a top-two error: if 2nd most active output was correct.
func (ev *LEDEnv) OutErr(tsr *etensor.Float32) (maxi int, err, err2 float64) {
	nc := ev.Output.Len() / ev.NOutPer
	maxi = 0
	maxv := 0.0
	for i := 0; i < nc; i++ {
		si := ev.NOutPer * i
		sum := 0.0
		for j := 0; j < ev.NOutPer; j++ {
			sum += tsr.FloatVal1D(si + j)
		}
		if sum > maxv {
			maxi = i
			maxv = sum
		}
	}
	err = 1.0
	if maxi == ev.CurLED {
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
			sum += tsr.FloatVal1D(si + j)
		}
		if sum > maxv2 {
			maxi2 = i
			maxv2 = sum
		}
	}
	err2 = err
	if maxi2 == ev.CurLED {
		err2 = 0
	}
	return
}

// DrawRndLED picks a new random LED and draws it
func (ev *LEDEnv) DrawRndLED() {
	rng := 1 + ev.MaxLED - ev.MinLED
	led := ev.MinLED + rand.Intn(rng)
	ev.DrawLED(led)
}

// DrawLED draw specified LED
func (ev *LEDEnv) DrawLED(led int) {
	ev.Draw.Clear()
	ev.Draw.DrawLED(led)
	ev.PrvLED = ev.CurLED
	ev.CurLED = led
	ev.SetOutput(ev.CurLED)
}

// FilterImg filters the image from LED
func (ev *LEDEnv) FilterImg() {
	ev.XFormRand.Gen(&ev.XForm)
	img := ev.XForm.Image(ev.Draw.Image)
	ev.Vis.Filter(img)
}
