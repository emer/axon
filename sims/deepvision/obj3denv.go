// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

import (
	"fmt"
	"image"
	"path/filepath"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/core/base/slicesx"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
	"github.com/emer/v1vision/v1std"
	"github.com/emer/v1vision/v1vision"
)

// TrialState contains the state for a given trial.
// Trials are processed data-parallel per Step().
type TrialState struct {
	// Cat is the current object category
	Cat string

	// Obj is the current object
	Obj string

	// Image is the rendered image as loaded
	Image image.Image `display:"-"`
}

// Obj3DSacEnv provides the rendered results of the Obj3D + Saccade generator.
type Obj3DSacEnv struct {

	// Name of this environment (Train, Test mode).
	Name string

	// NData is the number of steps to process in data-parallel.
	NData int

	// Path to data.tsv file as rendered, e.g., images/train.
	Path string

	// Trials has NData state per trial for last Step()
	Trials []TrialState

	// Table of generated trial / tick data.
	Table *table.Table

	// EyePop is the 2D population code for gaussian bump rendering
	// of eye position.
	EyePop popcode.TwoD

	// SacPop is the 2d population code for gaussian bump rendering
	// of saccade plan / execution.
	SacPop popcode.TwoD

	// ObjVelPop is the 2d population code for gaussian bump rendering
	// of object velocity.
	ObjVelPop popcode.TwoD

	// V1c has the full set of V1c complex and DoG color contrast filters.
	V1c v1std.V1cMulti

	// Objs is the list of objects, as cat/objfile.
	Objs []string

	// Cats is the list of categories.
	Cats []string

	// TrialCtr counts each object trajectory
	TrialCtr env.Counter `display:"inline"`

	// Tick counts each step along the trajectory
	Tick env.Counter `display:"inline"`

	// current rendered state tensors
	CurStates map[string]*tensor.Float32

	// Rand is the random number generator for the env.
	// All random calls must use this.
	// Set seed here for weight initialization values.
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`
}

func (ev *Obj3DSacEnv) Label() string { return ev.Name }

func (ev *Obj3DSacEnv) Trial(di int) *TrialState {
	return &ev.Trials[di]
}

func (ev *Obj3DSacEnv) Defaults() {

	// images is symlink, e.g., /Users/oreilly/ccn_images/CU3D_100_20obj8inst_8tick4sac
	// https://drive.google.com/drive/folders/13Mi9aUlF1A3sx3JaofX-qzKlxGoViT86?usp=sharing
	// CU3D_100_20obj8inst_8tick4sac.tar
	ev.Path = "images/train"

	ev.EyePop.Defaults()
	ev.EyePop.Min.Set(-1.1, -1.1)
	ev.EyePop.Max.Set(1.1, 1.1)
	ev.EyePop.Sigma.Set(0.1, 0.1)

	ev.SacPop.Defaults()
	ev.SacPop.Min.Set(-0.45, -0.45)
	ev.SacPop.Max.Set(0.45, 0.45)

	ev.ObjVelPop.Defaults()
	ev.ObjVelPop.Min.Set(-0.45, -0.45)
	ev.ObjVelPop.Max.Set(0.45, 0.45)

	ev.V1c.Defaults()
	ev.V1c.SplitColor = false // false > true
	ev.V1c.StdLowMed16DegNoDoG()

	ev.Tick.Max = 8 // important: must be sync'd with actual data
}

func (ev *Obj3DSacEnv) Config(ndata int, netGPU *gpu.GPU) {
	ev.NData = ndata
	v1vision.ComputeGPU = netGPU
	ev.Trials = slicesx.SetLength(ev.Trials, ndata)
	ev.V1c.Config(ndata)

	ev.CurStates = make(map[string]*tensor.Float32)
	ev.CurStates["EyePos"] = tensor.NewFloat32(ndata, 21, 21)
	ev.CurStates["SacPlan"] = tensor.NewFloat32(ndata, 11, 11)
	ev.CurStates["Saccade"] = tensor.NewFloat32(ndata, 11, 11)
	ev.CurStates["ObjVel"] = tensor.NewFloat32(ndata, 11, 11)
}

func (ev *Obj3DSacEnv) Init(run int) {
	ev.RandSeed = int64(73 + run)
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RandSeed)
	} else {
		ev.Rand.Seed(ev.RandSeed)
	}
	ev.TrialCtr.Init()
	ev.TrialCtr.Max = 0
	ev.TrialCtr.Same()
	ev.Tick.Init()
	ev.Tick.Cur = -1
}

// OpenTable loads data.tsv file at Path.
// only do this for one env and copy to the others.
func (ev *Obj3DSacEnv) OpenTable() {
	if ev.Table == nil {
		ev.Table = table.New("obj3dsac_data")
	}
	fnm := filepath.Join(ev.Path, "data.tsv")
	errors.Log(ev.Table.OpenCSV(fsx.Filename(fnm), tensor.Tab))
	errors.Log(jsonx.Open(&ev.Objs, filepath.Join(ev.Path, "objs.json")))
	errors.Log(jsonx.Open(&ev.Cats, filepath.Join(ev.Path, "cats.json")))
}

// OpenImage opens current image.
func (ev *Obj3DSacEnv) OpenImage(row int) (image.Image, error) {
	ifnm := ev.Table.Column("ImgFile").StringRow(row, 0)
	fnm := filepath.Join(ev.Path, ifnm)
	img, _, err := imagex.Open(fnm)
	return img, errors.Log(err)
}

// EncodePops encodes population codes from current row data
func (ev *Obj3DSacEnv) EncodePops(row, di int) {
	val := math32.Vector2{}
	val.X = float32(ev.Table.Column("EyePos").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("EyePos").FloatRow(row, 1))
	ps := ev.CurStates["EyePos"].SubSpace(di).(*tensor.Float32)
	ev.EyePop.Encode(ps, val, popcode.Set)

	ps = ev.CurStates["SacPlan"].SubSpace(di).(*tensor.Float32)
	val.X = float32(ev.Table.Column("SacPlan").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("SacPlan").FloatRow(row, 1))
	ev.SacPop.Encode(ps, val, popcode.Set)

	ps = ev.CurStates["Saccade"].SubSpace(di).(*tensor.Float32)
	val.X = float32(ev.Table.Column("Saccade").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("Saccade").FloatRow(row, 1))
	ev.SacPop.Encode(ps, val, popcode.Set)

	ps = ev.CurStates["ObjVel"].SubSpace(di).(*tensor.Float32)
	val.X = float32(ev.Table.Column("ObjVel").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("ObjVel").FloatRow(row, 1))
	ev.ObjVelPop.Encode(ps, val, popcode.Set)
}

// SetCtrs sets ctrs from current row data, returns the current row
func (ev *Obj3DSacEnv) SetCtrs(st *TrialState, di int) int {
	row := (ev.TrialCtr.Cur+di)*ev.Tick.Max + ev.Tick.Cur
	row = row % ev.Table.NumRows()
	// trial := ev.Table.Column("Trial").IntRow(row, 0)
	// if ev.TrialCtr.Cur+di != trial { // note: this is expected after first epoch!
	// 	fmt.Println("error trial mismatch:", row, ev.TrialCtr.Cur+di, trial)
	// }
	tick := ev.Table.Column("Tick").IntRow(row, 0)
	if ev.Tick.Cur != tick {
		fmt.Println("error tick mismatch:", row, ev.Tick.Cur, tick)
	}

	st.Cat = ev.Table.Column("Cat").StringRow(row, 0)
	st.Obj = ev.Table.Column("Obj").StringRow(row, 0)
	return row
}

func (ev *Obj3DSacEnv) String() string {
	return ev.TrialName(0)
}

// TrialName returns the string rep of the env state
func (ev *Obj3DSacEnv) TrialName(di int) string {
	st := ev.Trial(di)
	return fmt.Sprintf("%s:%s_%d", st.Cat, st.Obj, ev.Tick.Cur)
}

func (ev *Obj3DSacEnv) Step() bool {
	if ev.Tick.Incr() {
		ev.TrialCtr.Cur += ev.NData
	}
	imgs := make([]image.Image, ev.NData)
	for di := range ev.NData {
		st := ev.Trial(di)
		row := ev.SetCtrs(st, di)
		ev.EncodePops(row, di)
		img, err := ev.OpenImage(row)
		if err != nil {
			continue
		}
		imgs[di] = img
	}
	ev.V1c.RunImages(imgs...)
	return true
}

func (ev *Obj3DSacEnv) State(element string) tensor.Values {
	switch element {
	case "V1m": // todo: L and M actually
		return &ev.V1c.V1cParams[0].Output
	case "V1h":
		return &ev.V1c.V1cParams[1].Output
	default:
		return ev.CurStates[element]
	}
}

func (ev *Obj3DSacEnv) Action(element string, input tensor.Values) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*Obj3DSacEnv)(nil)
