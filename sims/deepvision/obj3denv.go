// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

import (
	"fmt"
	"image"
	"log"
	"path/filepath"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

// Obj3DSacEnv provides the rendered results of the Obj3D + Saccade generator.
type Obj3DSacEnv struct {

	// Name of this environment (Train, Test mode).
	Name string

	// Path to data.tsv file as rendered, e.g., images/train.
	Path string

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

	// image that we operate upon -- one image shared among all filters
	Img V1Img

	// V1Med is the v1 medium resolution filtering of image -- V1AllTsr has result.
	V1Med Vis

	// V1Hi is the v1 higher resolution filtering of image -- V1AllTsr has result.
	V1Hi Vis

	// Objs is the list of objects, as cat/objfile.
	Objs []string

	// Cats is the list of categories.
	Cats []string

	// Trial counts each object trajectory
	Trial env.Counter `display:"inline"`

	// Tick counts each step along the trajectory
	Tick env.Counter `display:"inline"`

	// Row of table -- this is actual counter driving everything
	Row env.Counter `display:"inline"`

	// Number of data-parallel environments.
	NData int

	// data-parallel index of this env.
	Di int

	// CurCat is the current object category
	CurCat string

	// CurObj is the current object
	CurObj string

	// current rendered state tensors
	CurStates map[string]*tensor.Float32

	//  Image is the rendered image as loaded
	Image image.Image `display:"-"`

	// Rand is the random number generator for the env.
	// All random calls must use this.
	// Set seed here for weight initialization values.
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`
}

func (ev *Obj3DSacEnv) Label() string { return ev.Name }

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

	ev.Img.Defaults()
	ev.V1Med.Defaults(24, 8, &ev.Img)
	ev.V1Hi.Defaults(12, 4, &ev.Img)

	ev.Tick.Max = 8 // important: must be sync'd with actual data
}

func (ev *Obj3DSacEnv) Config() {
	ev.CurStates = make(map[string]*tensor.Float32)

	ev.CurStates["EyePos"] = tensor.NewFloat32(21, 21)
	ev.CurStates["SacPlan"] = tensor.NewFloat32(11, 11)
	ev.CurStates["Saccade"] = tensor.NewFloat32(11, 11)
	ev.CurStates["ObjVel"] = tensor.NewFloat32(11, 11)
}

func (ev *Obj3DSacEnv) Init(run int) {
	ev.Trial.Init()
	ev.Trial.Cur = ev.Di
	ev.Trial.Max = 0
	ev.Trial.Same()
	ev.Tick.Init()
	ev.Tick.Cur = -1
	ev.Row.Cur = ev.Tick.Max * ev.Di
}

// OpenTable loads data.tsv file at Path.
// only do this for one env and copy to the others.
func (ev *Obj3DSacEnv) OpenTable() {
	if ev.Table == nil {
		ev.Table = table.New("obj3dsac_data")
	}
	fnm := filepath.Join(ev.Path, "data.tsv")
	errors.Log(ev.Table.OpenCSV(fsx.Filename(fnm), tensor.Tab))
	ev.Row.Max = ev.Table.NumRows()
	errors.Log(jsonx.Open(&ev.Objs, filepath.Join(ev.Path, "objs.json")))
	errors.Log(jsonx.Open(&ev.Cats, filepath.Join(ev.Path, "cats.json")))
}

// CurRow returns current row in table, ensuring table is updated.
func (ev *Obj3DSacEnv) CurRow() int {
	if ev.Row.Cur >= ev.Table.NumRows() {
		ev.Row.Max = ev.Table.NumRows()
		ev.Row.Cur = 0
	}
	return ev.Row.Cur
}

// OpenImage opens current image.
func (ev *Obj3DSacEnv) OpenImage() error {
	row := ev.CurRow()
	ifnm := ev.Table.Column("ImgFile").StringRow(row, 0)
	fnm := filepath.Join(ev.Path, ifnm)
	var err error
	ev.Image, _, err = imagex.Open(fnm)
	if err != nil {
		log.Println(err)
	}
	return err
}

// FilterImage opens and filters current image
func (ev *Obj3DSacEnv) FilterImage() error {
	err := ev.OpenImage()
	if err != nil {
		return err
	}
	ev.Img.SetImage(ev.Image, ev.V1Med.V1sGeom.FiltRt.X)
	ev.V1Med.Filter()
	ev.V1Hi.Filter()
	return nil
}

// EncodePops encodes population codes from current row data
func (ev *Obj3DSacEnv) EncodePops() {
	row := ev.CurRow()
	val := math32.Vector2{}
	val.X = float32(ev.Table.Column("EyePos").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("EyePos").FloatRow(row, 1))
	ev.EyePop.Encode(ev.CurStates["EyePos"], val, popcode.Set)

	val.X = float32(ev.Table.Column("SacPlan").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("SacPlan").FloatRow(row, 1))
	ev.SacPop.Encode(ev.CurStates["SacPlan"], val, popcode.Set)

	val.X = float32(ev.Table.Column("Saccade").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("Saccade").FloatRow(row, 1))
	ev.SacPop.Encode(ev.CurStates["Saccade"], val, popcode.Set)

	val.X = float32(ev.Table.Column("ObjVel").FloatRow(row, 0))
	val.Y = float32(ev.Table.Column("ObjVel").FloatRow(row, 1))
	ev.ObjVelPop.Encode(ev.CurStates["ObjVel"], val, popcode.Set)
}

// SetCtrs sets ctrs from current row data
func (ev *Obj3DSacEnv) SetCtrs() {
	row := ev.CurRow()
	trial := ev.Table.Column("Trial").IntRow(row, 0)
	ev.Trial.Set(trial)
	tick := ev.Table.Column("Tick").IntRow(row, 0)
	ev.Tick.Set(tick)

	ev.CurCat = ev.Table.Column("Cat").StringRow(row, 0)
	ev.CurObj = ev.Table.Column("Obj").StringRow(row, 0)
}

func (ev *Obj3DSacEnv) String() string {
	return fmt.Sprintf("%s:%s_%d", ev.CurCat, ev.CurObj, ev.Tick.Cur)
}

func (ev *Obj3DSacEnv) Step() bool {
	ev.Trial.Same()
	if ev.Tick.Incr() {
		ev.Trial.Cur += ev.NData
	}
	ev.Row.Cur = ev.Trial.Cur*ev.Tick.Max + ev.Tick.Cur
	ev.SetCtrs()
	ev.EncodePops()
	ev.FilterImage()

	return true
}

func (ev *Obj3DSacEnv) State(element string) tensor.Values {
	switch element {
	case "V1m":
		return &ev.V1Med.V1AllTsr
	case "V1h":
		return &ev.V1Hi.V1AllTsr
	default:
		return ev.CurStates[element]
	}
}

func (ev *Obj3DSacEnv) Action(element string, input tensor.Values) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*Obj3DSacEnv)(nil)
