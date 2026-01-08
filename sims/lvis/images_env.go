// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lvis

import (
	"fmt"
	"image"
	"path/filepath"
	"sort"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/core/base/slicesx"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensor/tensormpi"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/v1vision/v1std"
	"github.com/emer/v1vision/v1vision"
	"golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// TrialState contains the state for a given trial.
// Trials are processed data-parallel per Step().
type TrialState struct {
	// current category
	Cat string

	// index of current category
	CatIdx int

	// current image
	Img string

	// current translation
	Trans math32.Vector2

	// current scaling
	Scale float32

	// current rotation
	Rot float32

	// rendered image as loaded
	Image image.Image `display:"-"`
}

func (st *TrialState) String() string {
	return fmt.Sprintf("%s:%s", st.Cat, st.Img)
}

// ImagesEnv provides the rendered results of the Obj3D + Saccade generator.
type ImagesEnv struct {

	// Name of this environment (Train, Test mode).
	Name string

	// NData is the number of steps to process in data-parallel.
	NData int

	// Trials has NData state per trial for last Step()
	Trials []TrialState

	// image file name
	ImageFile string

	// present test items, else train
	Test bool

	// present items in sequential order -- else shuffled
	Sequential bool

	// compute high-res full field filtering. Off by default
	High16 bool

	// compute color DoG (blob) filtering. On by default.
	ColorDoG bool

	// images list
	Images Images

	// def 0.3 maximum amount of translation as proportion of half-width size in each direction -- 1 = something in center is now at right edge
	TransMax math32.Vector2

	// [def: 0.15] if > 0, generate translations using gaussian normal distribution with this standard deviation, and then clip to TransMax range -- this facilitates learning on the central region while still giving exposure to wider area.  Tyically turn off for last 100 epochs to measure true uniform distribution performance.
	TransSigma float32

	// def 0.5 - 1.1 range of scale
	ScaleRange minmax.F32

	// maximum degrees of rotation in plane.
	// image is rotated plus or minus in this range
	RotateMax float32 `default:"8"`

	// V1c has the full set of V1c complex and DoG color contrast filters.
	V1c v1std.V1cMulti

	// maximum number of output categories representable here
	MaxOut int

	// use random bit patterns instead of localist output units
	OutRandom bool

	// proportion activity for random patterns
	RndPctOn float32

	// proportion minimum difference for random patterns
	RndMinDiff float32

	// the output tensor geometry -- must be >= number of cats
	OutSize vecint.Vector2i

	// number of output units per category.
	// Spiking may benefit from replication; is Y inner dim of output tensor
	NOutPer int

	// output patterns: either localist or random
	Pats table.Table `display:"no-inline"`

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `display:"-"`

	// random seed
	RndSeed int64 `edit"-"`

	// output pattern for current item
	Output tensor.Float32

	// starting row, e.g., for mpi allocation across processors
	StRow int

	// ending row, ignored if 0
	EdRow int

	// suffled list of entire set of images -- re-shuffle every time through imgidxs
	Shuffle []int

	// indexes of images to present -- from StRow to EdRow
	ImgIdxs []int

	// Row of item list  -- this is actual counter driving everything
	Row env.Counter `display:"inline"`
}

func (ev *ImagesEnv) Label() string { return ev.Name }

func (ev *ImagesEnv) Trial(di int) *TrialState {
	return &ev.Trials[di]
}

func (ev *ImagesEnv) Defaults() {
	ev.TransSigma = 0
	// hard:
	ev.TransMax.Set(0.3, 0.3)   // 0.2 easy, 0.3 hard
	ev.ScaleRange.Set(0.7, 1.2) // 0.8, 1.1 easy, .7-1.2 hard
	ev.RotateMax = 16           // 8 easy, 16 hard
	// easy:
	// ev.TransMax.Set(0.2, 0.2)
	// ev.ScaleRange.Set(0.8, 1.1)
	// ev.RotateMax = 8
	ev.RndPctOn = 0.2
	ev.RndMinDiff = 0.5
	ev.NOutPer = 5
	ev.V1c.Defaults()
	// ev.V1c.GPU = false
	// todo: ev.High16, ColorDoG options.
	ev.V1c.SplitColor = true // split is better!

	ev.V1c.StdLowMed16DegZoom1()
}

// ImageList returns the list of images -- train or test
func (ev *ImagesEnv) ImageList() []string {
	if ev.Test {
		return ev.Images.FlatTest
	}
	return ev.Images.FlatTrain
}

// MPIAlloc allocate objects based on mpi processor number
func (ev *ImagesEnv) MPIAlloc() {
	ws := mpi.WorldSize()
	nim := ws * (len(ev.ImageList()) / ws) // even multiple of size -- few at end are lost..
	ev.StRow, ev.EdRow, _ = tensormpi.AllocN(nim)
	// mpi.PrintAllProcs = true
	// mpi.Printf("allocated images: n: %d st: %d ed: %d\n", nim, ev.StRow, ev.EdRow)
	// mpi.PrintAllProcs = false
}

func (ev *ImagesEnv) Config(ndata int, netGPU *gpu.GPU) {
	ev.NData = ndata
	v1vision.ComputeGPU = netGPU
	ev.Trials = slicesx.SetLength(ev.Trials, ndata)
	ev.V1c.Config(ndata)
}

func (ev *ImagesEnv) Init(run int) {
	ev.RndSeed = int64(73 + run)
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RndSeed)
	} else {
		ev.Rand.Seed(ev.RndSeed)
	}
	ev.Row.Cur = -1 // init state -- key so that first Step() = 0
	nitm := len(ev.ImageList())
	if ev.EdRow > 0 {
		ev.EdRow = min(ev.EdRow, nitm)
		ev.ImgIdxs = make([]int, ev.EdRow-ev.StRow)
	} else {
		ev.ImgIdxs = make([]int, nitm)
	}
	for i := range ev.ImgIdxs {
		ev.ImgIdxs[i] = ev.StRow + i
	}
	ev.Shuffle = ev.Rand.Perm(nitm)
	ev.Row.Max = len(ev.ImgIdxs)
	nc := len(ev.Images.Cats)
	ev.MaxOut = max(nc, ev.MaxOut)
	ev.ConfigPats()
}

// OpenConfig opens saved configuration for current images: config files are required,
// and an error is logged and returned if not present.
func (ev *ImagesEnv) OpenConfig() error {
	cfnm := fmt.Sprintf("%s_cats.json", ev.ImageFile)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.ImageFile, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.ImageFile, ev.Images.NTestPerCat)
	if errors.Log1(fsx.FileExistsFS(embedfs, cfnm)) {
		errors.Log(jsonx.OpenFS(&ev.Images.Cats, embedfs, cfnm))
		errors.Log(jsonx.OpenFS(&ev.Images.ImagesTest, embedfs, tsfnm))
		errors.Log(jsonx.OpenFS(&ev.Images.ImagesTrain, embedfs, trfnm))
		ev.Images.ToTrainAll()
		ev.Images.Flats()
		return nil
	}
	return errors.Log(errors.New("ImagesEnv.OpenConfig: Required Cats config file not found: " + cfnm))
}

// SaveConfig saves configuration for current images
func (ev *ImagesEnv) SaveConfig() {
	cfnm := fmt.Sprintf("%s_cats.json", ev.ImageFile)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.ImageFile, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.ImageFile, ev.Images.NTestPerCat)
	errors.Log(jsonx.Save(ev.Images.Cats, filepath.Join("..", cfnm)))
	errors.Log(jsonx.Save(ev.Images.ImagesTest, filepath.Join("..", tsfnm)))
	errors.Log(jsonx.Save(ev.Images.ImagesTrain, filepath.Join("..", trfnm)))
}

// ConfigPats configures the output patterns
func (ev *ImagesEnv) ConfigPats() {
	if ev.OutRandom {
		ev.ConfigPatsRandom()
	} else {
		ev.ConfigPatsLocalist2D()
	}
}

// ConfigPatsName names the patterns
func (ev *ImagesEnv) ConfigPatsName() {
	// for i := 0; i < ev.MaxOut; i++ {
	// 	nm := fmt.Sprintf("P%03d", i)
	// 	if i < len(ev.Images.Cats) {
	// 		nm = ev.Images.Cats[i]
	// 	}
	// 	ev.Pats.SetCellString("Name", i, nm)
	// }
}

// ConfigPatsLocalistPools configures the output patterns: localist case
// with pools for each sub-pool
func (ev *ImagesEnv) ConfigPatsLocalistPools() {
	oshp := []int{ev.OutSize.Y, ev.OutSize.X, ev.NOutPer, 1}
	ev.Output.SetShapeSizes(ev.NData, ev.OutSize.Y, ev.OutSize.X, ev.NOutPer, 1)
	ev.Pats.AddStringColumn("Name")
	out := ev.Pats.AddFloat32Column("Output", oshp...)
	ev.Pats.SetNumRows(ev.MaxOut)
	for pi := range ev.MaxOut {
		op := out.SubSpace(pi)
		si := ev.NOutPer * pi
		for i := range ev.NOutPer {
			op.SetFloat1D(1, si+i)
		}
	}
	ev.ConfigPatsName()
}

// ConfigPatsLocalist2D configures the output patterns: localist case
// as an overall 2D layer -- NOutPer goes along X axis to be contiguous
func (ev *ImagesEnv) ConfigPatsLocalist2D() {
	oshp := []int{ev.OutSize.Y, ev.OutSize.X * ev.NOutPer}
	ev.Output.SetShapeSizes(ev.NData, ev.OutSize.Y, ev.OutSize.X*ev.NOutPer)
	ev.Pats.Init()
	ev.Pats.AddStringColumn("Name")
	out := ev.Pats.AddFloat32Column("Output", oshp...)
	ev.Pats.SetNumRows(ev.MaxOut)
	for pi := range ev.MaxOut {
		op := out.SubSpace(pi)
		si := ev.NOutPer * pi
		for i := range ev.NOutPer {
			op.SetFloat1D(1, si+i)
		}
	}
	ev.ConfigPatsName()
}

// ConfigPatsRandom configures the output patterns: random case
func (ev *ImagesEnv) ConfigPatsRandom() {
	// oshp := []int{ev.OutSize.Y, ev.OutSize.X}
	// oshpnm := []string{"Y", "X"}
	// ev.Output.SetShape(oshp, nil, oshpnm)
	// sch := table.Schema{
	// 	{"Name", tensor.STRING, nil, nil},
	// 	{"Output", tensor.FLOAT32, oshp, oshpnm},
	// }
	// ev.Pats.SetFromSchema(sch, ev.MaxOut)
	// np := ev.OutSize.X * ev.OutSize.Y
	// nOn := patgen.NFmPct(ev.RndPctOn, np)
	// minDiff := patgen.NFmPct(ev.RndMinDiff, nOn)
	// fnm := fmt.Sprintf("rndpats_%dx%d_n%d_on%d_df%d.tsv", ev.OutSize.X, ev.OutSize.Y, ev.MaxOut, nOn, minDiff)
	// _, err := os.Stat(fnm)
	// if !os.IsNotExist(err) {
	// 	ev.Pats.OpenCSV(core.FileName(fnm), table.Tab)
	// } else {
	// 	out := ev.Pats.Col(1).(*tensor.Float32)
	// 	patgen.PermutedBinaryMinDiff(out, nOn, 1, 0, minDiff)
	// 	ev.ConfigPatsName()
	// 	ev.Pats.SaveCSV(core.FileName(fnm), table.Tab, table.Headers)
	// }
}

// NewShuffle generates a new random order of items to present
func (ev *ImagesEnv) NewShuffle() {
	randx.PermuteInts(ev.Shuffle, &ev.Rand)
}

// CurImage returns current image based on row and
func (ev *ImagesEnv) CurImage(st *TrialState) string {
	il := ev.ImageList()
	sz := len(ev.ImgIdxs)
	if ev.Row.Cur >= sz {
		ev.Row.Max = sz
		ev.Row.Cur = 0
		ev.NewShuffle()
	}
	r := ev.Row.Cur
	if r < 0 {
		r = 0
	}
	i := ev.ImgIdxs[r]
	if !ev.Sequential {
		i = ev.Shuffle[i]
	}
	st.Img = il[i]
	st.Cat = ev.Images.Cat(st.Img)
	st.CatIdx = ev.Images.CatMap[st.Cat]
	return st.Img
}

// OpenImage opens current image
func (ev *ImagesEnv) OpenImage(st *TrialState) (image.Image, error) {
	imgNm := ev.CurImage(st)
	fnm := filepath.Join(ev.Images.Path, imgNm)
	img, _, err := imagex.Open(fnm)
	return img, errors.Log(err)
}

// RandTransforms generates random transforms
func (ev *ImagesEnv) RandTransforms(st *TrialState) {
	if ev.TransSigma > 0 {
		st.Trans.X = float32(randx.GaussianGen(0, float64(ev.TransSigma), &ev.Rand))
		st.Trans.X = math32.Clamp(st.Trans.X, -ev.TransMax.X, ev.TransMax.X)
		st.Trans.Y = float32(randx.GaussianGen(0, float64(ev.TransSigma), &ev.Rand))
		st.Trans.Y = math32.Clamp(st.Trans.Y, -ev.TransMax.Y, ev.TransMax.Y)
	} else {
		st.Trans.X = (ev.Rand.Float32()*2 - 1) * ev.TransMax.X
		st.Trans.Y = (ev.Rand.Float32()*2 - 1) * ev.TransMax.Y
	}
	st.Scale = ev.ScaleRange.Min + ev.ScaleRange.Range()*ev.Rand.Float32()
	st.Rot = (ev.Rand.Float32()*2 - 1) * ev.RotateMax
}

// TransformImage transforms the image according to current translation and scaling
func (ev *ImagesEnv) TransformImage(st *TrialState, img image.Image) image.Image {
	s := math32.FromPoint(img.Bounds().Size())
	transformer := draw.BiLinear
	tx := 0.5 * st.Trans.X * s.X
	ty := 0.5 * st.Trans.Y * s.Y
	m := math32.Translate2D(s.X*.5+tx, s.Y*.5+ty).Scale(st.Scale, st.Scale).Rotate(math32.DegToRad(st.Rot)).Translate(-s.X*.5, -s.Y*.5)
	s2d := f64.Aff3{float64(m.XX), float64(m.XY), float64(m.X0), float64(m.YX), float64(m.YY), float64(m.Y0)}

	// use first color in upper left as fill color
	clr := img.At(0, 0)
	dst := image.NewRGBA(img.Bounds())
	src := image.NewUniform(clr)
	draw.Draw(dst, dst.Bounds(), src, image.ZP, draw.Src)

	transformer.Transform(dst, s2d, img, img.Bounds(), draw.Over, nil) // Over superimposes over bg
	return dst
}

// SetOutput sets output by category
func (ev *ImagesEnv) SetOutput(di, out int) {
	ot := ev.Output.SubSpace(di).(*tensor.Float32)
	ot.SetZeros()
	otc := ev.Pats.Column("Output").SubSpace(out)
	ot.CopyCellsFrom(otc, 0, 0, ot.Len())
}

// FloatIdx32 contains a float32 value and its index
type FloatIdx32 struct {
	Val float32
	Idx int
}

// ClosestRows32 returns the sorted list of distances from probe pattern
// and patterns in an tensor.Float32 where the outer-most dimension is
// assumed to be a row (e.g., as a column in an etable), using the given metric function,
// *which must have the Increasing property* -- i.e., larger = further.
// Col cell sizes must match size of probe (panics if not).
func ClosestRows32(probe *tensor.Float64, col *tensor.Float32, mfun metric.Metrics) []FloatIdx32 {
	pr1d := tensor.As1D(probe)
	rows := col.DimSize(0)
	csz := col.Len() / rows
	if csz != probe.Len() {
		panic("metric.ClosestRows32: probe size != cell size of tensor column!\n")
	}
	dsts := make([]FloatIdx32, rows)
	for ri := 0; ri < rows; ri++ {
		st := ri * csz
		rvals := col.Values[st : st+csz]
		v := mfun.Call(pr1d, tensor.NewFloat32FromValues(rvals...))
		dsts[ri].Val = float32(v.Float1D(0))
		dsts[ri].Idx = ri
	}
	sort.Slice(dsts, func(i, j int) bool {
		return dsts[i].Val < dsts[j].Val
	})
	return dsts
}

// OutErr scores the output activity of network, returning the index of
// item with closest fit to given pattern, and 1 if that is error, 0 if correct.
// also returns a top-two error: if 2nd closest pattern was correct.
func (ev *ImagesEnv) OutErr(tsr *tensor.Float64, curCatIdx int) (maxi int, err, err2 float64) {
	ocol := ev.Pats.Column("Output").Tensor.(*tensor.Float32)
	dsts := ClosestRows32(tsr, ocol, metric.MetricInvCorrelation)
	maxi = dsts[0].Idx
	err = 1.0
	if maxi == curCatIdx {
		err = 0
	}
	err2 = err
	if dsts[1].Idx == curCatIdx {
		err2 = 0
	}
	return
}

func (ev *ImagesEnv) String() string {
	return ev.TrialName(0)
}

// TrialName returns the string rep of the env state
func (ev *ImagesEnv) TrialName(di int) string {
	st := ev.Trial(di)
	return st.String()
}

func (ev *ImagesEnv) Step() bool {
	imgs := make([]image.Image, ev.NData)
	for di := range ev.NData {
		st := ev.Trial(di)
		if ev.Row.Incr() {
			ev.NewShuffle()
		}
		ev.RandTransforms(st)
		img, err := ev.OpenImage(st)
		if err != nil {
			continue
		}
		img = ev.TransformImage(st, img)
		st.Image = img
		imgs[di] = img
		ev.SetOutput(di, st.CatIdx)
	}
	ev.V1c.RunImages(imgs...)
	return true
}

func (ev *ImagesEnv) State(element string) tensor.Values {
	switch element {
	case "V1l16":
		return &ev.V1c.V1cParams[0].Output
	case "V1m16":
		return &ev.V1c.V1cParams[1].Output
	// case "V1h16": // todo
	// 	return &ev.V1h16.V1AllTsr
	case "V1l8":
		return &ev.V1c.V1cParams[2].Output
	case "V1m8":
		return &ev.V1c.V1cParams[3].Output
	case "V1Cl16":
		return &ev.V1c.DoGParams[0].Output
	case "V1Cm16":
		return &ev.V1c.DoGParams[1].Output
	case "V1Cl8":
		return &ev.V1c.DoGParams[2].Output
	case "V1Cm8":
		return &ev.V1c.DoGParams[3].Output
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *ImagesEnv) Action(action string, input tensor.Values) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImagesEnv)(nil)
