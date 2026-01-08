// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"image"
	"image/color"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/colors"
	"cogentcore.org/core/colors/colormap"
	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/styles"
	"cogentcore.org/core/styles/abilities"
	"cogentcore.org/core/tree"
	"cogentcore.org/core/xyz"
	"cogentcore.org/core/xyz/xyzcore"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/plotcore"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensorcore"
	"github.com/emer/axon/v2/axon"
)

// GUI renders multiple views of the flat world env
type GUI struct {

	// update display -- turn off to make it faster
	Disp bool

	// the env being visualized
	Env *Env

	// name of current env -- number is NData index
	EnvName string

	// 3D visualization of the Scene
	SceneEditor *xyzcore.SceneEditor

	// list of material colors
	MatColors []string

	// internal state colors
	StateColors map[string]string

	// current internal / behavioral state
	State TraceStates

	// trace record of recent activity
	Trace StateTrace

	// view of the gui obj
	EnvForm *core.Form `display:"-"`

	// offscreen render camera settings
	Camera phyxyz.Camera

	// first-person right-eye full field view
	EyeRFullImage *core.Image `display:"-"`

	// first-person right-eye fovea view
	EyeRFovImage *core.Image `display:"-"`

	// plot of positive valence drives, active OFC US state, and reward
	USposPlot *plotcore.Editor

	// data for USPlot
	USposData *table.Table

	// plot of negative valence active OFC US state, and outcomes
	USnegPlot *plotcore.Editor

	// data for USPlot
	USnegData *table.Table

	// Emery state
	Emery Emery `new-window:"+"`

	// Maze state
	Maze Maze `new-window:"+"`

	// The core physics elements: Model, Builder, Scene
	Physics builder.Physics
}

// ConfigGUI configures all the world view GUI elements
// pass an initial env to use for configuring
func (vw *GUI) ConfigGUI(ev *Env, b core.Widget) {
	vw.Disp = true
	vw.Env = ev
	vw.EnvName = ev.Name
	vw.Camera.Defaults()
	vw.Camera.FOV = 90

	vw.StateColors = map[string]string{
		"TrSearching":   "aqua",
		"TrDeciding":    "coral",
		"TrJustEngaged": "yellow",
		"TrApproaching": "cornflowerblue",
		"TrConsuming":   "purple",
		"TrRewarded":    "green",
		"TrGiveUp":      "black",
		"TrBumping":     "red",
	}
	vw.MatColors = []string{"blue", "orange", "red", "violet", "navy", "brown", "pink", "purple", "olive", "chartreuse", "cyan", "magenta", "salmon", "goldenrod", "SykBlue"}

	core.NewToolbar(b).Maker(vw.MakeToolbar)
	split := core.NewSplits(b)

	fr := core.NewFrame(split)
	fr.Styler(func(s *styles.Style) {
		s.Direction = styles.Column
	})

	// vw.EnvForm = core.NewForm(fr).SetStruct(vw)
	core.Bind(&vw.Disp, core.NewSwitch(fr).SetText("Display"))
	imfr := core.NewFrame(fr)
	imfr.Styler(func(s *styles.Style) {
		s.Display = styles.Grid
		s.Columns = 2
		s.Grow.Set(0, 0)
	})
	core.NewText(imfr).SetText("Eye-View, Fovea:")
	core.NewText(imfr).SetText("Full Field:")

	vw.EyeRFovImage = core.NewImage(imfr)
	vw.EyeRFovImage.Name = "eye-r-fov-img"
	vw.EyeRFovImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})

	vw.EyeRFullImage = core.NewImage(imfr)
	vw.EyeRFullImage.Name = "eye-r-full-img"
	vw.EyeRFullImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})

	wd := float32(420)
	ht := float32(120)
	vw.USposPlot = plotcore.NewEditor(fr)
	vw.USposPlot.Name = "us-pos"
	vw.USposPlot.Styler(func(s *styles.Style) {
		s.Max.X.Px(wd)
		s.Max.Y.Px(ht)
	})

	vw.USnegPlot = plotcore.NewEditor(fr)
	vw.USnegPlot.Name = "us-neg"
	vw.USnegPlot.Styler(func(s *styles.Style) {
		s.Max.X.Px(wd)
		s.Max.Y.Px(ht)
	})
	vw.ConfigUSPlots()

	vw.SceneEditor = xyzcore.NewSceneEditor(split)
	vw.SceneEditor.UpdateWidget()
	sc := vw.SceneEditor.SceneXYZ()
	vw.MakeModel(sc)

	sc.Camera.Pose.Pos = math32.Vec3(0, 29, -4)
	sc.Camera.LookAt(math32.Vec3(0, 4, -5), math32.Vec3(0, 1, 0))
	sc.SaveCamera("2")

	sc.Camera.Pose.Pos = math32.Vec3(0, 24, 32)
	sc.Camera.LookAt(math32.Vec3(0, 3.6, 0), math32.Vec3(0, 1, 0))
	sc.SaveCamera("1")
	sc.SaveCamera("default")

	split.SetSplits(.4, .6)
}

func (vw *GUI) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.Button) {
		w.SetText("Init").SetIcon(icons.ClearAll).
			SetTooltip("Init env").
			OnClick(func(e events.Event) {
				vw.Env.Init(0)
			})
	})
	tree.Add(p, func(w *core.Button) {
		w.SetText("Reset Trace").SetIcon(icons.Undo).
			SetTooltip("Reset trace of position, etc, shown in 2D View").
			OnClick(func(e events.Event) {
				vw.Trace = nil
			})
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(vw.Forward).SetText("Fwd").SetIcon(icons.SkipNext).
			Styler(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(vw.Left).SetText("Left").SetIcon(icons.KeyboardArrowLeft).
			Styler(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(vw.Right).SetText("Right").SetIcon(icons.KeyboardArrowRight).
			Styler(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(vw.Consume).SetText("Consume").SetIcon(icons.SentimentExcited).
			Styler(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
	})
}

// MakeModel makes the physics Model
func (vw *GUI) MakeModel(sc *xyz.Scene) {
	vw.Physics.Model = physics.NewModel()
	vw.Physics.Builder = builder.NewBuilder()
	vw.Physics.Model.GPU = false

	sc.Background = colors.Scheme.Select.Container
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)

	dir := xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	vw.Physics.Scene = phyxyz.NewScene(sc)

	vw.Maze.Config(vw.Env.Config.NArms, vw.Env.MaxLength)
	wl := vw.Physics.Builder.NewGlobalWorld()
	vw.Maze.Make(wl, vw.Physics.Scene, vw)

	ew := vw.Physics.Builder.NewWorld()
	vw.Emery.Make(ew, vw.Physics.Scene, vw)
	vw.Physics.Builder.ReplicateWorld(vw.Physics.Scene, 1, 1, vw.Env.NData)
	vw.Physics.Build()
}

func (vw *GUI) ConfigUSPlots() {
	dp := table.New()
	vw.USposData = dp
	// plot.SetStyler(dp.AddStringColumn("US"), func(s *plot.Style) {
	// 	s.Plotter = "Bar"
	// 	s.Role = plot.X
	// })
	ysty := func(s *plot.Style) {
		s.Plotter = "Bar"
		s.Role = plot.Y
		s.On = true
		s.NoLegend = true
		s.Range.SetMin(0).SetMax(1)
	}
	plot.SetStyler(dp.AddFloat64Column("Drive"), func(s *plot.Style) {
		s.Plot.Title = "Positive USs"
		ysty(s)
	})
	plot.SetStyler(dp.AddFloat64Column("OFC"), ysty)
	plot.SetStyler(dp.AddFloat64Column("USin"), ysty)
	dp.SetNumRows(vw.Env.Config.NDrives + 1)

	dn := table.New()
	vw.USnegData = dn
	// plot.SetStyler(dn.AddStringColumn("US"), func(s *plot.Style) {
	// 	s.Plotter = "Bar"
	// 	s.Role = plot.X
	// })
	plot.SetStyler(dn.AddFloat64Column("OFC"), func(s *plot.Style) {
		s.Plot.Title = "Negative USs"
		ysty(s)
	})
	plot.SetStyler(dn.AddFloat64Column("USin"), ysty)
	dn.SetNumRows(vw.Env.Config.NNegUSs + 2)

	vw.USposPlot.SetTable(dp)
	vw.USnegPlot.SetTable(dn)
}

// GrabEyeImg takes a snapshot from the perspective of Emer's right eye
func (vw *GUI) GrabEyeImg() {
	vw.Camera.FOV = 90
	img := vw.Physics.Scene.RenderFrom(vw.Emery.EyeR.Skin, &vw.Camera)[0]
	if img == nil {
		return
	}
	vw.EyeRFullImage.SetImage(img)
	vw.EyeRFullImage.NeedsRender()

	vw.Camera.FOV = 10
	img = vw.Physics.Scene.RenderFrom(vw.Emery.EyeR.Skin, &vw.Camera)[0]
	if img == nil {
		return
	}
	vw.EyeRFovImage.SetImage(img)
	vw.EyeRFovImage.NeedsRender()
}

func (vw *GUI) ConfigWorldView(tg *tensorcore.TensorGrid) {
	cnm := "ArmMazeColors"
	cm, ok := colormap.AvailableMaps[cnm]
	if !ok {
		ev := vw.Env
		cm = &colormap.Map{}
		cm.Name = cnm
		cm.Indexed = true
		nc := ev.Config.NArms
		cm.Colors = make([]color.RGBA, nc)
		cm.NoColor = colors.Black
		for i, cnm := range vw.MatColors {
			cm.Colors[i] = errors.Log1(colors.FromString(cnm))
		}
		colormap.AvailableMaps[cnm] = cm
	}
	tensorcore.AddGridStylerTo(tg, func(s *tensorcore.GridStyle) {
		s.ColorMap = core.ColorMapName(cnm)
		s.GridFill = 1
	})
}

func (vw *GUI) UpdateWorld(ctx *axon.Context, ev *Env, net *axon.Network, state TraceStates) {
	vw.State = state
	vw.Trace.AddRec(ctx, uint32(ev.Di), ev, net, state)
	if vw.SceneEditor == nil || !vw.Disp {
		return
	}

	if vw.Env != ev {
		vw.Env = ev
		vw.EnvName = ev.Name
		vw.Trace = nil
		vw.EnvForm.Update()
	}

	vw.UpdateGUI()
}

func (vw *GUI) UpdateGUI() {
	if vw.SceneEditor == nil || !vw.Disp {
		return
	}
	vw.Physics.Scene.Update()
	vw.GrabEyeImg()
	if vw.SceneEditor.IsVisible() {
		vw.SceneEditor.NeedsRender()
	}
}

func (vw *GUI) StateColor() string {
	return vw.StateColors[vw.State.String()]
}

func (vw *GUI) Left() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Left", nil)
	ev.Step()
	vw.UpdateGUI()
}

func (vw *GUI) Right() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Right", nil)
	ev.Step()
	vw.UpdateGUI()
}

func (vw *GUI) Forward() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Forward", nil)
	ev.Step()
	vw.UpdateGUI()
}

func (vw *GUI) Consume() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Consume", nil)
	ev.Step()
	vw.UpdateGUI()
}
