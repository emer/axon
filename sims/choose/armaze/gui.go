// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"fmt"
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
	"cogentcore.org/core/xyz/physics"
	"cogentcore.org/core/xyz/physics/world"
	"cogentcore.org/core/xyz/xyzcore"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/plotcore"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensorcore"
	"github.com/emer/axon/v2/axon"
)

// Geom is overall geometry of the space
type Geom struct {

	// width of arm -- emery rodent is 1 unit wide
	ArmWidth float32 `default:"2"`

	// total space between arms, ends up being divided on either side
	ArmSpace float32 `default:"1"`

	// multiplier per unit arm length -- keep square with width
	LengthScale float32 `default:"2"`

	// thickness of walls, floor
	Thick float32 `default:"0.1"`

	// height of walls
	Height float32 `default:"0.2"`

	// width + space
	ArmWidthTot float32 `edit:"-"`

	// computed total depth, starts at 0 goes deep
	Depth float32 `edit:"-"`

	// computed total width
	Width float32 `edit:"-"`

	// half width for centering on 0 X
	HalfWidth float32 `edit:"-"`
}

func (ge *Geom) Config(nArms int, maxLen int) {
	ge.ArmSpace = 1
	ge.ArmWidth = 2
	ge.LengthScale = 2
	ge.Thick = 0.1
	ge.Height = 0.2
	ge.ArmWidthTot = ge.ArmWidth + ge.ArmSpace
	ge.Width = float32(nArms) * ge.ArmWidthTot
	ge.Depth = float32(maxLen) * ge.LengthScale
	ge.HalfWidth = ge.Width / 2
}

// pos returns the center position for given arm, position coordinate
func (ge *Geom) Pos(arm, pos int) (x, y float32) {
	x = (float32(arm)+.5)*ge.ArmWidthTot - ge.HalfWidth
	y = -(float32(pos) + .5) * ge.LengthScale // not centered -- going back in depth
	return
}

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

	// thickness (X) and height (Y) of walls
	WallSize math32.Vector2

	// current internal / behavioral state
	State TraceStates

	// trace record of recent activity
	Trace StateTrace

	// view of the gui obj
	EnvForm *core.Form `display:"-"`

	// ArmMaze TabView
	WorldTabs *core.Tabs `display:"-"`

	// ArmMaze is running
	IsRunning bool `display:"-"`

	// current depth map
	DepthValues []float32

	// offscreen render camera settings
	Camera world.Camera

	// color map to use for rendering depth map
	DepthMap core.ColorMapName

	// first-person right-eye full field view
	EyeRFullImage *core.Image `display:"-"`

	// first-person right-eye fovea view
	EyeRFovImage *core.Image `display:"-"`

	// depth map bitmap view
	DepthImage *core.Image `display:"-"`

	// plot of positive valence drives, active OFC US state, and reward
	USposPlot *plotcore.Editor

	// data for USPlot
	USposData *table.Table

	// plot of negative valence active OFC US state, and outcomes
	USnegPlot *plotcore.Editor

	// data for USPlot
	USnegData *table.Table

	// geometry of world
	Geom Geom

	// world
	World *world.World `display:"no-inline"`

	// emer group
	Emery *physics.Group `display:"-"`

	// arms group
	Arms *physics.Group `display:"-"`

	// stims group
	Stims *physics.Group `display:"-"`

	// Right eye of emery
	EyeR physics.Body `display:"-"`

	// contacts from last step, for body
	Contacts physics.Contacts `display:"-"`
}

// ConfigWorldGUI configures all the world view GUI elements
// pass an initial env to use for configuring
func (vw *GUI) ConfigWorldGUI(ev *Env) *core.Body {
	vw.Disp = true
	vw.Env = ev
	vw.EnvName = ev.Name
	vw.WallSize.Set(0.1, 2)
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

	b := core.NewBody("armaze").SetTitle("Arm Maze")

	split := core.NewSplits(b)

	svfr := core.NewFrame(split)
	svfr.SetName("svfr")
	svfr.Styler(func(s *styles.Style) {
		s.Direction = styles.Column
	})

	vw.EnvForm = core.NewForm(svfr).SetStruct(vw)
	imfr := core.NewFrame(svfr)
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
	vw.USposPlot = plotcore.NewEditor(svfr)
	vw.USposPlot.Name = "us-pos"
	vw.USposPlot.Styler(func(s *styles.Style) {
		s.Max.X.Px(wd)
		s.Max.Y.Px(ht)
	})

	vw.USnegPlot = plotcore.NewEditor(svfr)
	vw.USnegPlot.Name = "us-neg"
	vw.USnegPlot.Styler(func(s *styles.Style) {
		s.Max.X.Px(wd)
		s.Max.Y.Px(ht)
	})
	vw.ConfigUSPlots()

	tv := core.NewTabs(split)
	vw.WorldTabs = tv

	scfr, _ := tv.NewTab("3D View")

	////////    3D Scene

	vw.SceneEditor = xyzcore.NewSceneEditor(scfr)
	vw.SceneEditor.UpdateWidget()
	sc := vw.SceneEditor.SceneXYZ()
	vw.MakeWorld(sc)

	sc.Camera.Pose.Pos = math32.Vec3(0, 29, -4)
	sc.Camera.LookAt(math32.Vec3(0, 4, -5), math32.Vec3(0, 1, 0))
	sc.SaveCamera("2")

	sc.Camera.Pose.Pos = math32.Vec3(0, 24, 32)
	sc.Camera.LookAt(math32.Vec3(0, 3.6, 0), math32.Vec3(0, 1, 0))
	sc.SaveCamera("1")
	sc.SaveCamera("default")

	//////////////////////////////////////////
	//    2D Scene

	twov := core.NewSVG(twofr)
	twov.Name = "sceneview"
	twov.Styler(func(s *styles.Style) {
		twov.SVG.Root.ViewBox.Size.Set(vw.Geom.Width+4, vw.Geom.Depth+4)
		twov.SVG.Root.ViewBox.Min.Set(-0.5*(vw.Geom.Width+4), -0.5*(vw.Geom.Depth+4))
		twov.SetReadOnly(false)
	})

	////////    Toolbar

	split.SetSplits(.4, .6)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(vw.MakeToolbar)
	})
	return b
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

// MakePhysicsWorld constructs a new virtual physics world
func (vw *GUI) MakePhysicsWorld() *physics.Group {
	ev := vw.Env

	pw := physics.NewGroup()
	pw.SetName("RoomWorld")

	vw.Geom.Config(ev.Config.NArms, ev.MaxLength)

	vw.MakeFloor(pw, "floor")
	vw.MakeArms(pw)
	vw.MakeStims(pw, .9, .1)

	vw.MakeEmery(pw, 1)

	pw.WorldInit()
	return pw
}

// MakeFloor makes a floor
func (vw *GUI) MakeFloor(par *physics.Group, name string) {
	ge := &vw.Geom
	dp := ge.Depth + 3*ge.LengthScale

	tree.AddChildAt(par, name, func(rm *physics.Group) {
		rm.Maker(func(p *tree.Plan) {
			tree.AddAt(p, "floor", func(n *physics.Box) {
				n.SetSize(math32.Vec3(ge.Width, ge.Thick, dp)).
					SetColor("grey").SetInitPos(math32.Vec3(0, -ge.Thick/2, -ge.Depth/2-ge.LengthScale))
			})
		})
	})
}

// MakeArms adds all the maze arms.
func (vw *GUI) MakeArms(par *physics.Group) {
	ev := vw.Env
	ge := &vw.Geom
	exln := ge.LengthScale
	halfarm := .5 * ge.ArmWidth
	halfht := .5 * ge.Height

	tree.AddChildAt(par, "arms", func(rm *physics.Group) {
		vw.Arms = rm
		rm.Maker(func(p *tree.Plan) {
			for i, arm := range ev.Config.Arms {
				anm := fmt.Sprintf("arm_%d\n", i)
				tree.AddAt(p, anm, func(agp *physics.Group) {
					x, _ := ge.Pos(i, 0)
					ln := ge.LengthScale * float32(arm.Length)
					halflen := .5*ln + exln

					agp.Maker(func(p *tree.Plan) {
						tree.AddAt(p, "left-wall", func(n *physics.Box) {
							n.SetSize(math32.Vec3(ge.Thick, ge.Height, ln)).
								SetColor("black").SetInitPos(math32.Vec3(x-halfarm, halfht, -halflen))
						})
						tree.AddAt(p, "right-wall", func(n *physics.Box) {
							n.SetSize(math32.Vec3(ge.Thick, ge.Height, ln)).
								SetColor("black").SetInitPos(math32.Vec3(x+halfarm, halfht, -halflen))
						})
					})
				})
			}
		})
	})
}

// MakeStims constructs stimuli: CSs, USs
func (vw *GUI) MakeStims(par *physics.Group, width, height float32) {
	ev := vw.Env
	ge := &vw.Geom
	exln := ge.LengthScale
	// halfarm := .5 * ge.ArmWidth
	usHt := ge.Height
	usDp := 0.2 * ge.LengthScale
	csHt := ge.LengthScale

	tree.AddChildAt(par, "stims", func(stms *physics.Group) {
		vw.Stims = stms
		stms.Maker(func(p *tree.Plan) {
			for i, arm := range ev.Config.Arms {
				x, _ := ge.Pos(i, 0)
				ln := ge.LengthScale * float32(arm.Length)
				usnm := fmt.Sprintf("us_%d\n", i)
				csnm := fmt.Sprintf("cs_%d\n", i)
				tree.AddAt(p, usnm, func(n *physics.Box) {
					n.SetSize(math32.Vec3(ge.ArmWidth, usHt, usDp)).
						SetColor(vw.MatColors[arm.US]).SetInitPos(math32.Vec3(x, 0.5*usHt, -ln-1.1*exln))
				})
				tree.AddAt(p, csnm, func(n *physics.Box) {
					n.SetSize(math32.Vec3(ge.ArmWidth, csHt, ge.Thick)).
						SetColor(vw.MatColors[arm.CS]).SetInitPos(math32.Vec3(x, usHt+0.5*csHt, -ln-2*exln))
					n.Updater(func() {
						n.Color = vw.MatColors[arm.CS]
					})
					n.InitView = func(vn tree.Node) {
						sld := vn.(*xyz.Solid)
						world.BoxInit(n, sld)
						sld.Updater(func() {
							world.UpdateColor(n.Color, n.View.(*xyz.Solid))
						})
					}
				})
			}
		})
	})
}

// MakeEmery constructs a new Emery virtual hamster.
func (vw *GUI) MakeEmery(par *physics.Group, length float32) {
	height := length / 2
	width := height
	headsz := height * 0.75
	hhsz := .5 * headsz
	eyesz := headsz * .2

	tree.AddChildAt(par, "emery", func(emr *physics.Group) {
		vw.Emery = emr
		emr.Maker(func(p *tree.Plan) {
			tree.AddAt(p, "body", func(n *physics.Box) {
				n.SetSize(math32.Vec3(width, height, length)).
					SetColor("purple").SetDynamic(true).
					SetInitPos(math32.Vec3(0, height/2, 0))
				n.Updater(func() {
					n.Color = vw.StateColors[vw.State.String()]
				})
			})
			tree.AddAt(p, "head", func(hgp *physics.Group) {
				hgp.SetInitPos(math32.Vec3(0, hhsz, -(length/2 + hhsz)))
				hgp.Maker(func(p *tree.Plan) {
					tree.AddAt(p, "head", func(n *physics.Box) {
						n.SetSize(math32.Vec3(headsz, headsz, headsz)).
							SetColor("tan").SetDynamic(true).SetInitPos(math32.Vec3(0, 0, 0))
					})
					tree.AddAt(p, "eye-l", func(n *physics.Box) {
						n.SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
							SetColor("green").SetDynamic(true).
							SetInitPos(math32.Vec3(-hhsz*.6, headsz*.1, -(hhsz + eyesz*.3)))
					})
					// note: centering this in head for now to get straight-on view
					tree.AddAt(p, "eye-r", func(n *physics.Box) {
						vw.EyeR = n
						n.SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
							SetColor("green").SetDynamic(true).
							SetInitPos(math32.Vec3(0, headsz*.1, -(hhsz + eyesz*.3)))
					})
				})
			})
		})
		emr.Updater(func() {
			ev := vw.Env
			x, y := vw.Geom.Pos(ev.Arm, ev.Pos)
			emr.Rel.Pos.Set(x, 0, y)
		})
	})
}

// MakeWorld makes the visual world for physical world
func (vw *GUI) MakeWorld(sc *xyz.Scene) {
	pw := vw.MakePhysicsWorld()
	sc.Background = colors.Uniform(colors.FromRGB(230, 230, 255)) // sky blue-ish
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)

	dir := xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	// sc.MultiSample = 1 // we are using depth grab so we need this = 1
	vw.World = world.NewWorld(pw, sc)
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
	img := vw.World.RenderFromNode(vw.EyeR, &vw.Camera)
	if img == nil {
		return
	}
	vw.EyeRFullImage.SetImage(img)
	vw.EyeRFullImage.NeedsRender()

	vw.Camera.FOV = 10
	img = vw.World.RenderFromNode(vw.EyeR, &vw.Camera)
	if img == nil {
		return
	}
	vw.EyeRFovImage.SetImage(img)
	vw.EyeRFovImage.NeedsRender()

	// depth, err := vw.World.DepthImage()
	// if err == nil && depth != nil {
	// 	vw.DepthValues = depth
	// 	vw.ViewDepth(depth)
	// }
}

// ViewDepth updates depth bitmap with depth data
func (vw *GUI) ViewDepth(depth []float32) {
	cmap := colormap.AvailableMaps[string(vw.DepthMap)]
	vw.DepthImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})
	world.DepthImage(vw.DepthImage.Image.(*image.RGBA), depth, cmap, &vw.Camera)
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

	vw.UpdateWorldGUI()
}

func (vw *GUI) UpdateWorldGUI() {
	if vw.SceneEditor == nil || !vw.Disp {
		return
	}
	pw := vw.World.World
	pw.Update()
	pw.WorldRelToAbs()
	vw.World.Update()

	vw.GrabEyeImg()
	if vw.SceneEditor.IsVisible() {
		vw.SceneEditor.NeedsRender()
	}
}

func (vw *GUI) Left() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Left", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Right() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Right", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Forward() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Forward", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Consume() { //types:add
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Consume", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}
