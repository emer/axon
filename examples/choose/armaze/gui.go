// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"cogentcore.org/core/abilities"
	"cogentcore.org/core/colors"
	"cogentcore.org/core/colors/colormap"
	"cogentcore.org/core/core"
	"cogentcore.org/core/errors"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/styles"
	"cogentcore.org/core/views"
	"cogentcore.org/core/xyz"
	"cogentcore.org/core/xyzview"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/etview"
	"github.com/emer/eve/v2/eve"
	"github.com/emer/eve/v2/evev"
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
	SceneView *xyzview.SceneView

	// 2D visualization of the Scene
	Scene2D *core.SVG

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
	StructView *views.StructView `view:"-"`

	// ArmMaze TabView
	WorldTabs *core.Tabs `view:"-"`

	// ArmMaze is running
	IsRunning bool `view:"-"`

	// current depth map
	DepthValues []float32

	// offscreen render camera settings
	Camera evev.Camera

	// color map to use for rendering depth map
	DepthMap views.ColorMapName

	// first-person right-eye full field view
	EyeRFullImage *core.Image `view:"-"`

	// first-person right-eye fovea view
	EyeRFovImage *core.Image `view:"-"`

	// depth map bitmap view
	DepthImage *core.Image `view:"-"`

	// plot of positive valence drives, active OFC US state, and reward
	USposPlot *eplot.Plot2D

	// data for USPlot
	USposData *etable.Table

	// plot of negative valence active OFC US state, and outcomes
	USnegPlot *eplot.Plot2D

	// data for USPlot
	USnegData *etable.Table

	// geometry of world
	Geom Geom

	// world
	World *eve.Group

	// 3D view of world
	View3D *evev.View `view:"-"`

	// emer group
	Emery *eve.Group `view:"-"`

	// arms group
	Arms *eve.Group `view:"-"`

	// stims group
	Stims *eve.Group `view:"-"`

	// Right eye of emery
	EyeR eve.Body `view:"-"`

	// contacts from last step, for body
	Contacts eve.Contacts `view:"-"`
}

// ConfigWorldGUI configures all the world view GUI elements
// pass an initial env to use for configuring
func (vw *GUI) ConfigWorldGUI(ev *Env) *core.Body {
	vw.Disp = true
	vw.Env = ev
	vw.EnvName = ev.Nm
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

	split := core.NewSplits(b, "split")

	svfr := core.NewFrame(split, "svfr")
	svfr.Style(func(s *styles.Style) {
		s.Direction = styles.Column
	})

	vw.StructView = views.NewStructView(svfr, "sv").SetStruct(vw)
	imfr := core.NewFrame(svfr).Style(func(s *styles.Style) {
		s.Display = styles.Grid
		s.Columns = 2
		s.Grow.Set(0, 0)
	})
	core.NewLabel(imfr).SetText("Eye-View, Fovea:")
	core.NewLabel(imfr).SetText("Full Field:")

	vw.EyeRFovImage = core.NewImage(imfr, "eye-r-fov-img")
	vw.EyeRFovImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})

	vw.EyeRFullImage = core.NewImage(imfr, "eye-r-full-img")
	vw.EyeRFullImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})

	wd := float32(200)
	ht := float32(100)
	vw.USposPlot = eplot.NewPlot2D(svfr, "us-pos")
	vw.USposPlot.Style(func(s *styles.Style) {
		s.Min.X.Px(wd)
		s.Min.Y.Px(ht)
		s.Grow.Set(0, 0)
	})

	vw.USnegPlot = eplot.NewPlot2D(svfr, "us-neg")
	vw.USnegPlot.Style(func(s *styles.Style) {
		s.Min.X.Px(wd)
		s.Min.Y.Px(ht)
		s.Grow.Set(0, 0)
	})
	vw.ConfigUSPlots()

	tv := core.NewTabs(split)
	vw.WorldTabs = tv

	scfr := tv.NewTab("3D View")
	twofr := tv.NewTab("2D View")

	//////////////////////////////////////////
	//    3D Scene

	vw.ConfigWorld()

	vw.SceneView = xyzview.NewSceneView(scfr, "sceneview")
	vw.SceneView.Config()
	se := vw.SceneView.SceneXYZ()
	vw.ConfigView3D(se)

	se.Camera.Pose.Pos = math32.Vec3(0, 29, -4)
	se.Camera.LookAt(math32.Vec3(0, 4, -5), math32.Vec3(0, 1, 0))
	se.SaveCamera("2")

	se.Camera.Pose.Pos = math32.Vec3(0, 24, 32)
	se.Camera.LookAt(math32.Vec3(0, 3.6, 0), math32.Vec3(0, 1, 0))
	se.SaveCamera("1")
	se.SaveCamera("default")

	//////////////////////////////////////////
	//    2D Scene

	twov := core.NewSVG(twofr, "sceneview")
	twov.Style(func(s *styles.Style) {
		twov.SVG.Root.ViewBox.Size.Set(vw.Geom.Width+4, vw.Geom.Depth+4)
		twov.SVG.Root.ViewBox.Min.Set(-0.5*(vw.Geom.Width+4), -0.5*(vw.Geom.Depth+4))
		twov.SetReadOnly(false)
	})

	//////////////////////////////////////////
	//    Toolbar

	split.SetSplits(.4, .6)

	b.AddAppBar(func(tb *core.Toolbar) {
		core.NewButton(tb).SetText("Init").SetIcon(icons.ClearAll).
			SetTooltip("Init env").
			OnClick(func(e events.Event) {
				vw.Env.Init(0)
			})
		core.NewButton(tb).SetText("Reset Trace").SetIcon(icons.Undo).
			SetTooltip("Reset trace of position, etc, shown in 2D View").
			OnClick(func(e events.Event) {
				vw.Trace = nil
			})
		views.NewFuncButton(tb, vw.Forward).SetText("Fwd").SetIcon(icons.SkipNext).
			Style(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
		views.NewFuncButton(tb, vw.Left).SetText("Left").SetIcon(icons.KeyboardArrowLeft).
			Style(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
		views.NewFuncButton(tb, vw.Right).SetText("Right").SetIcon(icons.KeyboardArrowRight).
			Style(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})
		views.NewFuncButton(tb, vw.Consume).SetText("Consume").SetIcon(icons.SentimentExcited).
			Style(func(s *styles.Style) {
				s.SetAbilities(true, abilities.RepeatClickable)
			})

		core.NewSeparator(tb)
	})
	return b
}

// ConfigWorld constructs a new virtual physics world for flat world
func (vw *GUI) ConfigWorld() {
	ev := vw.Env

	vw.World = &eve.Group{}
	vw.World.InitName(vw.World, "ArmMaze")

	vw.Geom.Config(ev.Config.NArms, ev.MaxLength)

	vw.AddFloor(vw.World, "floor")
	vw.Arms = vw.ConfigArms(vw.World)
	vw.Stims = vw.ConfigStims(vw.World, "stims", .9, .1)

	vw.Emery = vw.ConfigEmery(vw.World, 1)
	vw.EyeR = vw.Emery.ChildByName("head", 1).ChildByName("eye-r", 2).(eve.Body)

	vw.World.WorldInit()

	vw.SetEmeryPose()
}

// AddFloor adds a floor
func (vw *GUI) AddFloor(par *eve.Group, name string) *eve.Group {
	ge := &vw.Geom
	dp := ge.Depth + 3*ge.LengthScale
	rm := eve.NewGroup(par, name)
	eve.NewBox(rm, "floor").SetSize(math32.Vec3(ge.Width, ge.Thick, dp)).
		SetColor("grey").SetInitPos(math32.Vec3(0, -ge.Thick/2, -ge.Depth/2-ge.LengthScale))
	return rm
}

// ConfigArms adds all the arms
func (vw *GUI) ConfigArms(par *eve.Group) *eve.Group {
	ev := vw.Env
	rm := eve.NewGroup(par, "arms")
	ge := &vw.Geom
	exln := ge.LengthScale
	halfarm := .5 * ge.ArmWidth
	halfht := .5 * ge.Height
	for i, arm := range ev.Config.Arms {
		anm := fmt.Sprintf("arm_%d\n", i)
		agp := eve.NewGroup(rm, anm)
		x, _ := ge.Pos(i, 0)
		ln := ge.LengthScale * float32(arm.Length)
		halflen := .5*ln + exln

		eve.NewBox(agp, "left-wall").SetSize(math32.Vec3(ge.Thick, ge.Height, ln)).
			SetColor("black").SetInitPos(math32.Vec3(x-halfarm, halfht, -halflen))

		eve.NewBox(agp, "right-wall").SetSize(math32.Vec3(ge.Thick, ge.Height, ln)).
			SetColor("black").SetInitPos(math32.Vec3(x+halfarm, halfht, -halflen))
	}
	return rm
}

// ConfigStims constructs stimuli: CSs, USs
func (vw *GUI) ConfigStims(par *eve.Group, name string, width, height float32) *eve.Group {
	ev := vw.Env
	ge := &vw.Geom
	stms := eve.NewGroup(par, name)
	exln := ge.LengthScale
	// halfarm := .5 * ge.ArmWidth
	usHt := ge.Height
	usDp := 0.2 * ge.LengthScale
	csHt := ge.LengthScale

	for i, arm := range ev.Config.Arms {
		x, _ := ge.Pos(i, 0)
		ln := ge.LengthScale * float32(arm.Length)
		usnm := fmt.Sprintf("us_%d\n", i)
		csnm := fmt.Sprintf("cs_%d\n", i)

		eve.NewBox(stms, usnm).SetSize(math32.Vec3(ge.ArmWidth, usHt, usDp)).
			SetColor(vw.MatColors[arm.US]).SetInitPos(math32.Vec3(x, 0.5*usHt, -ln-1.1*exln))

		eve.NewBox(stms, csnm).SetSize(math32.Vec3(ge.ArmWidth, csHt, ge.Thick)).
			SetColor(vw.MatColors[arm.CS]).SetInitPos(math32.Vec3(x, usHt+0.5*csHt, -ln-2*exln))
	}
	return stms
}

func (vw *GUI) UpdateStims() {
	var updts []string
	ev := vw.Env
	stms := *vw.Stims.Children()
	for i, moi := range stms {
		mo := moi.(*eve.Box)
		if i%2 == 1 { // CS
			armi := i / 2
			arm := ev.Config.Arms[armi]
			clr := vw.MatColors[arm.CS]
			if mo.Color != clr {
				mo.Color = clr
				updts = append(updts, mo.Name())
			}
		}
	}
	if len(updts) > 0 {
		vw.View3D.UpdateBodyView(updts)
	}
}

// ConfigEmery constructs a new Emery virtual hamster
func (vw *GUI) ConfigEmery(par *eve.Group, length float32) *eve.Group {
	emr := eve.NewGroup(par, "emery")
	height := length / 2
	width := height

	eve.NewBox(emr, "body").SetSize(math32.Vec3(width, height, length)).
		SetColor("purple").SetDynamic().
		SetInitPos(math32.Vec3(0, height/2, 0))

	headsz := height * 0.75
	hhsz := .5 * headsz
	hgp := eve.NewGroup(emr, "head").SetInitPos(math32.Vec3(0, hhsz, -(length/2 + hhsz)))

	eve.NewBox(hgp, "head").SetSize(math32.Vec3(headsz, headsz, headsz)).
		SetColor("tan").SetDynamic().SetInitPos(math32.Vec3(0, 0, 0))

	eyesz := headsz * .2

	eve.NewBox(hgp, "eye-l").SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
		SetColor("green").SetDynamic().
		SetInitPos(math32.Vec3(-hhsz*.6, headsz*.1, -(hhsz + eyesz*.3)))

	// note: centering this in head for now to get straight-on view
	eve.NewBox(hgp, "eye-r").SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
		SetColor("green").SetDynamic().
		SetInitPos(math32.Vec3(0, headsz*.1, -(hhsz + eyesz*.3)))

	return emr
}

// ConfigView3D makes the 3D view
func (vw *GUI) ConfigView3D(se *xyz.Scene) {
	se.BackgroundColor = colors.FromRGB(230, 230, 255) // sky blue-ish
	xyz.NewAmbientLight(se, "ambient", 0.3, xyz.DirectSun)

	dir := xyz.NewDirLight(se, "dir", 1, xyz.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	// sc.MultiSample = 1 // we are using depth grab so we need this = 1
	wgp := xyz.NewGroup(se, "world")
	vw.View3D = evev.NewView(vw.World, se, wgp)
	vw.View3D.InitLibrary() // this makes a basic library based on body shapes, sizes
	// at this point the library can be updated to configure custom visualizations
	// for any of the named bodies.
	vw.View3D.Sync()
}

func (vw *GUI) ConfigUSPlots() {
	schP := etable.Schema{
		{"US", etensor.STRING, nil, nil},
		{"Drive", etensor.FLOAT64, nil, nil},
		{"OFC", etensor.FLOAT64, nil, nil},
		{"USin", etensor.FLOAT64, nil, nil},
	}
	dp := etable.New(schP, vw.Env.Config.NDrives+1)
	vw.USposData = dp
	vw.USposPlot.Params.Type = eplot.Bar
	vw.USposPlot.Params.Title = "Positive USs"
	vw.USposPlot.Params.Scale = 1
	vw.USposPlot.Params.XAxisCol = "US"

	schN := etable.Schema{
		{"US", etensor.STRING, nil, nil},
		{"OFC", etensor.FLOAT64, nil, nil},
		{"USin", etensor.FLOAT64, nil, nil},
	}
	dn := etable.New(schN, vw.Env.Config.NNegUSs+2)
	vw.USnegData = dn
	vw.USnegPlot.Params.Type = eplot.Bar
	vw.USnegPlot.Params.Title = "Negative USs"
	vw.USnegPlot.Params.Scale = 1
	vw.USnegPlot.Params.XAxisCol = "US"

	cols := []string{"Drive", "USin", "OFC"}
	for i, cl := range cols {
		dp.SetMetaData(cl+":On", "true")
		dp.SetMetaData(cl+":FixMin", "true")
		dp.SetMetaData(cl+":FixMax", "true")
		dp.SetMetaData(cl+":Max", "1")
		if i > 0 {
			dn.SetMetaData(cl+":On", "true")
			dn.SetMetaData(cl+":FixMin", "true")
			dn.SetMetaData(cl+":FixMax", "true")
			dn.SetMetaData(cl+":Max", "1")
		}
	}
	vw.USposPlot.SetTable(dp)
	vw.USnegPlot.SetTable(dn)
}

// GrabEyeImg takes a snapshot from the perspective of Emer's right eye
func (vw *GUI) GrabEyeImg() {
	vw.Camera.FOV = 90
	err := vw.View3D.RenderOffNode(vw.EyeR, &vw.Camera)
	if err != nil {
		log.Println(err)
		return
	}
	img, err := vw.View3D.Image()
	if err == nil && img != nil {
		vw.EyeRFullImage.SetImage(img)
		vw.EyeRFullImage.NeedsRender()
	} else {
		log.Println(err)
	}

	vw.Camera.FOV = 10
	err = vw.View3D.RenderOffNode(vw.EyeR, &vw.Camera)
	if err != nil {
		log.Println(err)
		return
	}
	img, err = vw.View3D.Image()
	if err == nil && img != nil {
		vw.EyeRFovImage.SetImage(img)
	} else {
		log.Println(err)
	}

	// depth, err := vw.View3D.DepthImage()
	// if err == nil && depth != nil {
	// 	vw.DepthValues = depth
	// 	vw.ViewDepth(depth)
	// }
}

// ViewDepth updates depth bitmap with depth data
func (vw *GUI) ViewDepth(depth []float32) {
	cmap := colormap.AvailableMaps[string(vw.DepthMap)]
	vw.DepthImage.Image = image.NewRGBA(image.Rectangle{Max: vw.Camera.Size})
	evev.DepthImage(vw.DepthImage.Image, depth, cmap, &vw.Camera)
}

func (vw *GUI) ConfigWorldView(tg *etview.TensorGrid) {
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
	tg.Disp.Defaults()
	tg.Disp.ColorMap = views.ColorMapName(cnm)
	tg.Disp.GridFill = 1
}

func (vw *GUI) UpdateWorld(ctx *axon.Context, ev *Env, net *axon.Network, state TraceStates) {
	vw.State = state
	vw.Trace.AddRec(ctx, uint32(ev.Di), ev, net, state)
	if vw.SceneView == nil || !vw.Disp {
		return
	}

	if vw.Env != ev {
		vw.Env = ev
		vw.EnvName = ev.Nm
		vw.Trace = nil
		vw.StructView.UpdateFields()
	}

	vw.UpdateWorldGUI()
}

func (vw *GUI) SetEmeryPose() {
	ev := vw.Env
	x, y := vw.Geom.Pos(ev.Arm, ev.Pos)
	vw.Emery.Rel.Pos.Set(x, 0, y)
	bod := vw.Emery.ChildByName("body", 0).(eve.Body).AsBodyBase()
	bod.Color = vw.StateColors[vw.State.String()]
}

func (vw *GUI) UpdateWorldGUI() {
	if vw.SceneView == nil || !vw.Disp {
		return
	}
	vw.SceneView.AsyncLock()
	defer vw.SceneView.AsyncUnlock()

	// update state:
	vw.SetEmeryPose()
	vw.UpdateStims()
	vw.World.WorldRelToAbs()
	vw.View3D.UpdatePose()
	vw.View3D.UpdateBodyView([]string{"body"})
	// vw.View2D.UpdatePose()

	// update views:
	vw.GrabEyeImg()
	if vw.SceneView.IsVisible() {
		vw.SceneView.NeedsRender()
	}
	// if vw.Scene2D.IsVisible() {
	// 	vw.Scene2D.SetNeedsRender(true)
	// }
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
