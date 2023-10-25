// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"fmt"
	"log"

	"github.com/emer/axon/axon"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/eve/eve"
	"github.com/emer/eve/evev"
	"github.com/goki/gi/colormap"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gi3d"
	"github.com/goki/gi/gist"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/svg"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

// Geom is overall geometry of the space
type Geom struct {

	// [def: 2] width of arm -- emery rodent is 1 unit wide
	ArmWidth float32 `def:"2" desc:"width of arm -- emery rodent is 1 unit wide"`

	// [def: 1] total space between arms, ends up being divided on either side
	ArmSpace float32 `def:"1" desc:"total space between arms, ends up being divided on either side"`

	// [def: 2] multiplier per unit arm length -- keep square with width
	LengthScale float32 `def:"2" desc:"multiplier per unit arm length -- keep square with width"`

	// [def: 0.1] thickness of walls, floor
	Thick float32 `def:"0.1" desc:"thickness of walls, floor"`

	// [def: 0.2] height of walls
	Height float32 `def:"0.2" desc:"height of walls"`

	// width + space
	ArmWidthTot float32 `inactive:"+" desc:"width + space"`

	// computed total depth, starts at 0 goes deep
	Depth float32 `inactive:"+" desc:"computed total depth, starts at 0 goes deep"`

	// computed total width
	Width float32 `inactive:"+" desc:"computed total width"`

	// half width for centering on 0 X
	HalfWidth float32 `inactive:"+" desc:"half width for centering on 0 X"`
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
	Disp bool `desc:"update display -- turn off to make it faster"`

	// the env being visualized
	Env *Env `desc:"the env being visualized"`

	// name of current env -- number is NData index
	EnvName string `desc:"name of current env -- number is NData index"`

	// list of material colors
	MatColors []string `desc:"list of material colors"`

	// internal state colors
	StateColors map[string]string `desc:"internal state colors"`

	// thickness (X) and height (Y) of walls
	WallSize mat32.Vec2 `desc:"thickness (X) and height (Y) of walls"`

	// current internal / behavioral state
	State TraceStates `desc:"current internal / behavioral state"`

	// trace record of recent activity
	Trace StateTrace `desc:"trace record of recent activity"`

	// [view: -] view of the gui obj
	StructView *giv.StructView `view:"-" desc:"view of the gui obj"`

	// [view: -] ArmMaze GUI window
	WorldWin *gi.Window `view:"-" desc:"ArmMaze GUI window"`

	// [view: -] ArmMaze TabView
	WorldTabs *gi.TabView `view:"-" desc:"ArmMaze TabView"`

	// [view: -] ArmMaze is running
	IsRunning bool `view:"-" desc:"ArmMaze is running"`

	// current depth map
	DepthVals []float32 `desc:"current depth map"`

	// offscreen render camera settings
	Camera evev.Camera `desc:"offscreen render camera settings"`

	// color map to use for rendering depth map
	DepthMap giv.ColorMapName `desc:"color map to use for rendering depth map"`

	// [view: -] first-person right-eye full field view
	EyeRFullImg *gi.Bitmap `view:"-" desc:"first-person right-eye full field view"`

	// [view: -] first-person right-eye fovea view
	EyeRFovImg *gi.Bitmap `view:"-" desc:"first-person right-eye fovea view"`

	// [view: -] depth map bitmap view
	DepthImg *gi.Bitmap `view:"-" desc:"depth map bitmap view"`

	// plot of positive valence drives, active OFC US state, and reward
	USposPlot *eplot.Plot2D `desc:"plot of positive valence drives, active OFC US state, and reward"`

	// data for USPlot
	USposData *etable.Table `desc:"data for USPlot"`

	// plot of negative valence active OFC US state, and outcomes
	USnegPlot *eplot.Plot2D `desc:"plot of negative valence active OFC US state, and outcomes"`

	// data for USPlot
	USnegData *etable.Table `desc:"data for USPlot"`

	// geometry of world
	Geom Geom `desc:"geometry of world"`

	// world
	World *eve.Group `desc:"world"`

	// [view: -] 3D view of world
	View3D *evev.View `view:"-" desc:"3D view of world"`

	// [view: -] emer group
	Emery *eve.Group `view:"-" desc:"emer group"`

	// [view: -] arms group
	Arms *eve.Group `view:"-" desc:"arms group"`

	// [view: -] stims group
	Stims *eve.Group `view:"-" desc:"stims group"`

	// [view: -] Right eye of emery
	EyeR eve.Body `view:"-" desc:"Right eye of emery"`

	// [view: -] contacts from last step, for body
	Contacts eve.Contacts `view:"-" desc:"contacts from last step, for body"`

	// [view: -] gui window
	Win *gi.Window `view:"-" desc:"gui window"`
}

// ConfigWorldGUI configures all the world view GUI elements
// pass an initial env to use for configuring
func (vw *GUI) ConfigWorldGUI(ev *Env) *gi.Window {
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

	width := 1600
	height := 1200

	win := gi.NewMainWindow("armaze", "Arm Maze", width, height)
	vw.WorldWin = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	svfr := gi.AddNewFrame(split, "svfr", gi.LayoutVert)
	svfr.SetStretchMax()
	svfr.SetReRenderAnchor()

	sv := giv.AddNewStructView(svfr, "sv")
	sv.SetStruct(vw)
	vw.StructView = sv

	imgLay := gi.AddNewLayout(svfr, "img-lay", gi.LayoutGrid)
	imgLay.SetProp("columns", 2)
	imgLay.SetProp("spacing", 8)

	gi.AddNewLabel(imgLay, "lab-img-fov", "Eye-View, Fovea:")
	gi.AddNewLabel(imgLay, "lab-img-full", "Full Field:")

	vw.EyeRFovImg = gi.AddNewBitmap(imgLay, "eye-r-fov-img")
	vw.EyeRFovImg.SetSize(vw.Camera.Size)
	vw.EyeRFovImg.LayoutToImgSize()
	// vw.EyeRFovImg.SetProp("vertical-align", gist.AlignTop)

	vw.EyeRFullImg = gi.AddNewBitmap(imgLay, "eye-r-full-img")
	vw.EyeRFullImg.SetSize(vw.Camera.Size)
	vw.EyeRFullImg.LayoutToImgSize()
	// vw.EyeRFullImg.SetProp("vertical-align", gist.AlignTop)

	// gi.AddNewLabel(imfr, "lab-depth", "Right Eye Depth:")
	// vw.DepthImg = gi.AddNewBitmap(imfr, "depth-img")
	// vw.DepthImg.SetSize(vw.Camera.Size)
	// vw.DepthImg.LayoutToImgSize()
	// vw.DepthImg.SetProp("vertical-align", gist.AlignTop)

	vw.USposPlot = eplot.AddNewPlot2D(svfr, "us-pos")
	vw.USnegPlot = eplot.AddNewPlot2D(svfr, "us-neg")
	wd := 700
	vw.USposPlot.SetProp("max-width", wd)
	vw.USnegPlot.SetProp("max-width", wd)
	ht := 160
	vw.USposPlot.SetProp("max-height", ht)
	vw.USnegPlot.SetProp("max-height", ht)
	vw.USposPlot.SetProp("height", ht)
	vw.USnegPlot.SetProp("height", ht)
	vw.ConfigUSPlots()

	tv := gi.AddNewTabView(split, "tv")
	vw.WorldTabs = tv

	scfr := tv.AddNewTab(gi.KiT_Frame, "3D View").(*gi.Frame)
	twofr := tv.AddNewTab(gi.KiT_Frame, "2D View").(*gi.Frame)

	scfr.SetStretchMax()
	twofr.SetStretchMax()

	//////////////////////////////////////////
	//    3D Scene

	vw.ConfigWorld()

	scvw := gi3d.AddNewSceneView(scfr, "sceneview")
	scvw.SetStretchMax()
	scvw.Config()
	sc := scvw.Scene()

	// first, add lights, set camera
	sc.BgColor.SetUInt8(230, 230, 255, 255) // sky blue-ish
	gi3d.AddNewAmbientLight(sc, "ambient", 0.3, gi3d.DirectSun)

	dir := gi3d.AddNewDirLight(sc, "dir", 1, gi3d.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	vw.ConfigView3D(sc)

	// grtx := gi3d.AddNewTextureFile(sc, "ground", "ground.png")
	// wdtx := gi3d.AddNewTextureFile(sc, "wood", "wood.png")

	// floorp := gi3d.AddNewPlane(sc, "floor-plane", 100, 100)
	// floor := gi3d.AddNewSolid(sc, sc, "floor", floorp.Name())
	// floor.Pose.Pos.Set(0, -5, 0)
	// // floor.Mat.Color.SetName("tan")
	// // floor.Mat.Emissive.SetName("brown")
	// floor.Mat.Bright = 2 // .5 for wood / brown
	// floor.Mat.SetTexture(sc, grtx)
	// floor.Mat.Tiling.Reveat.Set(40, 40)

	// sc.Camera.Pose.Pos = mat32.Vec3{0, 100, 0}
	// sc.Camera.LookAt(mat32.Vec3{0, 5, 0}, mat32.Vec3Y)
	// sc.SaveCamera("3")

	sc.Camera.Pose.Pos = mat32.Vec3{0, 29, -4}
	sc.Camera.LookAt(mat32.Vec3{0, 4, -5}, mat32.Vec3Y)
	sc.SaveCamera("2")

	sc.Camera.Pose.Pos = mat32.Vec3{0, 17, 21}
	sc.Camera.LookAt(mat32.Vec3{0, 3.6, 0}, mat32.Vec3Y)
	sc.SaveCamera("1")
	sc.SaveCamera("default")

	//////////////////////////////////////////
	//    2D Scene

	twov := svg.AddNewEditor(twofr, "sceneview")
	twov.Fill = true
	twov.SetProp("background-color", "lightgrey")
	twov.SetStretchMax()
	twov.InitScale()
	twov.Trans.Set(620, 560)
	twov.Scale = 20
	twov.SetTransform()

	//////////////////////////////////////////
	//    Toolbar

	split.SetSplits(.4, .6)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "reset", Tooltip: "Init env.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Env.Init(0)
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Reset Trace", Icon: "reset", Tooltip: "Reset the trace of position, state etc, shown in the 2D View", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Trace = nil
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Left", Icon: "wedge-left", Tooltip: "Rotate Left", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Left()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Right", Icon: "wedge-right", Tooltip: "Rotate Right", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Right()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Forward", Icon: "wedge-up", Tooltip: "Step Forward", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Forward()
		vp.SetFullReRender()
	})

	// tbar.AddAction(gi.ActOpts{Label: "Backward", Icon: "wedge-down", Tooltip: "Step Backward", UpdateFunc: func(act *gi.Action) {
	// 	act.SetActiveStateUpdt(!vw.IsRunning)
	// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 	vw.Backward()
	// 	vp.SetFullReRender()
	// })

	tbar.AddAction(gi.ActOpts{Label: "Consume", Icon: "svg", Tooltip: "Consume item -- only if directly in front", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!vw.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		vw.Consume()
		vp.SetFullReRender()
	})

	tbar.AddSeparator("sep-file")

	// tbar.AddAction(gi.ActOpts{Label: "Open Pats", Icon: "file-open", Tooltip: "Open bit patterns from .json file", UpdateFunc: func(act *gi.Action) {
	// 	act.SetActiveStateUpdt(!vw.IsRunning)
	// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 	giv.CallMethod(vw.Env, "OpenPats", vp)
	// })
	//
	// tbar.AddAction(gi.ActOpts{Label: "Save Pats", Icon: "file-save", Tooltip: "Save bit patterns to .json file", UpdateFunc: func(act *gi.Action) {
	// 	act.SetActiveStateUpdt(!vw.IsRunning)
	// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 	giv.CallMethod(vw.Env, "SavePats", vp)
	// })

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
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
	rm := eve.AddNewGroup(par, name)
	floor := eve.AddNewBox(rm, "floor", mat32.Vec3{0, -ge.Thick / 2, -ge.Depth/2 - ge.LengthScale}, mat32.Vec3{ge.Width, ge.Thick, dp})
	floor.Color = "grey"
	return rm
}

// ConfigArms adds all the arms
func (vw *GUI) ConfigArms(par *eve.Group) *eve.Group {
	ev := vw.Env
	rm := eve.AddNewGroup(par, "arms")
	ge := &vw.Geom
	exln := ge.LengthScale
	halfarm := .5 * ge.ArmWidth
	halfht := .5 * ge.Height
	for i, arm := range ev.Config.Arms {
		anm := fmt.Sprintf("arm_%d\n", i)
		agp := eve.AddNewGroup(rm, anm)
		x, _ := ge.Pos(i, 0)
		ln := ge.LengthScale * float32(arm.Length)
		halflen := .5*ln + exln
		// bwall := eve.AddNewBox(agp, "back-wall", mat32.Vec3{x, halfht, -ln - exln}, mat32.Vec3{ge.ArmWidth, ge.Height, ge.Thick})
		// bwall.Color = "blue"
		lwall := eve.AddNewBox(agp, "left-wall", mat32.Vec3{x - halfarm, halfht, -halflen}, mat32.Vec3{ge.Thick, ge.Height, ln})
		lwall.Color = "black" // "red"
		rwall := eve.AddNewBox(agp, "right-wall", mat32.Vec3{x + halfarm, halfht, -halflen}, mat32.Vec3{ge.Thick, ge.Height, ln})
		rwall.Color = "black" // "green"
	}
	return rm
}

// ConfigStims constructs stimuli: CSs, USs
func (vw *GUI) ConfigStims(par *eve.Group, name string, width, height float32) *eve.Group {
	ev := vw.Env
	ge := &vw.Geom
	stms := eve.AddNewGroup(par, name)
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
		uso := eve.AddNewBox(stms, usnm, mat32.Vec3{float32(x), 0.5 * usHt, -ln - exln}, mat32.Vec3{ge.ArmWidth, usHt, usDp})
		uso.Color = vw.MatColors[arm.US]
		cso := eve.AddNewBox(stms, csnm, mat32.Vec3{float32(x), usHt + .5*csHt, -ln - 2*exln}, mat32.Vec3{ge.ArmWidth, csHt, ge.Thick})
		cso.Color = vw.MatColors[arm.CS]
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
	emr := eve.AddNewGroup(par, "emery")
	height := length / 2
	width := height
	body := eve.AddNewBox(emr, "body", mat32.Vec3{0, height / 2, 0}, mat32.Vec3{width, height, length})
	// body := eve.AddNewCapsule(emr, "body", mat32.Vec3{0, height / 2, 0}, height, width/2)
	// body := eve.AddNewCylinder(emr, "body", mat32.Vec3{0, height / 2, 0}, height, width/2)
	body.Color = "purple"
	body.SetDynamic()

	headsz := height * 0.75
	hhsz := .5 * headsz
	hgp := eve.AddNewGroup(emr, "head")
	hgp.Initial.Pos = mat32.Vec3{0, hhsz, -(length/2 + hhsz)}

	head := eve.AddNewBox(hgp, "head", mat32.Vec3{0, 0, 0}, mat32.Vec3{headsz, headsz, headsz})
	head.Color = "tan"
	head.SetDynamic()
	eyesz := headsz * .2
	eyel := eve.AddNewBox(hgp, "eye-l", mat32.Vec3{-hhsz * .6, headsz * .1, -(hhsz + eyesz*.3)}, mat32.Vec3{eyesz, eyesz * .5, eyesz * .2})
	eyel.Color = "green"
	eyel.SetDynamic()
	// note: centering this in head for now to get straight-on view
	eyer := eve.AddNewBox(hgp, "eye-r", mat32.Vec3{0, headsz * .1, -(hhsz + eyesz*.3)}, mat32.Vec3{eyesz, eyesz * .5, eyesz * .2})
	eyer.Color = "green"
	eyer.Initial.Quat.SetFromEuler(mat32.Vec3{-0.02, 0, 0}) // look a bit down
	eyer.SetDynamic()
	return emr
}

// ConfigView3D makes the 3D view
func (vw *GUI) ConfigView3D(sc *gi3d.Scene) {
	// sc.MultiSample = 1 // we are using depth grab so we need this = 1
	wgp := gi3d.AddNewGroup(sc, sc, "world")
	vw.View3D = evev.NewView(vw.World, sc, wgp)
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
		vw.EyeRFullImg.SetImage(img, 0, 0)
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
		vw.EyeRFovImg.SetImage(img, 0, 0)
	} else {
		log.Println(err)
	}

	// depth, err := vw.View3D.DepthImage()
	// if err == nil && depth != nil {
	// 	vw.DepthVals = depth
	// 	vw.ViewDepth(depth)
	// }
	vw.View3D.Scene.Render2D()
	vw.View3D.Scene.DirectWinUpload()
}

// ViewDepth updates depth bitmap with depth data
func (vw *GUI) ViewDepth(depth []float32) {
	cmap := colormap.AvailMaps[string(vw.DepthMap)]
	vw.DepthImg.SetSize(vw.Camera.Size)
	evev.DepthImage(vw.DepthImg.Pixels, depth, cmap, &vw.Camera)
	vw.DepthImg.UpdateSig()
}

func (vw *GUI) ConfigWorldView(tg *etview.TensorGrid) {
	cnm := "ArmMazeColors"
	cm, ok := colormap.AvailMaps[cnm]
	if !ok {
		ev := vw.Env
		cm = &colormap.Map{}
		cm.Name = cnm
		cm.Indexed = true
		nc := ev.Config.NCSs
		cm.Colors = make([]gist.Color, nc)
		cm.NoColor = gist.Black
		for i, cnm := range vw.MatColors {
			cm.Colors[i].SetString(cnm, nil)
		}
		colormap.AvailMaps[cnm] = cm
	}
	tg.Disp.Defaults()
	tg.Disp.ColorMap = giv.ColorMapName(cnm)
	tg.Disp.GridFill = 1
	tg.SetStretchMax()
}

func (vw *GUI) UpdateWorld(ctx *axon.Context, ev *Env, net *axon.Network, state TraceStates) {
	vw.State = state
	vw.Trace.AddRec(ctx, uint32(ev.Di), ev, net, state)
	if vw.WorldWin == nil || !vw.Disp {
		return
	}

	if vw.Env != ev {
		vw.Env = ev
		vw.EnvName = ev.Nm
		vw.Trace = nil
		vw.StructView.UpdateSig()
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
	if vw.WorldWin == nil || !vw.Disp {
		return
	}
	// update state:
	vw.SetEmeryPose()
	vw.UpdateStims()
	vw.World.WorldRelToAbs()
	vw.View3D.UpdatePose()
	vw.View3D.UpdateBodyView([]string{"body"})

	// update views:
	vw.GrabEyeImg()
	vw.View3D.Scene.UpdateSig()
}

func (vw *GUI) Left() {
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Left", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Right() {
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Right", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Forward() {
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Forward", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}

func (vw *GUI) Consume() {
	ev := vw.Env
	ev.InstinctAct(ev.JustGated, ev.HasGated)
	ev.Action("Consume", nil)
	ev.Step()
	vw.UpdateWorldGUI()
}
