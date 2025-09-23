// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"image"

	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/styles"
	"cogentcore.org/core/tree"
	"cogentcore.org/core/xyz/xyzcore"
)

// GUI provides a GUI view onto the EmeryEnv
type GUI struct {
	// Env is the environment we're viewing
	Env *EmeryEnv

	// 3D visualization of the Scene
	SceneEditor *xyzcore.SceneEditor

	// first-person right-eye full field view
	EyeRImageDisp *core.Image `display:"-"`

	// first-person left-eye fovea view
	EyeLImageDisp *core.Image `display:"-"`
}

func (ge *GUI) ConfigGUI(ev *EmeryEnv, b core.Widget) {
	ge.Env = ev
	core.NewToolbar(b).Maker(ge.MakeToolbar)

	fr := core.NewFrame(b)
	fr.Styler(func(s *styles.Style) {
		s.Direction = styles.Column
		s.Grow.Set(1, 1)
	})

	imfr := core.NewFrame(fr)
	imfr.Styler(func(s *styles.Style) {
		s.Display = styles.Grid
		s.Columns = 2
		s.Grow.Set(0, 0)
	})

	core.NewText(imfr).SetText("Eye-View, Left:")
	core.NewText(imfr).SetText("Right:")

	ge.EyeLImageDisp = core.NewImage(imfr)
	ge.EyeLImageDisp.Name = "eye-l-image"
	ge.EyeLImageDisp.Image = image.NewRGBA(image.Rectangle{Max: ev.Camera.Size})

	ge.EyeRImageDisp = core.NewImage(imfr)
	ge.EyeRImageDisp.Name = "eye-r-image"
	ge.EyeRImageDisp.Image = image.NewRGBA(image.Rectangle{Max: ev.Camera.Size})

	// re-use existing scene!
	ge.SceneEditor = xyzcore.NewSceneEditorForScene(fr, ev.World.Scene)
	ge.SceneEditor.UpdateWidget()
	sc := ge.SceneEditor.SceneXYZ()

	sc.Camera.Pose.Pos = math32.Vec3(0, 29, -4)
	sc.Camera.LookAt(math32.Vec3(0, 4, -5), math32.Vec3(0, 1, 0))
	sc.SaveCamera("2")

	sc.Camera.Pose.Pos = math32.Vec3(0, 24, 32)
	sc.Camera.LookAt(math32.Vec3(0, 3.6, 0), math32.Vec3(0, 1, 0))
	sc.SaveCamera("1")
	sc.SaveCamera("default")
}

func (ge *GUI) Update() {
	if ge.SceneEditor == nil || !ge.SceneEditor.IsVisible() { // || !em.Disp {
		return
	}
	ev := ge.Env
	if ev.EyeRImage != nil {
		ge.EyeRImageDisp.SetImage(ev.EyeRImage)
		ge.EyeRImageDisp.NeedsRender()
	}
	if ev.EyeLImage != nil {
		ge.EyeLImageDisp.SetImage(ev.EyeLImage)
		ge.EyeLImageDisp.NeedsRender()
	}
	ge.SceneEditor.NeedsRender()
}

func (ge *GUI) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.Button) {
		w.SetText("Init").SetIcon(icons.ClearAll).
			SetTooltip("Init env").
			OnClick(func(e events.Event) {
				ge.Env.Init(0)
			})
	})
}
