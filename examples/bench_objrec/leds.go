// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"

	"github.com/goki/gi/gi"
	"github.com/goki/gi/girl"
)

// LEDraw renders old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDraw struct {

	// [def: 4] line width of LEDraw as percent of display size
	Width float32 `def:"4" desc:"line width of LEDraw as percent of display size"`

	// [def: 0.6] size of overall LED as proportion of overall image size
	Size float32 `def:"0.6" desc:"size of overall LED as proportion of overall image size"`

	// color name for drawing lines
	LineColor gi.ColorName `desc:"color name for drawing lines"`

	// color name for background
	BgColor gi.ColorName `desc:"color name for background"`

	// size of image to render
	ImgSize image.Point `desc:"size of image to render"`

	// [view: -] rendered image
	Image *image.RGBA `view:"-" desc:"rendered image"`

	// [view: +] painter object
	Paint girl.Paint `view:"+" desc:"painter object"`

	// [view: -] rendering state
	Render girl.State `view:"-" desc:"rendering state"`
}

func (ld *LEDraw) Defaults() {
	ld.ImgSize = image.Point{120, 120}
	ld.Width = 4
	ld.Size = 0.6
	ld.LineColor = "white"
	ld.BgColor = "black"
}

// Init ensures that the image is created and of the right size, and renderer is initialized
func (ld *LEDraw) Init() {
	if ld.ImgSize.X == 0 || ld.ImgSize.Y == 0 {
		ld.Defaults()
	}
	if ld.Image != nil {
		cs := ld.Image.Bounds().Size()
		if cs != ld.ImgSize {
			ld.Image = nil
		}
	}
	if ld.Image == nil {
		ld.Image = image.NewRGBA(image.Rectangle{Max: ld.ImgSize})
	}
	ld.Render.Init(ld.ImgSize.X, ld.ImgSize.Y, ld.Image)
	ld.Paint.Defaults()
	ld.Paint.StrokeStyle.Width.SetPct(ld.Width)
	ld.Paint.StrokeStyle.Color.SetName(string(ld.LineColor))
	ld.Paint.FillStyle.Color.SetName(string(ld.BgColor))
	ld.Paint.SetUnitContextExt(ld.ImgSize)
}

// Clear clears the image with BgColor
func (ld *LEDraw) Clear() {
	if ld.Image == nil {
		ld.Init()
	}
	ld.Paint.Clear(&ld.Render)
}

// DrawSeg draws one segment
func (ld *LEDraw) DrawSeg(seg LEDSegs) {
	rs := &ld.Render
	ctrX := float32(ld.ImgSize.X) * 0.5
	ctrY := float32(ld.ImgSize.Y) * 0.5
	szX := ctrX * ld.Size
	szY := ctrY * ld.Size
	// note: top-zero coordinates
	switch seg {
	case Bottom:
		ld.Paint.DrawLine(rs, ctrX-szX, ctrY+szY, ctrX+szX, ctrY+szY)
	case Left:
		ld.Paint.DrawLine(rs, ctrX-szX, ctrY-szY, ctrX-szX, ctrY+szY)
	case Right:
		ld.Paint.DrawLine(rs, ctrX+szX, ctrY-szY, ctrX+szX, ctrY+szY)
	case Top:
		ld.Paint.DrawLine(rs, ctrX-szX, ctrY-szY, ctrX+szX, ctrY-szY)
	case CenterH:
		ld.Paint.DrawLine(rs, ctrX-szX, ctrY, ctrX+szX, ctrY)
	case CenterV:
		ld.Paint.DrawLine(rs, ctrX, ctrY-szY, ctrX, ctrY+szY)
	}
	ld.Paint.Stroke(rs)
}

// DrawLED draws one LED of given number, based on LEDdata
func (ld *LEDraw) DrawLED(num int) {
	led := LEData[num]
	for _, seg := range led {
		ld.DrawSeg(seg)
	}
}

//////////////////////////////////////////////////////////////////////////
//  LED data

// LEDSegs are the led segments
type LEDSegs int32

const (
	Bottom LEDSegs = iota
	Left
	Right
	Top
	CenterH
	CenterV
	LEDSegsN
)

var LEData = [][3]LEDSegs{
	{CenterH, CenterV, Right},
	{Top, CenterV, Bottom},
	{Top, Right, Bottom},
	{Bottom, CenterV, Right},
	{Left, CenterH, Right},

	{Left, CenterV, CenterH},
	{Left, CenterV, Right},
	{Left, CenterV, Bottom},
	{Left, CenterH, Top},
	{Left, CenterH, Bottom},

	{Top, CenterV, Right},
	{Bottom, CenterV, CenterH},
	{Right, CenterH, Bottom},
	{Top, CenterH, Bottom},
	{Left, Top, Right},

	{Top, CenterH, Right},
	{Left, CenterV, Top},
	{Top, Left, Bottom},
	{Left, Bottom, Right},
	{Top, CenterV, CenterH},
}
