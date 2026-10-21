// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
	"fmt"

	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/lab/base/randx"
)

// partial lines are rendered across different "hypercolumn" groups
// where the maximally ambiguous bottom-up input is just the central
// point across groups, with horizontal, vertical, and diagonal lines
// possible as larger-scale structure. use full wrap-around.
//
//         o
//        / \
//      o     o
//     / \   / \
//
//  \|/   \|/   \|/
// - - - -  -- --- -  <- Lines, 3x3
//  /|\   /|\   /|\   <- Pixels 5x5
//

// Config has configuration parameters for the environment.
type Config struct {

	// NPixels is the size of one dimension (square) of the
	// lowest-level groups where lines are rendered.
	// Should be an odd number so there is an ambiguous
	// central point.
	NPixels int

	// NLines is the number of pixel-groups in each dimension.
	NLines int
}

func (cfg *Config) Defaults() {
	cfg.NPixels = 5
	cfg.NLines = 3
}

// Angles are the different line angles.
type Angles int32 //enums:enum
const (
	Horiz Angles = iota
	Vert
	// DiagR is up-right-going diagonal
	DiagR
	// DiagL is up-left-going diagonal
	DiagL
)

// Line defines a single line to render.
// There is full wraparound.
type Line struct {
	// Angle is the angle of the line.
	Angle Angles

	// Start is the starting point.
	Start vecint.Vector2i

	// Length is how long to make the line.
	Length int
}

// AllLines generates all line starting positions and lengths.
func (cfg *Config) AllLines() []Line {
	nl := cfg.NLines

	n := nl * nl * (nl - 1) * int(LineAnglesN)

	lns := make([]Line, 0, n)

	for sy := range nl {
		for sx := range nl {
			for ln := 2; ln < nl; ln++ { // not len = 3 b/c that needs to be canonicalized
				for la := range LineAnglesN {
					lns = append(lns, Line{Angle: la, Length: ln, Start: vecint.Vec2i(sx, sy)})
				}
			}
			if sy == 0 {
				lns = append(lns, Line{Angle: Vert, Length: nl, Start: vecint.Vec2i(sx, sy)})
			}
			if sx == 0 {
				lns = append(lns, Line{Angle: Horiz, Length: nl, Start: vecint.Vec2i(sx, sy)})
			}
			if sx == 0 && sy == 0 {
				lns = append(lns, Line{Angle: DiagR, Length: nl, Start: vecint.Vec2i(sx, sy)})
			}
			if sx == nl-1 && sy == 0 {
				lns = append(lns, Line{Angle: DiagL, Length: nl, Start: vecint.Vec2i(sx, sy)})
			}
		}
	}

	fmt.Println("nlines:", len(lns), n)
	return lns
}

func (ev *ConSatEnv) InitLines() {
	globalLock.Lock()
	defer globalLock.Unlock()

	if lines == nil {
		rs := ev.RunRandSeed
		ev.RunRandSeed = 137
		randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(0)+1))

		lines = ev.Configs.AllLines()
		order = ev.Rand.Perm(len(lines))

		ev.RunRandSeed = rs
	}
}

func (ev *ConSatEnv) RenderLine(ln Line) {
	in := ev.States["Input"]
	in.SetZeros()
	full := ev.States["Full"]
	full.SetZeros()
	var inc vecint.Vector2i
	switch ln.Angle {
	case Horiz:
		inc.Set(1, 0)
	case Vert:
		inc.Set(0, 1)
	case DiagR:
		inc.Set(1, 1)
	case DiagL:
		inc.Set(-1, 1)
	}
	pt := ln.Start
	nl := ev.Configs.NLines
	np := ev.Configs.NPixels
	pc := np / 2
	pp := vecint.Vec2i(pc, pc)
	for i := range ln.Length {
		in.Set(1.0-float32(i)*.1, pt.Y, pt.X, pp.Y, pp.X) // todo spread around here too..
		full.Set(1.0, pt.Y, pt.X, pp.Y, pp.X)
		pn := pp.Add(inc)
		n := pc
		if i >= 1 {
			n = np
			switch ln.Angle {
			case Horiz:
				pn.X = 0
			case Vert:
				pn.Y = 0
			case DiagR:
				pn.Set(0, 0)
			case DiagL:
				pn.Set(np-1, 0)
			}
		}
		if i == ln.Length-1 {
			n = pc
		}
		for range n {
			full.Set(1.0, pt.Y, pt.X, pn.Y, pn.X)
			pn.SetAdd(inc)
		}
		pt.SetAdd(inc)
		if pt.X < 0 {
			pt.X += nl
		} else {
			pt.X = pt.X % nl // wrap
		}
		if pt.Y < 0 {
			pt.Y += nl
		} else {
			pt.Y = pt.Y % nl
		}
	}
}
