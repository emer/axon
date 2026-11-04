// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
	"fmt"

	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/lab/base/randx"
)

// partial lines are rendered across different hypercolumn groups
// where the maximally ambiguous bottom-up input is just the central
// point across groups, with horizontal, vertical, and diagonal lines
// possible as larger-scale structure.
// do not use wraparound due to ambiguity.
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

	// NHypers is the number of hypercolumn groups in each dimension.
	NHypers int

	// Partial renders partial amounts of the full line -- if false,
	// then only the fully ambiguous points.
	Partial bool

	// PartialRange is the uniformly-sampled range of probability
	// to add partial line pixels, when Partial = true.
	PartialRange minmax.F32
}

func (cfg *Config) Defaults() {
	cfg.NPixels = 5
	cfg.NHypers = 5
	cfg.PartialRange.Set(0, 1)
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

// Delta returns the X,Y line delta per step.
func (la Angles) Delta() vecint.Vector2i {
	switch la {
	case Horiz:
		return vecint.Vec2i(1, 0)
	case Vert:
		return vecint.Vec2i(0, 1)
	case DiagR:
		return vecint.Vec2i(1, 1)
	case DiagL:
		return vecint.Vec2i(-1, 1)
	}
	return vecint.Vec2i(0, 0)
}

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
	nl := cfg.NHypers

	n := (nl - 1) * (nl - 1) * (nl - 2) * int(AnglesN)

	lns := make([]Line, 0, n)

	for sy := range nl - 1 { // -1 so that there is min 2 len line, no wrap
		for sx := range nl - 1 {
			for ll := 2; ll <= nl; ll++ {
				for la := range AnglesN {
					d := la.Delta()
					ep := vecint.Vec2i(sx, sy).Add(d.MulScalar(ll))
					if ep.X < nl && ep.Y < nl && ep.X >= 0 && ep.Y >= 0 {
						lns = append(lns, Line{Angle: la, Length: ll, Start: vecint.Vec2i(sx, sy)})
					}
				}
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
	cfg := &ev.Configs
	in := ev.States["Input"]
	in.SetZeros()
	full := ev.States["Full"]
	full.SetZeros()
	inc := ln.Angle.Delta()
	pt := ln.Start
	nl := cfg.NHypers
	np := cfg.NPixels
	pc := np / 2
	pp := vecint.Vec2i(pc, pc)
	partp := float32(0)
	if cfg.Partial {
		partp = cfg.PartialRange.Min + ev.Rand.Float32()*cfg.PartialRange.Range()
	}
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
			if partp > 0 {
				if randx.BoolP32(partp, ev.Rand) {
					in.Set(1.0, pt.Y, pt.X, pn.Y, pn.X)
				}
			}
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
