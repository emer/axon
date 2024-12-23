// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

import (
	"math"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/lab"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/kinase"
)

type SynCaPlot struct {

	// Ca time constants
	SynCa kinase.SynCaParams `display:"inline"`
	CaDt  kinase.CaDtParams  `display:"inline"`
	Minit float64
	Pinit float64
	Dinit float64

	// adjustment to dt to account for discrete time updating
	MdtAdj float64 `default:"0,0.11"`

	// adjustment to dt to account for discrete time updating
	PdtAdj float64 `default:"0,0.03"`

	// adjustment to dt to account for discrete time updating
	DdtAdj float64 `default:"0,0.03"`

	// number of time steps
	TimeSteps int

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *SynCaPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("SynCa")
	pl.Tabs = tabs

	pl.SynCa.Defaults()
	pl.CaDt.Defaults()
	pl.Minit = 0.7
	pl.Pinit = 0.5
	pl.Dinit = 0.3
	pl.MdtAdj = 0
	pl.PdtAdj = 0
	pl.DdtAdj = 0
	pl.TimeSteps = 1000
	pl.Update()
}

// Update updates computed values
func (pl *SynCaPlot) Update() {
}

// CaAtT computes the 3 Ca values at (currentTime + ti), assuming 0
// new Ca incoming (no spiking). It uses closed-form exponential functions.
func (pl *SynCaPlot) CaAtT(ti int32, caM, caP, caD *float32) {
	kp := &pl.CaDt

	t := float32(ti)
	mdt := kp.MDt
	pdt := kp.PDt
	ddt := kp.DDt
	// if kp.ExpAdj.IsTrue() { // adjust for discrete
	// mdt *= 1.11
	// pdt *= 1.03
	// ddt *= 1.03
	// }
	mi := *caM
	pi := *caP
	di := *caD

	*caM = mi * math32.FastExp(-t*mdt)

	em := math32.FastExp(t * mdt)
	ep := math32.FastExp(t * pdt)

	*caP = pi*math32.FastExp(-t*pdt) - (pdt*mi*math32.FastExp(-t*(mdt+pdt))*(em-ep))/(pdt-mdt)

	epd := math32.FastExp(t * (pdt + ddt))
	emd := math32.FastExp(t * (mdt + ddt))
	emp := math32.FastExp(t * (mdt + pdt))

	*caD = pdt*ddt*mi*math32.FastExp(-t*(mdt+pdt+ddt))*(ddt*(emd-epd)+(pdt*(epd-emp))+mdt*(emp-emd))/((mdt-pdt)*(mdt-ddt)*(pdt-ddt)) - ddt*pi*math32.FastExp(-t*(pdt+ddt))*(ep-math32.FastExp(t*ddt))/(ddt-pdt) + di*math32.FastExp(-t*ddt)
}

// CurCa returns the current Ca* values, dealing with updating for
// optimized spike-time update versions.
// ctime is current time in msec, and utime is last update time (-1 if never)
// to avoid running out of float32 precision, ctime should be reset periodically
// along with the Ca values -- in axon this happens during SlowAdapt.
func (pl *SynCaPlot) CurCa(ctime, utime float32, caM, caP, caD *float32) {
	kp := &pl.SynCa

	isi := int32(ctime - utime)
	if isi <= 0 {
		return
	}
	for j := int32(0); j < isi; j++ {
		kp.FromCa(0, caM, caP, caD) // just decay to 0
	}
	return
}

// TimeRun runs the equation.
func (pl *SynCaPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("Ca(t)")

	nv := 200
	mi := pl.Minit
	pi := pl.Pinit
	di := pl.Dinit
	mdt := float64(pl.CaDt.MDt) * (1.0 + pl.MdtAdj)
	pdt := float64(pl.CaDt.PDt) * (1.0 + pl.PdtAdj)
	ddt := float64(pl.CaDt.DDt) * (1.0 + pl.DdtAdj)
	for ti := range nv {
		t := float64(ti)
		m := pl.Minit * math.Exp(-t*mdt)

		em := math.Exp(t * mdt)
		ep := math.Exp(t * pdt)

		p := pl.Pinit*math.Exp(-t*pdt) - (pdt*pl.Minit*math.Exp(-t*(mdt+pdt))*(em-ep))/(pdt-mdt)

		epd := math.Exp(t * (pdt + ddt))
		emd := math.Exp(t * (mdt + ddt))
		emp := math.Exp(t * (mdt + pdt))

		d := pdt*ddt*pl.Minit*math.Exp(-t*(mdt+pdt+ddt))*(ddt*(emd-epd)+(pdt*(epd-emp))+mdt*(emp-emd))/((mdt-pdt)*(mdt-ddt)*(pdt-ddt)) - ddt*pl.Pinit*math.Exp(-t*(pdt+ddt))*(ep-math.Exp(t*ddt))/(ddt-pdt) + pl.Dinit*math.Exp(-t*ddt)

		// test eqs:
		caM := float32(pl.Minit)
		caP := float32(pl.Pinit)
		caD := float32(pl.Dinit)
		pl.CaAtT(int32(ti), &caM, &caP, &caD)
		m = float64(caM)
		p = float64(caP)
		d = float64(caD)

		caM = float32(pl.Minit)
		caP = float32(pl.Pinit)
		caD = float32(pl.Dinit)
		pl.CurCa(float32(ti), 0, &caM, &caP, &caD)
		mi4 := float64(caM)
		pi4 := float64(caP)
		di4 := float64(caD)

		dir.Float64("t", nv).SetFloat1D(t, ti)
		dir.Float64("mi", nv).SetFloat1D(mi, ti)
		dir.Float64("pi", nv).SetFloat1D(pi, ti)
		dir.Float64("di", nv).SetFloat1D(di, ti)
		dir.Float64("mi4", nv).SetFloat1D(mi4, ti)
		dir.Float64("pi4", nv).SetFloat1D(pi4, ti)
		dir.Float64("di4", nv).SetFloat1D(di4, ti)
		dir.Float64("m", nv).SetFloat1D(m, ti)
		dir.Float64("p", nv).SetFloat1D(p, ti)
		dir.Float64("d", nv).SetFloat1D(d, ti)

		mi += float64(pl.CaDt.MDt) * (0 - mi)
		pi += float64(pl.CaDt.PDt) * (mi - pi)
		di += float64(pl.CaDt.DDt) * (pi - di)

	}
	plot.SetFirstStylerTo(dir.Float64("t"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"m", "p", "d"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "SynCa"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *SynCaPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
