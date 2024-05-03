// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package main

import (
	"log"
	"strconv"
	"strings"

	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

// ParamSets for basic parameters
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Acts.Decay.Glong":  "0.6",  // 0.6
					"Layer.Acts.Dend.GbarExp": "0.5",  // 0.5 best
					"Layer.Acts.Dend.GbarR":   "6",    // 6 best
					"Layer.Acts.Dt.VmDendTau": "5",    // 5 > 2.81 here but small effect
					"Layer.Acts.NMDA.Gbar":    "0.15", // 0.15
					"Layer.Acts.NMDA.ITau":    "100",  // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Acts.NMDA.Tau":     "100",  // 30 not good
					"Layer.Acts.NMDA.MgC":     "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Acts.NMDA.Voff":    "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Acts.Noise.On":     "true",
					"Layer.Acts.Noise.Ge":     "0.02", // induces significant variability in Rn Ge clamp firing
					"Layer.Acts.Noise.Gi":     "0.05",
					"Layer.Acts.VGCC.Gbar":    "0.02",
					"Layer.Acts.AK.Gbar":      "2",
					// "Layer.Acts.AK.Hf":           "5.5",
					// "Layer.Acts.AK.Mf":           "0.2",
					"Layer.Learn.NeurCa.SpikeG": "8",
					"Layer.Learn.NeurCa.SynTau": "40", // 40 best in larger models
					"Layer.Learn.NeurCa.MTau":   "10",
					"Layer.Learn.NeurCa.PTau":   "40",
					"Layer.Learn.NeurCa.DTau":   "40",
					"Layer.Learn.NeurCa.CaMax":  "200",
					"Layer.Learn.NeurCa.CaThr":  "0.05",
					"Layer.Learn.LrnNMDA.ITau":  "1",  // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":   "50", // urakubo = 30 > 20 but no major effect on PCA
				}},
			{Sel: "Path", Desc: "basic path params",
				Params: params.Params{
					"Path.Learn.LRate.Base":         "0.1",         // 0.1 for SynSpkCa even though dwt equated
					"Path.SWts.Adapt.LRate":         "0.08",        // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Path.SWts.Init.SPct":           "0.5",         // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Path.SWts.Init.Var":            "0",           // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Path.Learn.KinaseCa.SpikeG":    "12",          // keep at 12 standard, adjust other things
					"Path.Learn.KinaseCa.NMDAG":     "40",          // just to match SynSpk..
					"Path.Learn.KinaseCa.Rule":      "SynSpkTheta", // "SynNMDACa",
					"Path.Learn.KinaseCa.MTau":      "5",           // 5 > 10 test more
					"Path.Learn.KinaseCa.PTau":      "40",
					"Path.Learn.KinaseCa.DTau":      "40",
					"Path.Learn.KinaseDWt.TWindow":  "10",
					"Path.Learn.KinaseDWt.DMaxPct":  "0.5",
					"Path.Learn.KinaseDWt.TrlDecay": "0.0",
					"Path.Learn.KinaseDWt.DScale":   "1",
					"Path.Learn.XCal.On":            "false",
					"Path.Learn.XCal.PThrMin":       "0.05", // 0.05 best for objrec, higher worse
					"Path.Learn.XCal.LrnThr":        "0.01", // 0.05 best for objrec, higher worse
				}},
		},
	}},
}

// Extra state for neuron
type NeuronEx struct {

	// time of last sending spike
	SCaUpT int

	// time of last recv spike
	RCaUpT int

	// sending poisson firing probability accumulator
	Sp float32

	// recv poisson firing probability accumulator
	Rp float32

	// NMDA mg-based blocking conductance
	NMDAGmg float32

	// when 0, it is time to learn according to theta cycle, otherwise increments up unless still -1 from init
	LearnNow float32
}

func (nex *NeuronEx) Init() {
	nex.SCaUpT = -1
	nex.RCaUpT = -1
	nex.Sp = 1
	nex.Rp = 1
	nex.NMDAGmg = 0
	nex.LearnNow = -1
}

////////////////////////////////////////////////////////////////////////

// RGeStimForHzMap is the strength of GeStim G clamp to obtain a given R firing rate
var RGeStimForHzMap = map[int]float32{
	25:  .09,
	50:  .12,
	100: .15,
}

func RGeStimForHz(hz float32) float32 {
	var gel, geh, hzl, hzh float32
	switch {
	case hz <= 25:
		gel = 0
		geh = RGeStimForHzMap[25]
		hzl = 0
		hzh = 25
	case hz <= 50:
		gel = RGeStimForHzMap[25]
		geh = RGeStimForHzMap[50]
		hzl = 25
		hzh = 50
	case hz <= 100:
		gel = RGeStimForHzMap[50]
		geh = RGeStimForHzMap[100]
		hzl = 50
		hzh = 100
	default:
		gel = RGeStimForHzMap[100]
		geh = 2 * gel
		hzl = 100
		hzh = 200
	}
	return (gel + ((hz-hzl)/(hzh-hzl))*(geh-gel))
}

////////////////////////////////////////////////////////////////////////
// Sim

func (ss *Sim) InitSyn(sy *axon.Synapse) {
	ss.Path.InitWtsSyn(sy, 0.5, 1)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	sly := net.AddLayer2D("Send", 1, 1, axon.SuperLayer).(*axon.Layer)
	rly := net.AddLayer2D("Recv", 1, 1, axon.SuperLayer).(*axon.Layer)
	pj := net.ConnectLayers(sly, rly, paths.NewFull(), emer.Forward)
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SendNeur = &sly.Neurons[0]
	ss.RecvNeur = &rly.Neurons[0]
	ss.Path = pj.(*axon.Path)
}

// NeuronUpdate updates the neuron with whether send or recv spiked
func (ss *Sim) NeuronUpdate(sSpk, rSpk bool, ge, gi float32) {
	ly := ss.Net.LayByName("Recv")
	ac := &ly.Params.Act
	sn := ss.SendNeur
	rn := ss.RecvNeur
	nex := &ss.NeuronEx

	if sSpk {
		sn.Spike = 1
		sn.ISI = 0
		nex.SCaUpT = ss.Context.CycleTot
	} else {
		sn.Spike = 0
		sn.ISI += 1
	}
	ly.Params.Learn.LrnNMDA.SnmdaFromSpike(sn.Spike, &sn.SnmdaO, &sn.SnmdaI)

	//	Recv

	ac.GeNoise(rn)
	ge += rn.GeNoise
	ac.GiNoise(rn)
	gi += rn.GiNoise

	if !ss.RGeClamp {
		if rSpk {
			rn.Spike = 1
			rn.ISI = 0
			nex.RCaUpT = ss.Context.CycleTot
		} else {
			rn.Spike = 0
			rn.ISI += 1
		}
		rn.Ge = ge
		rn.GeSyn = ge
		rn.Gi = gi
		rn.GnmdaSyn = ge
		rn.Gnmda = ac.NMDA.Gnmda(rn.GnmdaSyn, rn.VmDend)
		rn.RnmdaSyn = ge
		mgg, cav := ac.NMDA.VFactors(rn.VmDend) // note: using Vm does NOT work well at all
		nex.NMDAGmg = mgg
		rn.RCa = rn.RnmdaSyn * mgg * cav
		rn.RCa = ly.Params.Learn.NeurCa.CaNorm(rn.RCa) // NOTE: RCa update from spike is 1 cycle behind Snmda
	} else {
		rn.GeRaw = ge
		ac.Dt.GeSynFromRaw(rn.GeRaw, &rn.GeSyn, ac.Init.GeBase)
		rn.Ge = rn.GeSyn
		rn.Gi = gi
		ac.NMDAFromRaw(rn, 0)
		nex.NMDAGmg = ac.NMDA.MgGFromV(rn.VmDend)
	}
	rn.GABAB, rn.GABABx = ac.GABAB.GABAB(rn.GABAB, rn.GABABx, rn.Gi)
	rn.GgabaB = ac.GABAB.GgabaB(rn.GABAB, rn.VmDend)

	rn.Ge += rn.Gvgcc + rn.Gnmda
	rn.Gi += rn.GgabaB

	ac.VmFromG(rn)
	if ss.RGeClamp {
		ac.ActFromG(rn)
	}
	ly.Params.Learn.LrnNMDAFromRaw(rn, 0)

	ly.Params.Learn.CaFromSpike(rn)
	ly.Params.Learn.CaFromSpike(sn)

	ss.SynUpdate()
}

// SynUpdate updates the synapses based on current neuron state
func (ss *Sim) SynUpdate() {
	// ly := ss.Net.LayByName("Recv")
	pj := ss.Path
	kp := &pj.Params.Learn.KinaseCa
	twin := pj.Params.Learn.KinaseDWt.TWindow
	ctime := int32(ss.Context.CycleTot)

	pmsec := ss.MinusMsec + ss.PlusMsec

	sn := ss.SendNeur
	rn := ss.RecvNeur

	nst := &ss.SynNeurTheta
	sst := &ss.SynSpkTheta
	ssc := &ss.SynSpkCont
	snc := &ss.SynNMDACont

	//////////////////////////////
	// Theta

	// NeurSpkTheta continuous update: standard CHL s * r product form
	nst.CaM = ss.PGain * sn.CaM * rn.CaM
	nst.CaP = ss.PGain * sn.CaP * rn.CaP
	nst.CaD = ss.PGain * sn.CaD * rn.CaD

	synspk := false
	if sn.Spike > 0 || rn.Spike > 0 {
		synspk = true
	}

	// SynSpkTheta
	if synspk {
		sst.CaM, sst.CaP, sst.CaD = kp.CurCa(ctime-1, sst.CaUpT, sst.CaM, sst.CaP, sst.CaD)
		sst.Ca = kp.SpikeG * sn.CaSyn * rn.CaSyn
		kp.FromCa(sst.Ca, &sst.CaM, &sst.CaP, &sst.CaD)
		sst.CaUpT = ctime
	}

	if ss.Context.Cycle == pmsec {
		if pj.Params.Learn.XCal.On {
			nst.DWt = pj.Params.Learn.XCal.DWt(nst.CaP, nst.CaD)
			sst.DWt = pj.Params.Learn.XCal.DWt(sst.CaP, sst.CaD)
		} else {
			nst.DWt = nst.CaP - nst.CaD
			sst.DWt = sst.CaP - sst.CaD
		}
	}

	//////////////////////////////
	// Cont

	sisi := int(sn.ISI)
	tdw := (sisi == twin || (sn.Spike > 0 && sisi < twin))
	risi := int(rn.ISI)
	tdw = tdw || (risi == twin || (rn.Spike > 0 && risi < twin))

	// SynSpkCont: continuous synaptic updating
	if synspk {
		ssc.Ca = kp.SpikeG * sn.CaSyn * rn.CaSyn
		ssc.CaUpT = ctime
	} else {
		ssc.Ca = 0
	}
	kp.FromCa(ssc.Ca, &ssc.CaM, &ssc.CaP, &ssc.CaD)

	// SynNMDACont: NMDA driven synaptic updating
	snc.Ca = kp.NMDAG * sn.SnmdaO * rn.RCa
	kp.FromCa(snc.Ca, &snc.CaM, &snc.CaP, &snc.CaD)

	if tdw {
		pj.Params.Learn.KinaseTDWt(ssc)
		pj.Params.Learn.KinaseTDWt(snc)
	}
	pj.Params.Learn.CaDMax(ssc)
	pj.Params.Learn.CaDMax(snc)

	if ss.Context.Cycle == pmsec {
		axon.DecaySynCa(ssc, pj.Params.Learn.KinaseDWt.TrlDecay)
		axon.DecaySynCa(snc, pj.Params.Learn.KinaseDWt.TrlDecay)
	}

	pj.Params.Learn.DWtFromTDWt(ssc, 1)
	pj.Params.Learn.DWtFromTDWt(snc, 1)
}

func (ss *Sim) InitWts() {
	nst := &ss.SynNeurTheta
	sst := &ss.SynSpkTheta
	ssc := &ss.SynSpkCont
	snc := &ss.SynNMDACont
	nst.DWt = 0
	sst.DWt = 0
	ssc.DWt = 0
	snc.DWt = 0
}

///////////////////////////////////////////////////////////////////
//  Logging

func (ss *Sim) LogSyn(dt *table.Table, row int, pre string, sy *axon.Synapse) {
	dt.SetFloat(pre+"Ca", row, float64(sy.Ca))
	dt.SetFloat(pre+"CaM", row, float64(sy.CaM))
	dt.SetFloat(pre+"CaP", row, float64(sy.CaP))
	dt.SetFloat(pre+"CaD", row, float64(sy.CaD))
	dt.SetFloat(pre+"CaDMax", row, float64(sy.CaDMax))
	dt.SetFloat(pre+"TDWt", row, float64(sy.TDWt))
	dt.SetFloat(pre+"DWt", row, float64(sy.DWt))
	dt.SetFloat(pre+"Wt", row, float64(sy.Wt))
}

// LogState records data for given cycle
func (ss *Sim) LogState(dt *table.Table, row, trl, cyc int) {
	sn := ss.SendNeur
	rn := ss.RecvNeur
	dt.SetFloat("Trial", row, float64(trl))
	dt.SetFloat("Cycle", row, float64(cyc))
	dt.SetFloat("SSpike", row, float64(ss.SpikeDisp*sn.Spike))
	dt.SetFloat("RSpike", row, float64(ss.SpikeDisp*rn.Spike))

	dt.SetFloat("SnmdaO", row, float64(sn.SnmdaO))
	dt.SetFloat("SnmdaI", row, float64(sn.SnmdaI))

	dt.SetFloat("Ge", row, float64(rn.Ge))
	dt.SetFloat("Inet", row, float64(rn.Inet))
	dt.SetFloat("Vm", row, float64(rn.Vm))
	dt.SetFloat("Act", row, float64(rn.Act))
	dt.SetFloat("Gk", row, float64(rn.Gk))
	dt.SetFloat("ISI", row, float64(rn.ISI))
	dt.SetFloat("VmDend", row, float64(rn.VmDend))
	dt.SetFloat("Gnmda", row, float64(rn.Gnmda))
	dt.SetFloat("RnmdaSyn", row, float64(rn.RnmdaSyn))
	dt.SetFloat("RCa", row, float64(rn.RCa))
	// dt.SetFloat("NMDAGmg", row, float64(nex.NMDAGmg))
	// dt.SetFloat("GABAB", row, float64(rn.GABAB))
	// dt.SetFloat("GgabaB", row, float64(rn.GgabaB))
	dt.SetFloat("Gvgcc", row, float64(rn.Gvgcc))
	dt.SetFloat("VgccM", row, float64(rn.VgccM))
	dt.SetFloat("VgccH", row, float64(rn.VgccH))
	dt.SetFloat("VgccCa", row, float64(rn.VgccCa))
	dt.SetFloat("Gak", row, float64(rn.Gak))
	// dt.SetFloat("LearnNow", row, float64(nex.LearnNow))

	nst := &ss.SynNeurTheta
	sst := &ss.SynSpkTheta
	ssc := &ss.SynSpkCont
	snc := &ss.SynNMDACont

	dt.SetFloat("R_CaM", row, float64(rn.CaM))
	dt.SetFloat("R_CaP", row, float64(rn.CaP))
	dt.SetFloat("R_CaD", row, float64(rn.CaD))

	dt.SetFloat("S_CaM", row, float64(sn.CaM))
	dt.SetFloat("S_CaP", row, float64(sn.CaP))
	dt.SetFloat("S_CaD", row, float64(sn.CaD))

	ss.LogSyn(dt, row, "NST_", nst)
	ss.LogSyn(dt, row, "SST_", sst)
	ss.LogSyn(dt, row, "SSC_", ssc)
	ss.LogSyn(dt, row, "SNC_", snc)
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "Kinase Equations Table")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddStringColumn(Cond")
	dt.AddFloat64Column("ErrDWt")
	dt.AddFloat64Column("Trial")
	dt.AddFloat64Column("Cycle")
	dt.AddFloat64Column("SSpike")
	dt.AddFloat64Column("RSpike")

	dt.AddFloat64Column("SnmdaO")
	dt.AddFloat64Column("SnmdaI")

	dt.AddFloat64Column("Ge")
	dt.AddFloat64Column("Inet")
	dt.AddFloat64Column("Vm")
	dt.AddFloat64Column("Act")
	dt.AddFloat64Column("Gk")
	dt.AddFloat64Column("ISI")
	dt.AddFloat64Column("VmDend")
	dt.AddFloat64Column("Gnmda")
	dt.AddFloat64Column("RnmdaSyn")
	dt.AddFloat64Column("RCa")
		// {"NMDAGmg")
		// {"GABAB")
		// {"GgabaB")
	dt.AddFloat64Column("Gvgcc")
	dt.AddFloat64Column("VgccM")
	dt.AddFloat64Column("VgccH")
	dt.AddFloat64Column("VgccCa")
	dt.AddFloat64Column("Gak")
		// {"LearnNow")
	dt.AddFloat64Column("R_CaM")
	dt.AddFloat64Column("R_CaP")
	dt.AddFloat64Column("R_CaD")
	dt.AddFloat64Column("S_CaM")
	dt.AddFloat64Column("S_CaP")
	dt.AddFloat64Column("S_CaD")

	ss.ConfigSynapse(dt, "NST_")
	ss.ConfigSynapse(dt, "SST_")
	ss.ConfigSynapse(dt, "SSC_")
	ss.ConfigSynapse(dt, "SNC_")

	dt.SetNumRows(0)
}

func (ss *Sim) ConfigSynapse(dt *table.Table, pre string) {
	dt.AddFloat64Column(pre + "Ca")
	dt.AddFloat64Column(pre + "CaM")
	dt.AddFloat64Column(pre + "CaP")
	dt.AddFloat64Column(pre + "CaD")
	dt.AddFloat64Column(pre + "CaDMax")
	dt.AddFloat64Column(pre + "TDWt")
	dt.AddFloat64Column(pre + "DWt")
	dt.AddFloat64Column(pre + "Wt")
}

func (ss *Sim) ConfigTrialPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Kinase Equations Trial Plot"
	plt.Params.XAxisColumn = "Cycle"
	plt.SetTable(dt)

	for _, cn := range dt.ColumnNames {
		if cn == "Cycle" {
			continue
		}
		switch {
		case strings.Contains(cn, "DWt"):
			plt.SetColParams(cn, plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
		case cn == "SSC_CaP" || cn == "SSC_CaD":
			plt.SetColParams(cn, plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
		default:
			plt.SetColParams(cn, plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
		}
	}
	// plt.SetColParams("SynCSpkCaM", plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
	// plt.SetColParams("SynOSpkCaM", plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)

	plt.SetColParams("SSpike", plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("RSpike", plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)

	return plt
}

func (ss *Sim) ConfigRunPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Kinase Equations Run Plot"
	plt.Params.XAxisColumn = "Trial"
	// plt.Params.LegendCol = "Cond"
	plt.SetTable(dt)
	plt.Params.Points = true
	plt.Params.Lines = false

	for _, cn := range dt.ColumnNames {
		switch {
		case strings.Contains(cn, "DWt"):
			plt.SetColParams(cn, plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
		default:
			plt.SetColParams(cn, plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
		}
	}

	return plt
}

func (ss *Sim) ConfigDWtPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Kinase Equations DWt Plot"
	plt.Params.XAxisColumn = "ErrDWt"
	plt.Params.LegendCol = "Cond"
	plt.Params.Scale = 3
	plt.SetTable(dt)
	plt.Params.Points = true
	plt.Params.Lines = false

	for _, cn := range dt.ColumnNames {
		switch {
		case cn == "ErrDWt":
			plt.SetColParams(cn, plotview.Off, plotview.FixMin, -1, plotview.FixMax, 1.5)
		case cn == "SSC_DWt":
			plt.SetColParams(cn, plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
		case strings.Contains(cn, "_DWt"):
			plt.SetColParams(cn, plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
		// case strings.HasPrefix(cn, "X_"):
		// 	plt.SetColParams(cn, plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
		default:
			plt.SetColParams(cn, plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
		}
	}

	return plt
}
