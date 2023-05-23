// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package main

import (
	"log"
	"strconv"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
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
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":         "0.1",         // 0.1 for SynSpkCa even though dwt equated
					"Prjn.SWts.Adapt.LRate":         "0.08",        // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWts.Init.SPct":           "0.5",         // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.SWts.Init.Var":            "0",           // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.KinaseCa.SpikeG":    "12",          // keep at 12 standard, adjust other things
					"Prjn.Learn.KinaseCa.NMDAG":     "40",          // just to match SynSpk..
					"Prjn.Learn.KinaseCa.Rule":      "SynSpkTheta", // "SynNMDACa",
					"Prjn.Learn.KinaseCa.MTau":      "5",           // 5 > 10 test more
					"Prjn.Learn.KinaseCa.PTau":      "40",
					"Prjn.Learn.KinaseCa.DTau":      "40",
					"Prjn.Learn.KinaseDWt.TWindow":  "10",
					"Prjn.Learn.KinaseDWt.DMaxPct":  "0.5",
					"Prjn.Learn.KinaseDWt.TrlDecay": "0.0",
					"Prjn.Learn.KinaseDWt.DScale":   "1",
					"Prjn.Learn.XCal.On":            "false",
					"Prjn.Learn.XCal.PThrMin":       "0.05", // 0.05 best for objrec, higher worse
					"Prjn.Learn.XCal.LrnThr":        "0.01", // 0.05 best for objrec, higher worse
				}},
		},
	}},
}

// Extra state for neuron
type NeuronEx struct {
	SCaUpT   int     `desc:"time of last sending spike"`
	RCaUpT   int     `desc:"time of last recv spike"`
	Sp       float32 `desc:"sending poisson firing probability accumulator"`
	Rp       float32 `desc:"recv poisson firing probability accumulator"`
	NMDAGmg  float32 `desc:"NMDA mg-based blocking conductance"`
	LearnNow float32 `desc:"when 0, it is time to learn according to theta cycle, otherwise increments up unless still -1 from init"`
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
	ss.Prjn.InitWtsSyn(sy, 0.5, 1)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	sly := net.AddLayer2D("Send", 1, 1, axon.SuperLayer).(*axon.Layer)
	rly := net.AddLayer2D("Recv", 1, 1, axon.SuperLayer).(*axon.Layer)
	pj := net.ConnectLayers(sly, rly, prjn.NewFull(), emer.Forward)
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SendNeur = &sly.Neurons[0]
	ss.RecvNeur = &rly.Neurons[0]
	ss.Prjn = pj.(*axon.Prjn)
}

// NeuronUpdt updates the neuron with whether send or recv spiked
func (ss *Sim) NeuronUpdt(sSpk, rSpk bool, ge, gi float32) {
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
	ly.Params.Learn.LrnNMDA.SnmdaFmSpike(sn.Spike, &sn.SnmdaO, &sn.SnmdaI)

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
		ac.Dt.GeSynFmRaw(rn.GeRaw, &rn.GeSyn, ac.Init.GeBase)
		rn.Ge = rn.GeSyn
		rn.Gi = gi
		ac.NMDAFmRaw(rn, 0)
		nex.NMDAGmg = ac.NMDA.MgGFmV(rn.VmDend)
	}
	rn.GABAB, rn.GABABx = ac.GABAB.GABAB(rn.GABAB, rn.GABABx, rn.Gi)
	rn.GgabaB = ac.GABAB.GgabaB(rn.GABAB, rn.VmDend)

	rn.Ge += rn.Gvgcc + rn.Gnmda
	rn.Gi += rn.GgabaB

	ac.VmFmG(rn)
	if ss.RGeClamp {
		ac.ActFmG(rn)
	}
	ly.Params.Learn.LrnNMDAFmRaw(rn, 0)

	ly.Params.Learn.CaFmSpike(rn)
	ly.Params.Learn.CaFmSpike(sn)

	ss.SynUpdt()
}

// SynUpdt updates the synapses based on current neuron state
func (ss *Sim) SynUpdt() {
	// ly := ss.Net.LayByName("Recv")
	pj := ss.Prjn
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
		kp.FmCa(sst.Ca, &sst.CaM, &sst.CaP, &sst.CaD)
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
	kp.FmCa(ssc.Ca, &ssc.CaM, &ssc.CaP, &ssc.CaD)

	// SynNMDACont: NMDA driven synaptic updating
	snc.Ca = kp.NMDAG * sn.SnmdaO * rn.RCa
	kp.FmCa(snc.Ca, &snc.CaM, &snc.CaP, &snc.CaD)

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

	pj.Params.Learn.DWtFmTDWt(ssc, 1)
	pj.Params.Learn.DWtFmTDWt(snc, 1)
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

func (ss *Sim) LogSyn(dt *etable.Table, row int, pre string, sy *axon.Synapse) {
	dt.SetCellFloat(pre+"Ca", row, float64(sy.Ca))
	dt.SetCellFloat(pre+"CaM", row, float64(sy.CaM))
	dt.SetCellFloat(pre+"CaP", row, float64(sy.CaP))
	dt.SetCellFloat(pre+"CaD", row, float64(sy.CaD))
	dt.SetCellFloat(pre+"CaDMax", row, float64(sy.CaDMax))
	dt.SetCellFloat(pre+"TDWt", row, float64(sy.TDWt))
	dt.SetCellFloat(pre+"DWt", row, float64(sy.DWt))
	dt.SetCellFloat(pre+"Wt", row, float64(sy.Wt))
}

// LogState records data for given cycle
func (ss *Sim) LogState(dt *etable.Table, row, trl, cyc int) {
	sn := ss.SendNeur
	rn := ss.RecvNeur
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Cycle", row, float64(cyc))
	dt.SetCellFloat("SSpike", row, float64(ss.SpikeDisp*sn.Spike))
	dt.SetCellFloat("RSpike", row, float64(ss.SpikeDisp*rn.Spike))

	dt.SetCellFloat("SnmdaO", row, float64(sn.SnmdaO))
	dt.SetCellFloat("SnmdaI", row, float64(sn.SnmdaI))

	dt.SetCellFloat("Ge", row, float64(rn.Ge))
	dt.SetCellFloat("Inet", row, float64(rn.Inet))
	dt.SetCellFloat("Vm", row, float64(rn.Vm))
	dt.SetCellFloat("Act", row, float64(rn.Act))
	dt.SetCellFloat("Gk", row, float64(rn.Gk))
	dt.SetCellFloat("ISI", row, float64(rn.ISI))
	dt.SetCellFloat("VmDend", row, float64(rn.VmDend))
	dt.SetCellFloat("Gnmda", row, float64(rn.Gnmda))
	dt.SetCellFloat("RnmdaSyn", row, float64(rn.RnmdaSyn))
	dt.SetCellFloat("RCa", row, float64(rn.RCa))
	// dt.SetCellFloat("NMDAGmg", row, float64(nex.NMDAGmg))
	// dt.SetCellFloat("GABAB", row, float64(rn.GABAB))
	// dt.SetCellFloat("GgabaB", row, float64(rn.GgabaB))
	dt.SetCellFloat("Gvgcc", row, float64(rn.Gvgcc))
	dt.SetCellFloat("VgccM", row, float64(rn.VgccM))
	dt.SetCellFloat("VgccH", row, float64(rn.VgccH))
	dt.SetCellFloat("VgccCa", row, float64(rn.VgccCa))
	dt.SetCellFloat("Gak", row, float64(rn.Gak))
	// dt.SetCellFloat("LearnNow", row, float64(nex.LearnNow))

	nst := &ss.SynNeurTheta
	sst := &ss.SynSpkTheta
	ssc := &ss.SynSpkCont
	snc := &ss.SynNMDACont

	dt.SetCellFloat("R_CaM", row, float64(rn.CaM))
	dt.SetCellFloat("R_CaP", row, float64(rn.CaP))
	dt.SetCellFloat("R_CaD", row, float64(rn.CaD))

	dt.SetCellFloat("S_CaM", row, float64(sn.CaM))
	dt.SetCellFloat("S_CaP", row, float64(sn.CaP))
	dt.SetCellFloat("S_CaD", row, float64(sn.CaD))

	ss.LogSyn(dt, row, "NST_", nst)
	ss.LogSyn(dt, row, "SST_", sst)
	ss.LogSyn(dt, row, "SSC_", ssc)
	ss.LogSyn(dt, row, "SNC_", snc)
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "Kinase Equations Table")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Cond", etensor.STRING, nil, nil},
		{"ErrDWt", etensor.FLOAT64, nil, nil},
		{"Trial", etensor.FLOAT64, nil, nil},
		{"Cycle", etensor.FLOAT64, nil, nil},
		{"SSpike", etensor.FLOAT64, nil, nil},
		{"RSpike", etensor.FLOAT64, nil, nil},

		{"SnmdaO", etensor.FLOAT64, nil, nil},
		{"SnmdaI", etensor.FLOAT64, nil, nil},

		{"Ge", etensor.FLOAT64, nil, nil},
		{"Inet", etensor.FLOAT64, nil, nil},
		{"Vm", etensor.FLOAT64, nil, nil},
		{"Act", etensor.FLOAT64, nil, nil},
		{"Gk", etensor.FLOAT64, nil, nil},
		{"ISI", etensor.FLOAT64, nil, nil},
		{"VmDend", etensor.FLOAT64, nil, nil},
		{"Gnmda", etensor.FLOAT64, nil, nil},
		{"RnmdaSyn", etensor.FLOAT64, nil, nil},
		{"RCa", etensor.FLOAT64, nil, nil},
		// {"NMDAGmg", etensor.FLOAT64, nil, nil},
		// {"GABAB", etensor.FLOAT64, nil, nil},
		// {"GgabaB", etensor.FLOAT64, nil, nil},
		{"Gvgcc", etensor.FLOAT64, nil, nil},
		{"VgccM", etensor.FLOAT64, nil, nil},
		{"VgccH", etensor.FLOAT64, nil, nil},
		{"VgccCa", etensor.FLOAT64, nil, nil},
		{"Gak", etensor.FLOAT64, nil, nil},
		// {"LearnNow", etensor.FLOAT64, nil, nil},
		{"R_CaM", etensor.FLOAT64, nil, nil},
		{"R_CaP", etensor.FLOAT64, nil, nil},
		{"R_CaD", etensor.FLOAT64, nil, nil},
		{"S_CaM", etensor.FLOAT64, nil, nil},
		{"S_CaP", etensor.FLOAT64, nil, nil},
		{"S_CaD", etensor.FLOAT64, nil, nil},
	}

	ss.ConfigSynapse(&sch, "NST_")
	ss.ConfigSynapse(&sch, "SST_")
	ss.ConfigSynapse(&sch, "SSC_")
	ss.ConfigSynapse(&sch, "SNC_")

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSynapse(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "Ca", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaM", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaP", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaD", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaDMax", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "TDWt", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "DWt", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Wt", etensor.FLOAT64, nil, nil})
}

func (ss *Sim) ConfigTrialPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Kinase Equations Trial Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)

	for _, cn := range dt.ColNames {
		if cn == "Cycle" {
			continue
		}
		switch {
		case strings.Contains(cn, "DWt"):
			plt.SetColParams(cn, eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
		case cn == "SSC_CaP" || cn == "SSC_CaD":
			plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		default:
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		}
	}
	// plt.SetColParams("SynCSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	// plt.SetColParams("SynOSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)

	plt.SetColParams("SSpike", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("RSpike", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)

	return plt
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Kinase Equations Run Plot"
	plt.Params.XAxisCol = "Trial"
	// plt.Params.LegendCol = "Cond"
	plt.SetTable(dt)
	plt.Params.Points = true
	plt.Params.Lines = false

	for _, cn := range dt.ColNames {
		switch {
		case strings.Contains(cn, "DWt"):
			plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		default:
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		}
	}

	return plt
}

func (ss *Sim) ConfigDWtPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Kinase Equations DWt Plot"
	plt.Params.XAxisCol = "ErrDWt"
	plt.Params.LegendCol = "Cond"
	plt.Params.Scale = 3
	plt.SetTable(dt)
	plt.Params.Points = true
	plt.Params.Lines = false

	for _, cn := range dt.ColNames {
		switch {
		case cn == "ErrDWt":
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1.5)
		case cn == "SSC_DWt":
			plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		case strings.Contains(cn, "_DWt"):
			plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		// case strings.HasPrefix(cn, "X_"):
		// 	plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		default:
			plt.SetColParams(cn, eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
		}
	}

	return plt
}
