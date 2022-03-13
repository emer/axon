// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"strconv"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/chans"
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
					"Layer.Act.Decay.Glong":     "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":    "0.5",  // 0.5 best
					"Layer.Act.Dend.GbarR":      "6",    // 6 best
					"Layer.Act.Dt.VmDendTau":    "5",    // 5 > 2.81 here but small effect
					"Layer.Act.NMDA.Gbar":       "0.15", // 0.15
					"Layer.Act.NMDA.ITau":       "100",  // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":        "100",  // 30 not good
					"Layer.Act.NMDA.MgC":        "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":       "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Learn.NeurCa.SpikeG": "8",
					"Layer.Learn.NeurCa.SynTau": "40", // 40 best in larger models
					"Layer.Learn.NeurCa.MTau":   "10",
					"Layer.Learn.NeurCa.PTau":   "40",
					"Layer.Learn.NeurCa.DTau":   "40",
					"Layer.Learn.NeurCa.VGCCCa": "20", // 20 seems reasonable, but not obviously better than 0
					"Layer.Learn.NeurCa.CaMax":  "100",
					"Layer.Learn.NeurCa.CaThr":  "0.2",
					"Layer.Learn.LrnNMDA.ITau":  "1",  // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":   "50", // urakubo = 30 > 20 but no major effect on PCA
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":         "0.1",      // 0.1 for SynSpkCa even though dwt equated
					"Prjn.SWt.Adapt.Lrate":          "0.08",     // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":            "0.5",      // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.SWt.Init.Var":             "0",        // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.KinaseCa.SpikeG":    "12",       // keep at 12 standard, adjust other things
					"Prjn.Learn.KinaseCa.Rule":      "SynSpkCa", // "SynNMDACa",
					"Prjn.Learn.KinaseCa.MTau":      "5",        // 5 > 10 test more
					"Prjn.Learn.KinaseCa.PTau":      "40",
					"Prjn.Learn.KinaseCa.DTau":      "40",
					"Prjn.Learn.KinaseDWt.TWindow":  "10",
					"Prjn.Learn.KinaseDWt.DMaxPct":  "0.5",
					"Prjn.Learn.KinaseDWt.TrlDecay": "0.0",
					"Prjn.Learn.KinaseDWt.DScale":   "1",
					"Prjn.Learn.XCal.On":            "true",
					"Prjn.Learn.XCal.PThrMin":       "0.05", // 0.05 best for objrec, higher worse
					"Prjn.Learn.XCal.LrnThr":        "0.01", // 0.05 best for objrec, higher worse
				}},
		},
	}},
}

// Extra state for neuron
type NeuronEx struct {
	SCaUpT     int     `desc:"time of last sending spike"`
	RCaUpT     int     `desc:"time of last recv spike"`
	Sp         float32 `desc:"sending poisson firing probability accumulator"`
	Rp         float32 `desc:"recv poisson firing probability accumulator"`
	NMDAGmg    float32 `desc:"NMDA mg-based blocking conductance"`
	Gvgcc      float32 `desc:"VGCC total conductance"`
	VGCCm      float32 `desc:"VGCC M gate -- activates with increasing Vm"`
	VGCCh      float32 `desc:"VGCC H gate -- deactivates with increasing Vm"`
	VGCCJcaPSD float32 `desc:"VGCC Ca calcium contribution to PSD"`
	VGCCJcaCyt float32 `desc:"VGCC Ca calcium contribution to Cyt"`
	Gak        float32 `desc:"AK total conductance"`
	AKm        float32 `desc:"AK M gate -- activates with increasing Vm"`
	AKh        float32 `desc:"AK H gate -- deactivates with increasing Vm"`
	LearnNow   float32 `desc:"when 0, it is time to learn according to theta cycle, otherwise increments up unless still -1 from init"`
}

func (nex *NeuronEx) Init() {
	nex.SCaUpT = -1
	nex.RCaUpT = -1
	nex.Sp = 1
	nex.Rp = 1
	nex.LearnNow = -1
	nex.NMDAGmg = 0
	nex.Gvgcc = 0
	nex.VGCCm = 0
	nex.VGCCh = 1
	nex.VGCCJcaPSD = 0
	nex.VGCCJcaCyt = 0
	nex.Gak = 0
	nex.AKm = 0
	nex.AKh = 1
}

func (ss *Sim) InitSyn(sy *axon.Synapse) {
	ss.Prjn.InitWtsSyn(sy, 0.5, 1)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	sly := net.AddLayer2D("Send", 1, 1, emer.Hidden).(*axon.Layer)
	rly := net.AddLayer2D("Recv", 1, 1, emer.Hidden).(*axon.Layer)
	pj := net.ConnectLayers(sly, rly, prjn.NewFull(), emer.Forward)
	net.Defaults()
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.SendNeur = &sly.Neurons[0]
	ss.RecvNeur = &rly.Neurons[0]
	ss.Prjn = pj.(*axon.Prjn)
}

// NeuronUpdt updates the neuron with whether send or recv spiked
func (ss *Sim) NeuronUpdt(sSpk, rSpk bool, ge, gi float32) {
	ly := ss.Net.LayerByName("Recv").(axon.AxonLayer).AsAxon()
	ac := &ly.Act
	sn := ss.SendNeur
	rn := ss.RecvNeur
	nex := &ss.NeuronEx

	vbio := chans.VToBio(rn.Vm) // dend

	if sSpk {
		sn.Spike = 1
		sn.ISI = 0
		nex.SCaUpT = ss.Time.CycleTot
	} else {
		sn.Spike = 0
		sn.ISI += 1
	}
	ly.Learn.LrnNMDA.SnmdaFmSpike(sn.Spike, &sn.SnmdaO, &sn.SnmdaI)
	if !ss.RGeClamp {
		if rSpk {
			rn.Spike = 1
			rn.ISI = 0
			nex.RCaUpT = ss.Time.CycleTot
		} else {
			rn.Spike = 0
			rn.ISI += 1
		}
		ly.Learn.LrnNMDA.SnmdaFmSpike(rn.Spike, &rn.SnmdaO, &rn.SnmdaI)
		rn.Ge = ge
		rn.GeSyn = ge
		rn.Gi = gi
		rn.GnmdaSyn = ge
		rn.Gnmda = ac.NMDA.Gnmda(rn.GnmdaSyn, rn.VmDend)
		rn.RnmdaSyn = ge
		mgg, cav := ac.NMDA.VFactors(rn.VmDend) // note: using Vm does NOT work well at all
		nex.NMDAGmg = mgg
		rn.RCa = rn.RnmdaSyn * mgg * cav
		rn.GnmdaRaw = 0
		if rn.Spike > 0 {
			rn.RCa += ly.Learn.NeurCa.VGCCCa
		}
		rn.RCa = ly.Learn.NeurCa.CaNorm(rn.RCa) // NOTE: RCa update from spike is 1 cycle behind Snmda
	} else {
		rn.GeRaw = ge
		rn.GnmdaRaw = ge
		ac.Dt.GeSynFmRaw(rn.GeRaw, &rn.GeSyn, ac.Init.Ge)
		rn.Ge = rn.GeSyn
		rn.Gi = gi
		ac.NMDAFmRaw(rn, 0)
		nex.NMDAGmg = ac.NMDA.MgGFmV(rn.VmDend)
	}
	rn.GABAB, rn.GABABx = ac.GABAB.GABAB(rn.GABAB, rn.GABABx, rn.Gi)
	rn.GgabaB = ac.GABAB.GgabaB(rn.GABAB, rn.VmDend)

	// nex.Gvgcc = ss.VGCC.Gvgcc(vmd, nex.VGCCm, nex.VGCCh)
	// dm, dh := ss.VGCC.DMHFmV(vmd, nex.VGCCm, nex.VGCCh)
	// nex.VGCCm += dm
	// nex.VGCCh += dh
	// isi := rn.ISI
	// if isi >= ac.Spike.VmR-1 && isi <= ac.Spike.VmR {
	// 	nex.VGCCm = 0 // resets
	// }

	// nex.Gak = ss.AK.Gak(nex.AKm, nex.AKh)
	// dm, dh = ss.AK.DMHFmV(vmd, nex.AKm, nex.AKh)
	// nex.AKm += dm
	// nex.AKh += dh

	// rn.Gk += nex.Gak
	rn.Ge += nex.Gvgcc + rn.Gnmda
	// if !ss.NMDAAxon {
	// 	rn.Ge += ss.NMDAGbar * float32(ss.Spine.States.NMDAR.G)
	// }
	rn.Gi += rn.GgabaB

	psd_pca := float32(1.7927e5 * 0.04) //  SVR_PSD
	cyt_pca := float32(1.0426e5 * 0.04) // SVR_CYT

	nex.VGCCJcaPSD = -vbio * psd_pca * nex.Gvgcc
	nex.VGCCJcaCyt = -vbio * cyt_pca * nex.Gvgcc

	ac.VmFmG(rn)
	if ss.RGeClamp {
		ac.ActFmG(rn)
	}
	ly.Learn.CaFmSpike(rn)
	ly.Learn.CaFmSpike(sn)

	ss.SynUpdt()
}

// SynUpdt updates the synapses based on current neuron state
func (ss *Sim) SynUpdt() {
	// ly := ss.Net.LayerByName("Recv").(axon.AxonLayer).AsAxon()
	pj := ss.Prjn
	kp := &pj.Learn.KinaseCa
	twin := pj.Learn.KinaseDWt.TWindow
	ctime := int32(ss.Time.CycleTot)

	pmsec := ss.MinusMsec + ss.PlusMsec

	sn := ss.SendNeur
	rn := ss.RecvNeur
	psy := &ss.SynNeur
	ssy := &ss.SynSpk
	nsy := &ss.SynNMDA
	// osy := &ss.SynOpt

	// NeurSpkCa continuous update: standard CHL s * r product form
	psy.CaM = ss.PGain * sn.CaM * rn.CaM
	psy.CaP = ss.PGain * sn.CaP * rn.CaP
	psy.CaD = ss.PGain * sn.CaD * rn.CaD
	if ss.Time.Cycle == pmsec {
		if pj.Learn.XCal.On {
			psy.DWt = pj.Learn.XCal.DWt(psy.CaP, psy.CaD)
		} else {
			psy.DWt = psy.CaP - psy.CaD
		}
	}

	sisi := int(sn.ISI)
	tdw := (sisi == twin || (sn.Spike > 0 && sisi < twin))
	risi := int(rn.ISI)
	tdw = tdw || (risi == twin || (rn.Spike > 0 && risi < twin))

	// SynSpkCa: continuous synaptic updating
	synspk := false
	if sn.Spike > 0 || rn.Spike > 0 {
		synspk = true
	}
	if synspk {
		ssy.Ca = kp.SpikeG * sn.CaSyn * rn.CaSyn
		ssy.CaUpT = ctime
	} else {
		ssy.Ca = 0
	}
	kp.FmCa(ssy.Ca, &ssy.CaM, &ssy.CaP, &ssy.CaD)

	// SynNMDACa: NMDA driven synaptic updating
	nsy.Ca = kp.SpikeG * sn.SnmdaO * rn.RCa
	kp.FmCa(nsy.Ca, &nsy.CaM, &nsy.CaP, &nsy.CaD)

	// // OptInteg form of SynSpkCa
	// if synspk || stdw || rtdw {
	// 	osy.CaM, osy.CaP, osy.CaD = kp.CurCa(ctime-1, osy.CaUpT, osy.CaM, osy.CaP, osy.CaD)
	// 	if stdw || rtdw {
	// 		osy.TDWt = kp.DWt(osy.CaP, osy.CaD) //
	// 	}
	// 	if synspk {
	// 		osy.Ca = kp.SpikeG * sn.CaSyn * rn.CaSyn
	// 	} else {
	// 		osy.Ca = 0
	// 	}
	// 	kp.FmCa(osy.Ca, &osy.CaM, &osy.CaP, &osy.CaD)
	// 	osy.CaUpT = ctime
	//
	// 	if osy.CaD > osy.CaDMax {
	// 		osy.CaDMax = osy.CaD
	// 	}
	// 	if ss.Time.CycleTot%ly.Learn.NeurCa.SynDWtInt == 0 {
	// 		pj.Learn.DWtFmTDWt(osy, 1)
	// 	}
	// }

	if tdw {
		pj.Learn.KinaseTDWt(ssy)
		pj.Learn.KinaseTDWt(nsy)
	}
	pj.Learn.CaDMax(ssy)
	pj.Learn.CaDMax(nsy)

	if ss.Time.Cycle == pmsec {
		axon.DecaySynCa(ssy, pj.Learn.KinaseDWt.TrlDecay)
		axon.DecaySynCa(nsy, pj.Learn.KinaseDWt.TrlDecay)
	}

	pj.Learn.DWtFmTDWt(ssy, 1)
	pj.Learn.DWtFmTDWt(nsy, 1)
}

func (ss *Sim) InitWts() {
	ssy := &ss.SynSpk
	nsy := &ss.SynNMDA
	osy := &ss.SynOpt
	ssy.DWt = 0
	nsy.DWt = 0
	osy.DWt = 0
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
	// dt.SetCellFloat("Gvgcc", row, float64(nex.Gvgcc))
	// dt.SetCellFloat("VGCCm", row, float64(nex.VGCCm))
	// dt.SetCellFloat("VGCCh", row, float64(nex.VGCCh))
	// dt.SetCellFloat("VGCCJcaPSD", row, float64(nex.VGCCJcaPSD))
	// dt.SetCellFloat("VGCCJcaCyt", row, float64(nex.VGCCJcaCyt))
	// dt.SetCellFloat("Gak", row, float64(nex.Gak))
	// dt.SetCellFloat("AKm", row, float64(nex.AKm))
	// dt.SetCellFloat("AKh", row, float64(nex.AKh))
	// dt.SetCellFloat("LearnNow", row, float64(nex.LearnNow))

	psy := &ss.SynNeur
	ssy := &ss.SynSpk
	nsy := &ss.SynNMDA
	osy := &ss.SynOpt

	dt.SetCellFloat("R_CaM", row, float64(rn.CaM))
	dt.SetCellFloat("R_CaP", row, float64(rn.CaP))
	dt.SetCellFloat("R_CaD", row, float64(rn.CaD))

	dt.SetCellFloat("S_CaM", row, float64(sn.CaM))
	dt.SetCellFloat("S_CaP", row, float64(sn.CaP))
	dt.SetCellFloat("S_CaD", row, float64(sn.CaD))

	ss.LogSyn(dt, row, "P_", psy)
	ss.LogSyn(dt, row, "X_", ssy)
	ss.LogSyn(dt, row, "N_", nsy)
	ss.LogSyn(dt, row, "O_", osy)
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
		// {"Gvgcc", etensor.FLOAT64, nil, nil},
		// {"VGCCm", etensor.FLOAT64, nil, nil},
		// {"VGCCh", etensor.FLOAT64, nil, nil},
		// {"VGCCJcaPSD", etensor.FLOAT64, nil, nil},
		// {"VGCCJcaCyt", etensor.FLOAT64, nil, nil},
		// {"Gak", etensor.FLOAT64, nil, nil},
		// {"AKm", etensor.FLOAT64, nil, nil},
		// {"AKh", etensor.FLOAT64, nil, nil},
		// {"LearnNow", etensor.FLOAT64, nil, nil},
		{"R_CaM", etensor.FLOAT64, nil, nil},
		{"R_CaP", etensor.FLOAT64, nil, nil},
		{"R_CaD", etensor.FLOAT64, nil, nil},
		{"S_CaM", etensor.FLOAT64, nil, nil},
		{"S_CaP", etensor.FLOAT64, nil, nil},
		{"S_CaD", etensor.FLOAT64, nil, nil},
	}

	ss.ConfigSynapse(&sch, "P_")
	ss.ConfigSynapse(&sch, "X_")
	ss.ConfigSynapse(&sch, "N_")
	ss.ConfigSynapse(&sch, "O_")

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
		case cn == "X_CaP" || cn == "X_CaD":
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
		case cn == "X_DWt":
			plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		// case strings.Contains(cn, "DWt"):
		// 	plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		// case strings.HasPrefix(cn, "X_"):
		// 	plt.SetColParams(cn, eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
		default:
			plt.SetColParams(cn, eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
		}
	}

	return plt
}
