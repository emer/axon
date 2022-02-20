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
					"Layer.Act.Decay.Glong":  "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp": "0.5",  // 0.5 best
					"Layer.Act.Dend.GbarR":   "6",    // 6 best
					"Layer.Act.Dt.VmDendTau": "5",    // 5 > 2.81 here but small effect
					"Layer.Act.NMDA.Gbar":    "0.15", // 0.15
					"Layer.Act.NMDA.ITau":    "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":     "100",  // 30 not good
					"Layer.Act.NMDA.MgC":     "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":    "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Act.Dend.VGCCCa":  "20",   // 20 seems reasonable, but not obviously better than 0
					"Layer.Act.Dend.CaMax":   "100",
					"Layer.Act.Dend.CaThr":   "0.2",
					"Layer.Act.Dend.CaVm":    "false",
					"Layer.Learn.SpkCa.MTau": "10",
					"Layer.Learn.SpkCa.PTau": "40",
					"Layer.Learn.SpkCa.DTau": "40",
				}},
		},
	}},
}

// Extra state for neuron
type NeuronEx struct {
	SSpikeT    int     `desc:"time of last sending spike"`
	RSpikeT    int     `desc:"time of last recv spike"`
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
	nex.SSpikeT = -1
	nex.RSpikeT = -1
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

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	sly := net.AddLayer2D("Send", 1, 1, emer.Hidden).(*axon.Layer)
	rly := net.AddLayer2D("Recv", 1, 1, emer.Hidden).(*axon.Layer)
	net.Defaults()
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.SendNeur = &sly.Neurons[0]
	ss.RecvNeur = &rly.Neurons[0]
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
		nex.SSpikeT = ss.Time.CycleTot
		inh := (1 - sn.SnmdaI)
		sn.SnmdaO += inh * (1 - sn.SnmdaO)
		sn.SnmdaI += inh
	} else {
		sn.Spike = 0
		sn.ISI += 1
		sn.SnmdaO -= ac.NMDA.Dt * sn.SnmdaO
		sn.SnmdaI -= ac.NMDA.IDt * sn.SnmdaI
	}
	if !ss.RGeClamp {
		if rSpk {
			rn.Spike = 1
			rn.ISI = 0
			nex.RSpikeT = ss.Time.CycleTot
			inh := (1 - rn.SnmdaI)
			rn.SnmdaO += inh * (1 - rn.SnmdaO)
			rn.SnmdaI += inh
		} else {
			rn.Spike = 0
			rn.ISI += 1
			rn.SnmdaO -= ac.NMDA.Dt * rn.SnmdaO
			rn.SnmdaI -= ac.NMDA.IDt * rn.SnmdaI
		}
	}

	// note: Ge should only
	rn.GeRaw = ge
	rn.GnmdaRaw = ge
	ac.Dt.GeSynFmRaw(rn.GeRaw, &rn.GeSyn, ac.Init.Ge)
	rn.Ge = rn.GeSyn
	rn.Gi = gi
	ac.NMDAFmRaw(rn, 0)

	vmd := rn.VmDend
	// if ss.DendVm {
	// 	vmd = rn.VmDend
	// }

	nex.NMDAGmg = ac.NMDA.MgGFmV(vmd)
	rn.GABAB, rn.GABABx = ac.GABAB.GABAB(rn.GABAB, rn.GABABx, rn.Gi)
	rn.GgabaB = ac.GABAB.GgabaB(rn.GABAB, vmd)

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
	ly.Learn.SpkCaFmSpike(rn)
	ly.Learn.SpkCaFmSpike(sn)

	ss.SynUpdt()
}

// SynUpdt updates the synapses based on current neuron state
func (ss *Sim) SynUpdt() {
	kp := &ss.Kinase

	sn := ss.SendNeur
	rn := ss.RecvNeur
	psy := &ss.SynNeur
	ssy := &ss.SynSpk
	nsy := &ss.SynNNMDA
	// osy := &ss.SynOpt

	// this is standard CHL s * r product form
	psy.CaM = ss.PGain * sn.SpkCaM * rn.SpkCaM
	psy.CaP = ss.PGain * sn.SpkCaP * rn.SpkCaP
	psy.CaD = ss.PGain * sn.SpkCaD * rn.SpkCaD
	psy.DWt = psy.CaP - psy.CaD

	// SynSpkCA
	ssy.Ca = sn.SpkCaM * rn.SpkCaM
	kp.FmCa(ssy.Ca, &ssy.CaM, &ssy.CaP, &ssy.CaD)
	ssy.DWt = kp.DWt(ssy.CaP, ssy.CaD)

	// SynNMDACa
	nsy.Ca = sn.SnmdaO * rn.RCa
	kp.FmCa(nsy.Ca, &nsy.CaM, &nsy.CaP, &nsy.CaD)
	nsy.DWt = kp.DWt(nsy.CaP, nsy.CaD)

	// // optimized
	// if cSpk > 0 {
	// 	kp.FuntCaFmSpike(cSpk, &cISI, &oSpkCaM, &oSpkCaP, &oSpkCaD)
	// 	oCaM, oCaP, oCaD = kp.CurCaFmISI(cISI, oSpkCaM, oSpkCaP, oSpkCaD)
	// } else if cISI >= 0 {
	// 	cISI += 1
	// 	oCaM, oCaP, oCaD = kp.CurCaFmISI(cISI, oSpkCaM, oSpkCaP, oSpkCaD)
	// }
	// oDWt = kp.DWt(oCaP, oCaD)
}

// Log records data for given cycle
func (ss *Sim) Log(dt *etable.Table, row, cyc int) {
	sn := ss.SendNeur
	rn := ss.RecvNeur
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
	nsy := &ss.SynNNMDA
	osy := &ss.SynOpt

	dt.SetCellFloat("R_CaM", row, float64(rn.SpkCaM))
	dt.SetCellFloat("R_CaP", row, float64(rn.SpkCaP))
	dt.SetCellFloat("R_CaD", row, float64(rn.SpkCaD))

	dt.SetCellFloat("S_CaM", row, float64(sn.SpkCaM))
	dt.SetCellFloat("S_CaP", row, float64(sn.SpkCaP))
	dt.SetCellFloat("S_CaD", row, float64(sn.SpkCaD))

	dt.SetCellFloat("P_CaM", row, float64(psy.CaM))
	dt.SetCellFloat("P_CaP", row, float64(psy.CaP))
	dt.SetCellFloat("P_CaD", row, float64(psy.CaD))
	dt.SetCellFloat("P_DWt", row, float64(psy.DWt))
	dt.SetCellFloat("P_Wt", row, float64(psy.Wt))

	dt.SetCellFloat("X_CaM", row, float64(ssy.CaM))
	dt.SetCellFloat("X_CaP", row, float64(ssy.CaP))
	dt.SetCellFloat("X_CaD", row, float64(ssy.CaD))
	dt.SetCellFloat("X_DWt", row, float64(ssy.DWt))
	dt.SetCellFloat("X_Wt", row, float64(ssy.Wt))

	dt.SetCellFloat("N_Ca", row, float64(nsy.Ca))
	dt.SetCellFloat("N_CaM", row, float64(nsy.CaM))
	dt.SetCellFloat("N_CaP", row, float64(nsy.CaP))
	dt.SetCellFloat("N_CaD", row, float64(nsy.CaD))
	dt.SetCellFloat("N_DWt", row, float64(nsy.DWt))
	dt.SetCellFloat("N_Wt", row, float64(nsy.Wt))

	dt.SetCellFloat("O_Ca", row, float64(osy.Ca))
	dt.SetCellFloat("O_CaM", row, float64(osy.CaM))
	dt.SetCellFloat("O_CaP", row, float64(osy.CaP))
	dt.SetCellFloat("O_CaD", row, float64(osy.CaD))
	dt.SetCellFloat("O_DWt", row, float64(osy.DWt))
	dt.SetCellFloat("O_Wt", row, float64(osy.Wt))
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "Kinase Opt Table")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
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

		{"P_CaM", etensor.FLOAT64, nil, nil},
		{"P_CaP", etensor.FLOAT64, nil, nil},
		{"P_CaD", etensor.FLOAT64, nil, nil},
		{"P_DWt", etensor.FLOAT64, nil, nil},
		{"P_Wt", etensor.FLOAT64, nil, nil},

		{"X_CaM", etensor.FLOAT64, nil, nil},
		{"X_CaP", etensor.FLOAT64, nil, nil},
		{"X_CaD", etensor.FLOAT64, nil, nil},
		{"X_DWt", etensor.FLOAT64, nil, nil},
		{"X_Wt", etensor.FLOAT64, nil, nil},

		{"N_Ca", etensor.FLOAT64, nil, nil},
		{"N_CaM", etensor.FLOAT64, nil, nil},
		{"N_CaP", etensor.FLOAT64, nil, nil},
		{"N_CaD", etensor.FLOAT64, nil, nil},
		{"N_DWt", etensor.FLOAT64, nil, nil},
		{"N_Wt", etensor.FLOAT64, nil, nil},

		{"O_CaM", etensor.FLOAT64, nil, nil},
		{"O_CaP", etensor.FLOAT64, nil, nil},
		{"O_CaD", etensor.FLOAT64, nil, nil},
		{"O_DWt", etensor.FLOAT64, nil, nil},
		{"O_Wt", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Kinase Learning Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// plt.Params.Points = true

	for _, cn := range dt.ColNames {
		if cn == "Cycle" {
			continue
		}
		if strings.Contains(cn, "DWt") {
			plt.SetColParams(cn, eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
		} else {
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		}
	}
	// plt.SetColParams("SynCSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	// plt.SetColParams("SynOSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)

	plt.SetColParams("SSpike", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("RSpike", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)

	plt.SetColParams("N_Ca", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N_CaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N_CaP", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N_CaD", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N_DWt", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N_Wt", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)

	return plt
}
