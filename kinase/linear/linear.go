// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"
	"log/slog"
	"math/rand"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/glm"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/kinase"
)

// Linear performs a linear regression to approximate the synaptic Ca
// integration between send and recv neurons.
type Linear struct {
	// Kinase Neuron params
	Neuron kinase.NeurCaParams

	// Kinase Synapse params
	Synapse kinase.SynCaParams

	// total number of cycles (1 MSec) to run per learning trial
	NCycles int `min:"10" default:"200"`

	// number of plus cycles
	PlusCycles int `default:"50"`

	// NumBins is the number of bins to accumulate spikes over NCycles
	NumBins int `default:"8"`

	// CyclesPerBin = NCycles / NumBins
	CyclesPerBin int `edit:"-"`

	// MaxHz is the maximum firing rate to sample in minus, plus phases
	MaxHz int `default:"120"`

	// StepHz is the step size for sampling Hz
	StepHz int `default:"10"`

	// NTrials is number of trials per Hz case
	NTrials int `default:"100"`

	// Total Trials is number of trials for all data
	TotalTrials int `edit:"-"`

	// Sending neuron
	Send Neuron

	// Receiving neuron
	Recv Neuron

	// Standard synapse values
	StdSyn Synapse

	// Linear synapse values
	LinearSyn Synapse

	// ErrDWt is the target error dwt: PlusHz - MinusHz
	ErrDWt float32

	// binned integration of send, recv spikes
	SpikeBins []float32

	// Data to fit the regression
	Data table.Table
}

func (ls *Linear) Defaults() {
	ls.Neuron.Defaults()
	ls.Synapse.Defaults()
	ls.Synapse.CaScale = 6 // 12 is too fast relative to prior std learning rates
	ls.NCycles = 200
	ls.PlusCycles = 50
	ls.CyclesPerBin = 50
	ls.MaxHz = 100
	ls.StepHz = 10 // note: 5 gives same results
	ls.NTrials = 2 // 20 "
	ls.NumBins = 8
	ls.Update()
}

func (ls *Linear) Update() {
	ls.CyclesPerBin = ls.NCycles / ls.NumBins
	ls.Neuron.Dt.PDTauForNCycles(ls.NCycles)
	ls.Synapse.Dt.PDTauForNCycles(ls.NCycles)
	nhz := ls.MaxHz / ls.StepHz
	ls.TotalTrials = nhz * nhz * nhz * nhz * ls.NTrials
	ls.SpikeBins = make([]float32, ls.NumBins)
	ls.Send.SpikeBins = make([]float32, ls.NumBins)
	ls.Recv.SpikeBins = make([]float32, ls.NumBins)
}

func (ls *Linear) Init() {
	ls.Send.Init()
	ls.Recv.Init()
	ls.StdSyn.Init()
	ls.LinearSyn.Init()
	ls.InitTable()
}

func (ls *Linear) InitTable() {
	if ls.Data.NumColumns() > 0 {
		return
	}
	nneur := ls.NumBins
	ls.Data.AddIntColumn("Trial")
	ls.Data.AddFloat64Column("Hz", 4)
	ls.Data.AddFloat64Column("State", nneur)
	ls.Data.AddFloat64Column("StdCa", 2)
	ls.Data.AddFloat64Column("PredCa", 2)
	ls.Data.AddFloat64Column("ErrCa", 2)
	ls.Data.AddFloat64Column("SSE") // total SSE
	ls.Data.SetNumRows(ls.TotalTrials)
}

func (ls *Linear) StartTrial() {
	ls.Send.StartTrial()
	ls.Recv.StartTrial()
}

// Neuron has Neuron state
type Neuron struct {
	// Neuron spiking (0,1)
	Spike float32

	// Neuron probability of spiking
	SpikeP float32

	// CaSyn is spike-driven calcium trace for synapse-level Ca-driven learning:
	// exponential integration of SpikeG * Spike at SynTau time constant (typically 30).
	// Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for
	// the synaptic trace driving credit assignment in learning.
	// Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically,
	// and determines time window where pre * post spiking must overlap to drive learning.
	CaSyn float32

	// neuron-level spike-driven Ca integration
	CaSpkM, CaSpkP, CaSpkD float32

	TotalSpikes float32

	// binned count of spikes, for regression learning
	SpikeBins []float32
}

func (kn *Neuron) Init() {
	kn.Spike = 0
	kn.SpikeP = 1
	kn.CaSyn = 0
	kn.CaSpkM = 0
	kn.CaSpkP = 0
	kn.CaSpkD = 0
	kn.StartTrial()
}

func (kn *Neuron) StartTrial() {
	kn.TotalSpikes = 0
	for i := range kn.SpikeBins {
		kn.SpikeBins[i] = 0
	}
}

// Cycle does one cycle of neuron updating, with given exponential spike interval
// based on target spiking firing rate.
func (ls *Linear) Cycle(nr *Neuron, expInt float32, cyc int) {
	nr.Spike = 0
	bin := cyc / ls.CyclesPerBin
	if expInt > 0 {
		nr.SpikeP *= rand.Float32()
		if nr.SpikeP <= expInt {
			nr.Spike = 1
			nr.SpikeP = 1
			nr.TotalSpikes += 1
			nr.SpikeBins[bin] += 1
		}
	}
	ls.Neuron.CaFromSpike(nr.Spike, &nr.CaSyn, &nr.CaSpkM, &nr.CaSpkP, &nr.CaSpkD)
}

// Synapse has Synapse state
type Synapse struct {
	CaSyn float32

	// CaM is first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
	CaM float32

	// CaP is shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
	CaP float32

	// CaD is longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
	CaD float32

	// DWt is the CaP - CaD
	DWt float32
}

func (ks *Synapse) Init() {
	ks.CaSyn = 0
	ks.CaM = 0
	ks.CaP = 0
	ks.CaD = 0
	ks.DWt = 0
}

// Run generates data
func (ls *Linear) Run() {
	nhz := ls.MaxHz / ls.StepHz
	hz := make([]float32, nhz)
	i := 0
	for h := float32(ls.StepHz); h <= float32(ls.MaxHz); h += float32(ls.StepHz) {
		hz[i] = h
		i++
	}
	row := 0
	for smi := 0; smi < nhz; smi++ {
		sendMinusHz := hz[smi]
		for spi := 0; spi < nhz; spi++ {
			sendPlusHz := hz[spi]
			for rmi := 0; rmi < nhz; rmi++ {
				recvMinusHz := hz[rmi]
				for rpi := 0; rpi < nhz; rpi++ {
					recvPlusHz := hz[rpi]

					for ti := 0; ti < ls.NTrials; ti++ {
						ls.Trial(sendMinusHz, sendPlusHz, recvMinusHz, recvPlusHz, ti, row)
						row++
					}
				}
			}
		}
	}
}

func (ls *Linear) SetSynState(sy *Synapse, row int) {
	ls.Data.Column("StdCa").SetFloatRowCell(float64(sy.CaP), row, 0)
	ls.Data.Column("StdCa").SetFloatRowCell(float64(sy.CaD), row, 1)
}

func (ls *Linear) SetBins(sn, rn *Neuron, off, row int) {
	for i, s := range sn.SpikeBins {
		r := rn.SpikeBins[i]
		bs := (r * s) / 10.0
		ls.SpikeBins[i] = bs
		ls.Data.Column("State").SetFloatRowCell(float64(bs), row, off+i)
	}
}

// Trial runs one trial
func (ls *Linear) Trial(sendMinusHz, sendPlusHz, recvMinusHz, recvPlusHz float32, ti, row int) {
	// ls.ErrDWt = (plusHz - minusHz) / 100

	ls.Data.Column("Trial").SetFloatRow(float64(ti), row)
	ls.Data.Column("Hz").SetFloatRowCell(float64(sendMinusHz), row, 0)
	ls.Data.Column("Hz").SetFloat(float64(sendPlusHz), row, 1)
	ls.Data.Column("Hz").SetFloat(float64(recvMinusHz), row, 2)
	ls.Data.Column("Hz").SetFloat(float64(recvPlusHz), row, 3)

	minusCycles := ls.NCycles - ls.PlusCycles

	ls.StartTrial()
	cyc := 0
	for phs := 0; phs < 2; phs++ {
		var maxcyc int
		var rhz, shz float32
		switch phs {
		case 0:
			rhz = recvMinusHz
			shz = sendMinusHz
			maxcyc = minusCycles
		case 1:
			rhz = recvPlusHz
			shz = sendPlusHz
			maxcyc = ls.PlusCycles
		}
		Rint := math32.Exp(-1000.0 / float32(rhz))
		Sint := math32.Exp(-1000.0 / float32(shz))
		for t := 0; t < maxcyc; t++ {
			ls.Cycle(&ls.Send, Sint, cyc)
			ls.Cycle(&ls.Recv, Rint, cyc)

			ls.StdSyn.CaSyn = ls.Send.CaSyn * ls.Recv.CaSyn
			ls.Synapse.FromCa(ls.StdSyn.CaSyn, &ls.StdSyn.CaM, &ls.StdSyn.CaP, &ls.StdSyn.CaD)
			cyc++
		}
	}
	ls.StdSyn.DWt = ls.StdSyn.CaP - ls.StdSyn.CaD

	ls.SetSynState(&ls.StdSyn, row)

	ls.SetBins(&ls.Send, &ls.Recv, 0, row)
}

// Regress runs the linear regression on the data
func (ls *Linear) Regress() {
	r := glm.NewGLM()
	err := r.SetTable(&ls.Data, "State", "StdCa", "PredCa", "ErrCa")
	if err != nil {
		slog.Error(err.Error())
		return
	}
	r.DepNames = []string{"CaP", "CaD"}
	r.L1Cost = 0.1
	r.L2Cost = 0.1
	r.StopTolerance = 0.00001
	r.ZeroOffset = true

	// NBins = 4
	// r.Coeff.Values = []float64{
	// 	0.05, 0.25, 0.5, 0.6, 0, // linear progression
	// 	0.25, 0.5, 0.5, 0.25, 0} // hump in the middle

	// NBins = 8, 200+50 cycles
	// r.Coeff.Values = []float64{
	// 	0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 1.0, 1.0, 0, // linear progression
	// 	0.5, 0.65, 0.75, 0.9, 0.9, 0.9, 0.65, 0.55, .0} // hump in the middle

	// NBins = 8, 280+70 cycles
	r.Coeff.Values = []float64{
		0.0, 0.1, 0.23, 0.35, 0.45, 0.55, 0.75, 0.75, 0, // linear progression
		0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3, .0} // hump in the middle

	fmt.Println(r.Coeffs())

	r.Run()

	fmt.Println(r.Variance())
	fmt.Println(r.Coeffs())

	/*
		for vi := 0; vi < 2; vi++ {
			r := new(regression.Regression)
			r.SetObserved("CaD")
			for bi := 0; bi < ls.NumBins; bi++ {
				r.SetVar(bi, fmt.Sprintf("Bin_%d", bi))
			}

			for row := 0; row < ls.Data.Rows; row++ {
				st := ls.Data.Tensor("State", row).(*tensor.Float64)
				cad := ls.Data.TensorFloat1D("StdCa", row, vi)
				r.Train(regression.DataPoint(cad, st.Values))
			}
			r.Run()
			fmt.Printf("Regression formula:\n%v\n", r.Formula)
			fmt.Printf("Variance observed = %v\nVariance Predicted = %v", r.Varianceobserved, r.VariancePredicted)
			fmt.Printf("\nR2 = %v\n", r.R2)
			str := "{"
			for ci := 0; ci <= ls.NumBins; ci++ {
				str += fmt.Sprintf("%8.6g, ", r.Coeff(ci))
			}
			fmt.Println(str + "}")
		}
	*/
	ls.Data.SaveCSV("linear_data.tsv", tensor.Tab, table.Headers)
}
