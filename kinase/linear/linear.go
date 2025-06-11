// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"
	"image"
	"log/slog"
	"math/rand"
	"slices"
	"strings"

	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/plot/plots"
	"cogentcore.org/lab/stats/glm"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/axon/v2/kinase"
)

// Linear performs a linear regression to approximate the synaptic Ca
// integration between send and recv neurons.
type Linear struct {
	// Kinase CaSpike params
	CaSpike kinase.CaSpikeParams `display:"no-inline" new-window:"+"`

	// SynCa20 uses 20 msec time bin integration.
	SynCa20 bool

	// total number of cycles (1 MSec) to run per learning trial
	Cycles int `min:"10" default:"200"`

	// number of plus cycles
	PlusCycles int `default:"50"`

	// NumBins is the number of bins to accumulate spikes over Cycles
	NumBins int `default:"8"`

	// CyclesPerBin = Cycles / NumBins
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
	CaBins []float32

	// Data to fit the regression
	Data table.Table
}

func (ls *Linear) Defaults() {
	ls.CaSpike.Defaults()
	ls.Cycles = 200
	ls.PlusCycles = 50
	ls.CyclesPerBin = 10
	ls.MaxHz = 100
	ls.StepHz = 10  // note: 5 gives same results
	ls.NTrials = 10 // 20 "
	ls.NumBins = ls.Cycles / ls.CyclesPerBin
	ls.Update()
}

func (ls *Linear) Update() {
	ls.NumBins = ls.Cycles / ls.CyclesPerBin
	// ls.CaSpike.Dt.PDTauForNCycles(ls.Cycles)
	// ls.Synapse.Dt.PDTauForNCycles(ls.Cycles)
	nhz := ls.MaxHz / ls.StepHz
	ls.TotalTrials = nhz * nhz * nhz * nhz * ls.NTrials
	ls.CaBins = make([]float32, ls.NumBins)
	ls.Send.CaBins = make([]float32, ls.NumBins)
	ls.Recv.CaBins = make([]float32, ls.NumBins)
}

func (ls *Linear) Init() {
	ls.Data.Init()
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
	ls.Data.AddFloat64Column("Bins", nneur)
	ls.Data.AddFloat64Column("SynCa", 2)
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
	CaM, CaP, CaD float32

	TotalSpikes float32

	// binned count of spikes, for regression learning
	CaBins []float32
}

func (kn *Neuron) Init() {
	kn.Spike = 0
	kn.SpikeP = 1
	kn.CaSyn = 0
	kn.CaM = 0
	kn.CaP = 0
	kn.CaD = 0
	kn.StartTrial()
}

func (kn *Neuron) StartTrial() {
	kn.TotalSpikes = 0
	for i := range kn.CaBins {
		kn.CaBins[i] = 0
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
		}
	}
	nr.CaSyn += ls.CaSpike.CaSynDt * (ls.CaSpike.SpikeCaSyn*nr.Spike - nr.CaSyn)
	nr.CaBins[bin] += (nr.CaSyn / float32(ls.CyclesPerBin))
	ls.CaSpike.CaMFromSpike(nr.Spike, &nr.CaM, &nr.CaP, &nr.CaD)
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
	ls.Data.Column("SynCa").SetFloatRow(float64(sy.CaP), row, 0)
	ls.Data.Column("SynCa").SetFloatRow(float64(sy.CaD), row, 1)
}

func (ls *Linear) SetBins(sn, rn *Neuron, off, row int) {
	ls.CaBins[0] = rn.CaBins[0] * sn.CaBins[0]
	for i := 2; i < ls.NumBins; i++ {
		if ls.SynCa20 {
			ls.CaBins[i] = 0.25 * (rn.CaBins[i] + rn.CaBins[i-1]) * (sn.CaBins[i] + sn.CaBins[i-1])
		} else {
			ls.CaBins[i] = rn.CaBins[i] * sn.CaBins[i]
		}
		ls.Data.Column("Bins").SetFloatRow(float64(ls.CaBins[i]), row, off+i)
	}
}

// Trial runs one trial
func (ls *Linear) Trial(sendMinusHz, sendPlusHz, recvMinusHz, recvPlusHz float32, ti, row int) {
	// ls.ErrDWt = (plusHz - minusHz) / 100

	ls.Data.Column("Trial").SetFloatRow(float64(ti), row, 0)
	ls.Data.Column("Hz").SetFloatRow(float64(sendMinusHz), row, 0)
	ls.Data.Column("Hz").SetFloatRow(float64(sendPlusHz), row, 1)
	ls.Data.Column("Hz").SetFloatRow(float64(recvMinusHz), row, 2)
	ls.Data.Column("Hz").SetFloatRow(float64(recvPlusHz), row, 3)

	minusCycles := ls.Cycles - ls.PlusCycles

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

			ls.StdSyn.CaSyn = 8 * ls.Send.CaSyn * ls.Recv.CaSyn // 12 is standard CaGain factor
			ls.CaSpike.Dt.FromCa(ls.StdSyn.CaSyn, &ls.StdSyn.CaM, &ls.StdSyn.CaP, &ls.StdSyn.CaD)
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
	err := r.SetTable(&ls.Data, "Bins", "SynCa", "PredCa", "ErrCa")
	if err != nil {
		slog.Error(err.Error())
		return
	}
	r.DepNames = []string{"CaP", "CaD"}
	r.L1Cost = 0.1
	r.L2Cost = 0.1
	r.StopTolerance = 0.00001
	r.ZeroOffset = true

	// default coefficients are the current ones..
	cp := make([]float32, ls.NumBins)
	cd := make([]float32, ls.NumBins)
	kinase.CaBinWts(ls.PlusCycles, cp, cd)
	cp = append(cp, 0)
	cd = append(cd, 0)

	cp64 := make([]float64, ls.NumBins+1)
	cd64 := make([]float64, ls.NumBins+1)
	for i := range ls.NumBins + 1 {
		cp64[i] = float64(cp[i])
		cd64[i] = float64(cd[i])
	}

	r.Coeff.Values = append(cp64, cd64...)

	// NBins = 8, 200+50 cycles for CaSyn
	// r.Coeff.Values = []float64{
	// 	0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 1.9, 3.0, 0, // big at the end; insensitive to start
	// 	0.35, 0.65, 0.95, 1.25, 1.25, 1.25, 1.125, 1.0, .0} // up and down

	// NBins = 12, 300+50 cycles for CaSyn
	// r.Coeff.Values = []float64{
	// 	0, 0, 0, 0, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 1.9, 3.0, 0, // big at the end; insensitive to start
	// 	0, 0, 0, 0, 0.35, 0.65, 0.95, 1.25, 1.25, 1.25, 1.125, 1.0, .0} // up and down

	prc := func() string {
		s := "CaP:\t"
		for i := range ls.NumBins {
			s += fmt.Sprintf("%7.4f\t", r.Coeff.Values[i])
		}
		s += "\nCaD:\t"
		for i := range ls.NumBins {
			s += fmt.Sprintf("%7.4f\t", r.Coeff.Values[i+ls.NumBins+1])
		}
		return s
	}

	start := prc()
	startCaP := slices.Clone(r.Coeff.Values[:ls.NumBins])
	startCaD := slices.Clone(r.Coeff.Values[ls.NumBins+1 : 2*ls.NumBins+1])

	r.Run()

	fmt.Println(r.Variance())
	fmt.Println("Starting Coeff:")
	fmt.Println(start)
	fmt.Println("Final Coeff:")
	fmt.Println(prc())

	endCaP := slices.Clone(r.Coeff.Values[:ls.NumBins])
	endCaD := slices.Clone(r.Coeff.Values[ls.NumBins+1 : 2*ls.NumBins+1])

	estr := "synca10"
	if ls.SynCa20 {
		estr = "synca20"
	}
	esfn := strings.ToLower(estr)

	plt := plot.New()
	plt.SetSize(image.Point{1280, 1024})
	plots.NewLine(plt, tensor.NewFloat64FromValues(startCaP...)).Styler(func(s *plot.Style) {
		s.Plot.Scale = 2
		s.Plot.Title = "CaP Linear Regression Coefficients: " + estr
		s.Plot.XAxis.Label = "Bins"
		s.Label = "Starting"
	})
	plots.NewLine(plt, tensor.NewFloat64FromValues(endCaP...)).Styler(func(s *plot.Style) {
		s.Label = "Final"
	})
	imagex.Save(plt.RenderImage(), "plot-coefficients-cap-"+esfn+".png")

	plt = plot.New()
	plt.SetSize(image.Point{1280, 1024})
	plots.NewLine(plt, tensor.NewFloat64FromValues(startCaD...)).Styler(func(s *plot.Style) {
		s.Plot.Scale = 2
		s.Plot.Title = "CaD Linear Regression Coefficients: " + estr
		s.Plot.XAxis.Label = "Bins"
		s.Label = "Starting"
	})
	plots.NewLine(plt, tensor.NewFloat64FromValues(endCaD...)).Styler(func(s *plot.Style) {
		s.Label = "Final"
	})
	imagex.Save(plt.RenderImage(), "plot-coefficients-cad-"+esfn+".png")

	/*
		for vi := 0; vi < 2; vi++ {
			r := new(regression.Regression)
			r.SetObserved("CaD")
			for bi := 0; bi < ls.NumBins; bi++ {
				r.SetVar(bi, fmt.Sprintf("Bin_%d", bi))
			}

			for row := 0; row < ls.Data.Rows; row++ {
				st := ls.Data.Tensor("Bins", row).(*tensor.Float64)
				cad := ls.Data.TensorFloat1D("SynCa", row, vi)
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
	// ls.Data.SaveCSV("linear_data.tsv", tensor.Tab, table.Headers)
}
