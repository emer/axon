// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

func init() {
	// must lock main thread for gpu!
	runtime.LockOSThread()
}

// number of distinct sets of learning parameters to test
const NLrnPars = 1

// Note: subsequent params applied after Base
var ParamSets = params.Sets{
	{Name: "Base", Desc: "base testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Act.Gbar.L":      "0.2",
					"Layer.Learn.RLRate.On": "false",
					"Layer.Inhib.Layer.FB":  "0.5",
				}},
			{Sel: "Prjn", Desc: "for reproducibility, identical weights",
				Params: params.Params{
					"Prjn.SWt.Init.Var": "0",
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
		},
		"InhibOff": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Act.Gbar.L":     "0.2",
					"Layer.Inhib.Layer.On": "false",
				}},
			{Sel: ".InhibPrjn", Desc: "weaker inhib",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
		},
	}},
}

func newTestNet(ctx *Context) *Network {
	var testNet Network
	testNet.InitName(&testNet, "testNet")
	inLay := testNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), BackPrjn)

	testNet.Build(ctx)
	testNet.Defaults()
	testNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWts(ctx)                                       // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

func TestSynVals(t *testing.T) {
	ctx := NewContext()
	testNet := newTestNet(ctx)
	hidLay := testNet.AxonLayerByName("Hidden")
	p, err := hidLay.SendNameTry("Input")
	if err != nil {
		t.Error(err)
	}
	fmIn := p.(*Prjn)

	bfWt := fmIn.SynVal("Wt", 1, 1)
	if mat32.IsNaN(bfWt) {
		t.Errorf("Wt syn var not found")
	}
	bfLWt := fmIn.SynVal("LWt", 1, 1)

	fmIn.SetSynVal("Wt", 1, 1, .15)

	afWt := fmIn.SynVal("Wt", 1, 1)
	afLWt := fmIn.SynVal("LWt", 1, 1)

	cmprFloats([]float32{bfWt, bfLWt, afWt, afLWt}, []float32{0.5, 0.5, 0.15, 0.42822415}, "syn val setting test", t)
}

func newInPats() *etensor.Float32 {
	inPats := etensor.NewFloat32([]int{4, 4, 1}, nil, []string{"pat", "Y", "X"})
	for pi := 0; pi < 4; pi++ {
		inPats.Set([]int{pi, pi, 0}, 1)
	}
	return inPats
}

func cmprFloats(out, cor []float32, msg string, t *testing.T) {
	// TOLERANCE is the numerical difference tolerance for comparing vs. target values
	const TOLERANCE = float32(1.0e-6)

	t.Helper()
	hadErr := false
	for i := range out {
		if mat32.IsNaN(out[1]) {
			t.Errorf("%v err: out: %v is NaN, index: %v\n", msg, out[i], i)
		}
		dif := mat32.Abs(out[i] - cor[i])
		if dif > TOLERANCE { // allow for small numerical diffs
			hadErr = true
			t.Errorf("%v err: out: %v, cor: %v, dif: %v index: %v\n", msg, out[i], cor[i], dif, i)
		}
	}
	if hadErr {
		fmt.Printf("\t%s := []float32{", msg)
		for i := range out {
			fmt.Printf("%g", out[i])
			if i < len(out)-1 {
				fmt.Printf(", ")
			}
		}
		fmt.Printf("}\n")
	}
}

func TestSpikeProp(t *testing.T) {
	net := NewNetwork("SpikeNet")
	inLay := net.AddLayer("Input", []int{1, 1}, InputLayer)
	hidLay := net.AddLayer("Hidden", []int{1, 1}, SuperLayer)

	prj := net.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), ForwardPrjn)

	ctx := NewContext()

	net.Build(ctx)
	net.Defaults()
	net.ApplyParams(ParamSets[0].Sheets["Network"], false)

	net.InitExt(ctx)

	pat := etensor.NewFloat32([]int{1, 1}, nil, []string{"Y", "X"})
	pat.Set([]int{0, 0}, 1)

	for del := 0; del <= 4; del++ {
		prj.Params.Com.Delay = uint32(del)
		prj.Params.Com.MaxDelay = uint32(del) // now need to ensure that >= Delay
		net.InitWts(ctx)                      // resets Gbuf
		net.NewState(ctx)

		inLay.ApplyExt(ctx, 0, pat)

		net.NewState(ctx)
		ctx.NewState(etime.Train)

		inCyc := 0
		hidCyc := 0
		for cyc := 0; cyc < 100; cyc++ {
			net.Cycle(ctx)
			ctx.CycleInc()

			if NrnV(ctx, inLay.NeurStIdx, 0, Spike) > 0 {
				inCyc = cyc
			}

			ge := NrnV(ctx, hidLay.NeurStIdx, 0, Ge)
			if ge > 0 {
				hidCyc = cyc
				break
			}
		}
		if hidCyc-inCyc != del+1 {
			t.Errorf("SpikeProp error -- delay: %d  actual: %d\n", del, hidCyc-inCyc)
		}
	}
}

func TestNetAct(t *testing.T) {
	NetActTest(t, false)
}

func TestGPUAct(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	// vgpu.Debug = true
	NetActTest(t, true)
}

func NetActTest(t *testing.T, gpu bool) {
	ctx := NewContext()
	testNet := newTestNet(ctx)
	testNet.InitExt(ctx)
	inPats := newInPats()

	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
	}

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.6944439, 0, 0, 0}
	qtr0HidGes := []float32{0.35385746, 0, 0, 0}
	qtr0HidGis := []float32{0.15478331, 0.15478331, 0.15478331, 0.15478331}
	qtr0OutActs := []float32{0.5638285, 0, 0, 0}
	qtr0OutGes := []float32{0.38044316, 0, 0, 0}
	qtr0OutGis := []float32{0.19012947, 0.19012947, 0.19012947, 0.19012947}

	qtr3HidActs := []float32{0.56933826, 0, 0, 0}
	qtr3HidGes := []float32{0.43080646, 0, 0, 0}
	qtr3HidGis := []float32{0.21780373, 0.21780373, 0.21780373, 0.21780373}
	qtr3OutActs := []float32{0.69444436, 0, 0, 0}
	qtr3OutGes := []float32{0.8, 0, 0, 0}
	qtr3OutGis := []float32{0.48472303, 0.48472303, 0.48472303, 0.48472303}

	p1qtr0HidActs := []float32{1.2795964e-10, 0.47059, 0, 0}
	p1qtr0HidGes := []float32{0.011436448, 0.44748923, 0, 0}
	p1qtr0HidGis := []float32{0.19098659, 0.19098659, 0.19098659, 0.19098659}
	p1qtr0OutActs := []float32{1.5607746e-10, 0, 0, 0}
	p1qtr0OutGes := []float32{0.0136372205, 0.22609714, 0, 0}
	p1qtr0OutGis := []float32{0.089633144, 0.089633144, 0.089633144, 0.089633144}

	p1qtr3HidActs := []float32{2.837341e-39, 0.5439926, 0, 0}
	p1qtr3HidGes := []float32{0.002279978, 0.6443535, 0, 0}
	p1qtr3HidGis := []float32{0.31420222, 0.31420222, 0.31420222, 0.31420222}
	p1qtr3OutActs := []float32{3.460815e-39, 0.72627467, 0, 0}
	p1qtr3OutGes := []float32{0, 0.8, 0, 0}
	p1qtr3OutGis := []float32{0.4725598, 0.4725598, 0.4725598, 0.4725598}

	inActs := []float32{}
	hidActs := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	outActs := []float32{}
	outGes := []float32{}
	outGis := []float32{}

	cycPerQtr := 50

	for pi := 0; pi < 4; pi++ {
		inpat, err := inPats.SubSpaceTry([]int{pi})
		if err != nil {
			t.Fatal(err)
		}
		testNet.InitExt(ctx)
		inLay.ApplyExt(ctx, 0, inpat)
		outLay.ApplyExt(ctx, 0, inpat)
		testNet.ApplyExts(ctx) // key now for GPU

		testNet.NewState(ctx)
		ctx.NewState(etime.Train)

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()
				if gpu {
					testNet.GPU.SyncNeuronsFmGPU()
				}

				if printCycs {
					inLay.UnitVals(&inActs, "Act")
					hidLay.UnitVals(&hidActs, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					outLay.UnitVals(&outActs, "Act")
					outLay.UnitVals(&outGes, "Ge")
					outLay.UnitVals(&outGis, "Gi")
					fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, cyc, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
				}
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				testNet.PlusPhaseStart(ctx)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			inLay.UnitVals(&inActs, "Act")
			hidLay.UnitVals(&hidActs, "Act")
			hidLay.UnitVals(&hidGes, "Ge")
			hidLay.UnitVals(&hidGis, "Gi")
			outLay.UnitVals(&outActs, "Act")
			outLay.UnitVals(&outGes, "Ge")
			outLay.UnitVals(&outGis, "Gi")

			if printQtrs {
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctx.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				cmprFloats(hidActs, qtr0HidActs, "qtr0HidActs", t)
				cmprFloats(hidGes, qtr0HidGes, "qtr0HidGes", t)
				cmprFloats(hidGis, qtr0HidGis, "qtr0HidGis", t)
				cmprFloats(outActs, qtr0OutActs, "qtr0OutActs", t)
				cmprFloats(outGes, qtr0OutGes, "qtr0OutGes", t)
				cmprFloats(outGis, qtr0OutGis, "qtr0OutGis", t)
			}
			if pi == 0 && qtr == 3 {
				cmprFloats(hidActs, qtr3HidActs, "qtr3HidActs", t)
				cmprFloats(hidGes, qtr3HidGes, "qtr3HidGes", t)
				cmprFloats(hidGis, qtr3HidGis, "qtr3HidGis", t)
				cmprFloats(outActs, qtr3OutActs, "qtr3OutActs", t)
				cmprFloats(outGes, qtr3OutGes, "qtr3OutGes", t)
				cmprFloats(outGis, qtr3OutGis, "qtr3OutGis", t)
			}
			if pi == 1 && qtr == 0 {
				cmprFloats(hidActs, p1qtr0HidActs, "p1qtr0HidActs", t)
				cmprFloats(hidGes, p1qtr0HidGes, "p1qtr0HidGes", t)
				cmprFloats(hidGis, p1qtr0HidGis, "p1qtr0HidGis", t)
				cmprFloats(outActs, p1qtr0OutActs, "p1qtr0OutActs", t)
				cmprFloats(outGes, p1qtr0OutGes, "p1qtr0OutGes", t)
				cmprFloats(outGis, p1qtr0OutGis, "p1qtr0OutGis", t)
			}
			if pi == 1 && qtr == 3 {
				cmprFloats(hidActs, p1qtr3HidActs, "p1qtr3HidActs", t)
				cmprFloats(hidGes, p1qtr3HidGes, "p1qtr3HidGes", t)
				cmprFloats(hidGis, p1qtr3HidGis, "p1qtr3HidGis", t)
				cmprFloats(outActs, p1qtr3OutActs, "p1qtr3OutActs", t)
				cmprFloats(outGes, p1qtr3OutGes, "p1qtr3OutGes", t)
				cmprFloats(outGis, p1qtr3OutGis, "p1qtr3OutGis", t)
			}
		}
		testNet.PlusPhase(ctx)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}

	testNet.GPU.Destroy()
}

func TestNetLearn(t *testing.T) {
	NetTestLearn(t, false)
}

func TestGPULearn(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	NetTestLearn(t, true)
}

func NetTestLearn(t *testing.T, gpu bool) {
	ctx := NewContext()

	testNet := newTestNet(ctx)
	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	// allp := testNet.AllParams()
	// os.WriteFile("test_net_act_all_pars.txt", []byte(allp), 0664)

	printCycs := false
	printQtrs := false

	// these are organized by pattern within and then by test iteration (params) outer
	// only the single active synapse is represented -- one per pattern
	// if there are differences, they will multiply over patterns and layers..
	qtr3HidCaP := []float32{0.54922855, 0.54092765, 0.5374942, 0.5424088}
	qtr3HidCaD := []float32{0.5214639, 0.49507803, 0.4993692, 0.5030094}
	qtr3OutCaP := []float32{0.5834704, 0.5698648, 0.5812334, 0.5744743}
	qtr3OutCaD := []float32{0.5047723, 0.4639851, 0.48322654, 0.47419468}

	q3hidCaP := make([]float32, 4*NLrnPars)
	q3hidCaD := make([]float32, 4*NLrnPars)
	q3outCaP := make([]float32, 4*NLrnPars)
	q3outCaD := make([]float32, 4*NLrnPars)

	hidDwts := []float32{0.0015591943, 0.002412954, 0.0018998333, 0.0019943935}
	outDwts := []float32{0.003556001, 0.008800001, 0.0067477366, 0.0069709825}
	hidWts := []float32{0.5093542, 0.5144739, 0.51139706, 0.51196396}
	outWts := []float32{0.5213235, 0.5526102, 0.5404005, 0.54173136}

	hiddwt := make([]float32, 4*NLrnPars)
	outdwt := make([]float32, 4*NLrnPars)
	hidwt := make([]float32, 4*NLrnPars)
	outwt := make([]float32, 4*NLrnPars)

	hidAct := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	hidCaM := []float32{}
	hidCaP := []float32{}
	hidCaD := []float32{}
	outCaP := []float32{}
	outCaD := []float32{}

	cycPerQtr := 50

	for ti := 0; ti < NLrnPars; ti++ {
		testNet.Defaults()
		testNet.ApplyParams(ParamSets[0].Sheets["Network"], false)  // always apply base
		testNet.ApplyParams(ParamSets[ti].Sheets["Network"], false) // then specific
		testNet.InitWts(ctx)
		testNet.InitExt(ctx)

		if gpu {
			testNet.ConfigGPUnoGUI(ctx)
		}

		for pi := 0; pi < 4; pi++ {
			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt(ctx)
			inLay.ApplyExt(ctx, 0, inpat)
			outLay.ApplyExt(ctx, 0, inpat)
			testNet.ApplyExts(ctx) // key now for GPU

			ctx.NewState(etime.Train)
			testNet.NewState(ctx)
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					testNet.Cycle(ctx)
					ctx.CycleInc()
					if gpu {
						testNet.GPU.SyncNeuronsFmGPU()
					}

					hidLay.UnitVals(&hidAct, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					hidLay.UnitVals(&hidCaM, "CaM")
					hidLay.UnitVals(&hidCaP, "CaP")
					hidLay.UnitVals(&hidCaD, "CaD")

					outLay.UnitVals(&outCaP, "CaP")
					outLay.UnitVals(&outCaD, "CaD")

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAct, hidGes, hidGis, hidCaM, hidCaP, hidCaD, outCaP, outCaD)
					}
				}
				if qtr == 2 {
					testNet.MinusPhase(ctx)
					ctx.NewPhase(false)
					testNet.PlusPhaseStart(ctx)
				}

				hidLay.UnitVals(&hidCaP, "CaP")
				hidLay.UnitVals(&hidCaD, "CaD")

				outLay.UnitVals(&outCaP, "CaP")
				outLay.UnitVals(&outCaD, "CaD")

				if qtr == 3 {
					didx := ti*4 + pi
					q3hidCaD[didx] = hidCaD[pi]
					q3hidCaP[didx] = hidCaP[pi]
					q3outCaD[didx] = outCaD[pi]
					q3outCaP[didx] = outCaP[pi]
				}

				if printQtrs {
					fmt.Printf("pat: %v qtr: %v cyc: %v\nhid avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidCaP, hidCaD, outCaP, outCaD)
				}

			}
			testNet.PlusPhase(ctx)

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			testNet.DWt(ctx)
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
			}

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctx)
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
			}

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)
		}
	}

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr3HidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr3HidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr3OutCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr3OutCaD", t)

	cmprFloats(hiddwt, hidDwts, "hidDwts", t)
	cmprFloats(outdwt, outDwts, "outDwts", t)
	cmprFloats(hidwt, hidWts, "hidWts", t)
	cmprFloats(outwt, outWts, "outWts", t)

	testNet.GPU.Destroy()
}

func TestGPURLRate(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	NetTestRLRate(t, true)
}

func TestNetRLRate(t *testing.T) {
	NetTestRLRate(t, false)
}

func NetTestRLRate(t *testing.T, gpu bool) {
	ctx := NewContext()

	testNet := newTestNet(ctx)
	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	// allp := testNet.AllParams()
	// os.WriteFile("test_net_act_all_pars.txt", []byte(allp), 0664)

	printCycs := false
	printQtrs := false

	patHidRLRates := []float32{5.0000002e-05, 5.0000002e-05, 5.0000002e-05, 5.0000002e-05, 0.00019181852, 0.0030487436, 5.0000002e-05, 5.0000002e-05, 5.0000002e-05, 0.00016288209, 5.0000002e-05, 5.0000002e-05, 5.0000002e-05, 5.0000002e-05, 0.00015015423, 0.002566926}

	// these are organized by pattern within and then by test iteration (params) outer
	// only the single active synapse is represented -- one per pattern
	// if there are differences, they will multiply over patterns and layers..

	qtr3HidCaP := []float32{0.54922855, 0.54092765, 0.5374942, 0.5424088}
	qtr3HidCaD := []float32{0.5214639, 0.49507803, 0.4993692, 0.5030094}
	qtr3OutCaP := []float32{0.5834704, 0.5698648, 0.5812334, 0.5744743}
	qtr3OutCaD := []float32{0.5047723, 0.4639851, 0.48322654, 0.47419468}

	q3hidCaP := make([]float32, 4*NLrnPars)
	q3hidCaD := make([]float32, 4*NLrnPars)
	q3outCaP := make([]float32, 4*NLrnPars)
	q3outCaD := make([]float32, 4*NLrnPars)

	hidDwts := []float32{7.795972e-08, 7.3564784e-06, 9.499167e-08, 5.11946e-06}
	outDwts := []float32{0.003556001, 0.008800001, 0.0067477366, 0.0069709825}
	hidWts := []float32{0.50000036, 0.500044, 0.5000007, 0.50003076}
	outWts := []float32{0.5213235, 0.5526102, 0.5404005, 0.54173136}

	hiddwt := make([]float32, 4*NLrnPars)
	outdwt := make([]float32, 4*NLrnPars)
	hidwt := make([]float32, 4*NLrnPars)
	outwt := make([]float32, 4*NLrnPars)
	hidrlrs := make([]float32, 4*4*NLrnPars) // 4 units, 4 pats

	hidAct := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	hidCaM := []float32{}
	hidCaP := []float32{}
	hidCaD := []float32{}
	hidRLRate := []float32{}
	outCaP := []float32{}
	outCaD := []float32{}

	cycPerQtr := 50

	for ti := 0; ti < NLrnPars; ti++ {
		testNet.Defaults()
		testNet.ApplyParams(ParamSets[0].Sheets["Network"], false)  // always apply base
		testNet.ApplyParams(ParamSets[ti].Sheets["Network"], false) // then specific
		hidLay.Params.Learn.RLRate.On.SetBool(true)
		testNet.InitWts(ctx)
		testNet.InitExt(ctx)

		for pi := 0; pi < 4; pi++ {
			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt(ctx)
			inLay.ApplyExt(ctx, 0, inpat)
			outLay.ApplyExt(ctx, 0, inpat)
			testNet.ApplyExts(ctx) // key now for GPU

			ctx.NewState(etime.Train)
			testNet.NewState(ctx)
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					testNet.Cycle(ctx)
					ctx.CycleInc()
					testNet.GPU.SyncNeuronsFmGPU()

					hidLay.UnitVals(&hidAct, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					hidLay.UnitVals(&hidCaM, "CaM")
					hidLay.UnitVals(&hidCaP, "CaP")
					hidLay.UnitVals(&hidCaD, "CaD")

					outLay.UnitVals(&outCaP, "CaP")
					outLay.UnitVals(&outCaD, "CaD")

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAct, hidGes, hidGis, hidCaM, hidCaP, hidCaD, outCaP, outCaD)
					}
				}
				if qtr == 2 {
					testNet.MinusPhase(ctx)
					ctx.NewPhase(false)
					testNet.PlusPhaseStart(ctx)
				}

				hidLay.UnitVals(&hidCaP, "CaP")
				hidLay.UnitVals(&hidCaD, "CaD")

				outLay.UnitVals(&outCaP, "CaP")
				outLay.UnitVals(&outCaD, "CaD")

				if qtr == 3 {
					didx := ti*4 + pi
					q3hidCaD[didx] = hidCaD[pi]
					q3hidCaP[didx] = hidCaP[pi]
					q3outCaD[didx] = outCaD[pi]
					q3outCaP[didx] = outCaP[pi]
				}

				if printQtrs {
					fmt.Printf("pat: %v qtr: %v cyc: %v\nhid avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidCaP, hidCaD, outCaP, outCaD)
				}
			}
			testNet.PlusPhase(ctx)
			if gpu {
				testNet.GPU.SyncNeuronsFmGPU() // RLRate updated after plus
			}

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			hidLay.UnitVals(&hidRLRate, "RLRate")
			ridx := ti*4*4 + pi*4
			copy(hidrlrs[ridx:ridx+4], hidRLRate)

			testNet.DWt(ctx)
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
			}

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctx)
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
			}

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)
		}
	}

	cmprFloats(hidrlrs, patHidRLRates, "patHidRLRates", t)

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr3HidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr3HidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr3OutCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr3OutCaD", t)

	cmprFloats(hiddwt, hidDwts, "hidDwts", t)
	cmprFloats(outdwt, outDwts, "outDwts", t)
	cmprFloats(hidwt, hidWts, "hidWts", t)
	cmprFloats(outwt, outWts, "outWts", t)

	testNet.GPU.Destroy()
}

func TestInhibAct(t *testing.T) {
	// t.Skip("Skipping TestInhibAct for now until stable")
	inPats := newInPats()
	var InhibNet Network
	InhibNet.InitName(&InhibNet, "InhibNet")

	inLay := InhibNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := InhibNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := InhibNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	one2one := prjn.NewOneToOne()

	InhibNet.ConnectLayers(inLay, hidLay, one2one, ForwardPrjn)
	InhibNet.ConnectLayers(inLay, hidLay, one2one, InhibPrjn)
	InhibNet.ConnectLayers(hidLay, outLay, one2one, ForwardPrjn)
	InhibNet.ConnectLayers(outLay, hidLay, one2one, BackPrjn)

	ctx := NewContext()

	InhibNet.Build(ctx)
	InhibNet.Defaults()
	InhibNet.ApplyParams(ParamSets[0].Sheets["Network"], false)
	InhibNet.ApplyParams(ParamSets[0].Sheets["InhibOff"], false)
	InhibNet.InitWts(ctx) // get GScale
	InhibNet.NewState(ctx)

	InhibNet.InitWts(ctx)
	InhibNet.InitExt(ctx)

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.8761159, 0, 0, 0}
	qtr0HidGes := []float32{0.90975666, 0, 0, 0}
	qtr0HidGis := []float32{0.0930098, 0, 0, 0}
	qtr0OutActs := []float32{0.793471, 0, 0, 0}
	qtr0OutGes := []float32{0.74590594, 0, 0, 0}
	qtr0OutGis := []float32{0, 0, 0, 0}

	qtr3HidActs := []float32{0.9202804, 0, 0, 0}
	qtr3HidGes := []float32{1.1153994, 0, 0, 0}
	qtr3HidGis := []float32{0.09305161, 0, 0, 0}
	qtr3OutActs := []float32{0.92592585, 0, 0, 0}
	qtr3OutGes := []float32{0.8, 0, 0, 0}
	qtr3OutGis := []float32{0, 0, 0, 0}

	inActs := []float32{}
	hidActs := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	outActs := []float32{}
	outGes := []float32{}
	outGis := []float32{}

	cycPerQtr := 50

	for pi := 0; pi < 4; pi++ {
		inpat, err := inPats.SubSpaceTry([]int{pi})
		if err != nil {
			t.Error(err)
		}
		inLay.ApplyExt(ctx, 0, inpat)
		outLay.ApplyExt(ctx, 0, inpat)

		InhibNet.NewState(ctx)
		ctx.NewState(etime.Train)
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				InhibNet.Cycle(ctx)
				ctx.CycleInc()

				if printCycs {
					inLay.UnitVals(&inActs, "Act")
					hidLay.UnitVals(&hidActs, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					outLay.UnitVals(&outActs, "Act")
					outLay.UnitVals(&outGes, "Ge")
					outLay.UnitVals(&outGis, "Gi")
					fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, cyc, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
				}
			}
			if qtr == 2 {
				InhibNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				InhibNet.PlusPhaseStart(ctx)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			inLay.UnitVals(&inActs, "Act")
			hidLay.UnitVals(&hidActs, "Act")
			hidLay.UnitVals(&hidGes, "Ge")
			hidLay.UnitVals(&hidGis, "Gi")
			outLay.UnitVals(&outActs, "Act")
			outLay.UnitVals(&outGes, "Ge")
			outLay.UnitVals(&outGis, "Gi")

			if printQtrs {
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctx.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				cmprFloats(hidActs, qtr0HidActs, "qtr0HidActs", t)
				cmprFloats(hidGes, qtr0HidGes, "qtr0HidGes", t)
				cmprFloats(hidGis, qtr0HidGis, "qtr0HidGis", t)
				cmprFloats(outActs, qtr0OutActs, "qtr0OutActs", t)
				cmprFloats(outGes, qtr0OutGes, "qtr0OutGes", t)
				cmprFloats(outGis, qtr0OutGis, "qtr0OutGis", t)
			}
			if pi == 0 && qtr == 3 {
				cmprFloats(hidActs, qtr3HidActs, "qtr3HidActs", t)
				cmprFloats(hidGes, qtr3HidGes, "qtr3HidGes", t)
				cmprFloats(hidGis, qtr3HidGis, "qtr3HidGis", t)
				cmprFloats(outActs, qtr3OutActs, "qtr3OutActs", t)
				cmprFloats(outGes, qtr3OutGes, "qtr3OutGes", t)
				cmprFloats(outGis, qtr3OutGis, "qtr3OutGis", t)
			}
		}
		InhibNet.PlusPhase(ctx)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}

func saveToFile(net *Network, t *testing.T) {
	var buf bytes.Buffer
	net.WriteWtsJSON(&buf)
	wb := buf.Bytes()
	fmt.Printf("testNet Trained Weights:\n\n%v\n", string(wb))

	fp, err := os.Create("testdata/testnet_train.wts")
	defer fp.Close()
	if err != nil {
		t.Error(err)
	}
	fp.Write(wb)
}

/* todo: fixme
func TestSWtInit(t *testing.T) {
	pj := &PrjnParams{}
	pj.Defaults()

	nsamp := 100
	sch := etable.Schema{
		{"Wt", etensor.FLOAT32, nil, nil},
		{"LWt", etensor.FLOAT32, nil, nil},
		{"SWt", etensor.FLOAT32, nil, nil},
	}
	dt := &etable.Table{}
	dt.SetFromSchema(sch, nsamp)

	/////////////////////////////////////////////
	mean := float32(0.5)
	vr := float32(0.25)
	spct := float32(0.5)
	pj.SWt.Init.Var = vr

	nt := NewNetwork("test")
	nt.SetRndSeed(1)

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWt.InitWtsSyn(ctx, &nt.Rand, sy, mean, spct)
		dt.SetCellFloat("Wt", i, float64(sy.Wt))
		dt.SetCellFloat("LWt", i, float64(sy.LWt))
		dt.SetCellFloat("SWt", i, float64(sy.SWt))
	}
	ix := etable.NewIdxView(dt)
	desc := agg.DescAll(ix)

	meanRow := desc.RowsByString("Agg", "Mean", etable.Equals, etable.UseCase)[0]
	minRow := desc.RowsByString("Agg", "Min", etable.Equals, etable.UseCase)[0]
	maxRow := desc.RowsByString("Agg", "Max", etable.Equals, etable.UseCase)[0]
	semRow := desc.RowsByString("Agg", "Sem", etable.Equals, etable.UseCase)[0]

	if desc.CellFloat("Wt", minRow) > 0.3 || desc.CellFloat("Wt", maxRow) < 0.7 {
		t.Errorf("SPct: %g\t Wt Min and Max should be < 0.3, > 0.7 not: %g, %g\n", spct, desc.CellFloat("Wt", minRow), desc.CellFloat("Wt", maxRow))
	}
	if desc.CellFloat("Wt", meanRow) < 0.45 || desc.CellFloat("Wt", meanRow) > 0.55 {
		t.Errorf("SPct: %g\t Wt Mean should be > 0.45, < 0.55 not: %g\n", spct, desc.CellFloat("Wt", meanRow))
	}
	if desc.CellFloat("Wt", semRow) < 0.01 || desc.CellFloat("Wt", semRow) > 0.02 {
		t.Errorf("SPct: %g\t Wt SEM should be > 0.01, < 0.02 not: %g\n", spct, desc.CellFloat("Wt", semRow))
	}

	// b := bytes.NewBuffer(nil)
	// desc.WriteCSV(b, etable.Tab, etable.Headers)
	// fmt.Printf("%s\n", string(b.Bytes()))

	/////////////////////////////////////////////
	mean = float32(0.5)
	vr = float32(0.25)
	spct = float32(1.0)
	pj.SWt.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
		dt.SetCellFloat("Wt", i, float64(sy.Wt))
		dt.SetCellFloat("LWt", i, float64(sy.LWt))
		dt.SetCellFloat("SWt", i, float64(sy.SWt))
	}
	desc = agg.DescAll(ix)
	if desc.CellFloat("Wt", minRow) > 0.3 || desc.CellFloat("Wt", maxRow) < 0.7 {
		t.Errorf("SPct: %g\t Wt Min and Max should be < 0.3, > 0.7 not: %g, %g\n", spct, desc.CellFloat("Wt", minRow), desc.CellFloat("Wt", maxRow))
	}
	if desc.CellFloat("Wt", meanRow) < 0.45 || desc.CellFloat("Wt", meanRow) > 0.55 {
		t.Errorf("SPct: %g\t Wt Mean should be > 0.45, < 0.55 not: %g\n", spct, desc.CellFloat("Wt", meanRow))
	}
	if desc.CellFloat("Wt", semRow) < 0.01 || desc.CellFloat("Wt", semRow) > 0.02 {
		t.Errorf("SPct: %g\t Wt SEM should be > 0.01, < 0.02 not: %g\n", spct, desc.CellFloat("Wt", semRow))
	}
	if desc.CellFloat("LWt", minRow) != 0.5 || desc.CellFloat("LWt", maxRow) != 0.5 {
		t.Errorf("SPct: %g\t LWt Min and Max should both be 0.5, not: %g, %g\n", spct, desc.CellFloat("LWt", minRow), desc.CellFloat("LWt", maxRow))
	}
	// b.Reset()
	// desc.WriteCSV(b, etable.Tab, etable.Headers)
	// fmt.Printf("%s\n", string(b.Bytes()))

	/////////////////////////////////////////////
	mean = float32(0.5)
	vr = float32(0.25)
	spct = float32(0.0)
	pj.SWt.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
		dt.SetCellFloat("Wt", i, float64(sy.Wt))
		dt.SetCellFloat("LWt", i, float64(sy.LWt))
		dt.SetCellFloat("SWt", i, float64(sy.SWt))
	}
	desc = agg.DescAll(ix)
	if desc.CellFloat("Wt", minRow) > 0.3 || desc.CellFloat("Wt", maxRow) < 0.7 {
		t.Errorf("SPct: %g\t Wt Min and Max should be < 0.3, > 0.7 not: %g, %g\n", spct, desc.CellFloat("Wt", minRow), desc.CellFloat("Wt", maxRow))
	}
	if desc.CellFloat("Wt", meanRow) < 0.45 || desc.CellFloat("Wt", meanRow) > 0.55 {
		t.Errorf("SPct: %g\t Wt Mean should be > 0.45, < 0.55 not: %g\n", spct, desc.CellFloat("Wt", meanRow))
	}
	if desc.CellFloat("Wt", semRow) < 0.01 || desc.CellFloat("Wt", semRow) > 0.02 {
		t.Errorf("SPct: %g\t Wt SEM should be > 0.01, < 0.02 not: %g\n", spct, desc.CellFloat("Wt", semRow))
	}
	if desc.CellFloat("SWt", minRow) != 0.5 || desc.CellFloat("SWt", maxRow) != 0.5 {
		t.Errorf("SPct: %g\t SWt Min and Max should both be 0.5, not: %g, %g\n", spct, desc.CellFloat("LWt", minRow), desc.CellFloat("LWt", maxRow))
	}
	// b.Reset()
	// desc.WriteCSV(b, etable.Tab, etable.Headers)
	// fmt.Printf("%s\n", string(b.Bytes()))

	/////////////////////////////////////////////
	mean = float32(0.1)
	vr = float32(0.05)
	spct = float32(0.0)
	pj.SWt.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
		dt.SetCellFloat("Wt", i, float64(sy.Wt))
		dt.SetCellFloat("LWt", i, float64(sy.LWt))
		dt.SetCellFloat("SWt", i, float64(sy.SWt))
	}
	desc = agg.DescAll(ix)
	if desc.CellFloat("Wt", minRow) > 0.08 || desc.CellFloat("Wt", maxRow) < 0.12 {
		t.Errorf("SPct: %g\t Wt Min and Max should be < 0.08, > 0.12 not: %g, %g\n", spct, desc.CellFloat("Wt", minRow), desc.CellFloat("Wt", maxRow))
	}
	if desc.CellFloat("Wt", meanRow) < 0.08 || desc.CellFloat("Wt", meanRow) > 0.12 {
		t.Errorf("SPct: %g\t Wt Mean should be > 0.08, < 0.12 not: %g\n", spct, desc.CellFloat("Wt", meanRow))
	}
	if desc.CellFloat("SWt", minRow) != 0.5 || desc.CellFloat("SWt", maxRow) != 0.5 {
		t.Errorf("SPct: %g\t SWt Min and Max should both be 0.5, not: %g, %g\n", spct, desc.CellFloat("LWt", minRow), desc.CellFloat("LWt", maxRow))
	}
	// b.Reset()
	// desc.WriteCSV(b, etable.Tab, etable.Headers)
	// fmt.Printf("%s\n", string(b.Bytes()))

	/////////////////////////////////////////////
	mean = float32(0.8)
	vr = float32(0.05)
	spct = float32(0.5)
	pj.SWt.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
		dt.SetCellFloat("Wt", i, float64(sy.Wt))
		dt.SetCellFloat("LWt", i, float64(sy.LWt))
		dt.SetCellFloat("SWt", i, float64(sy.SWt))
	}
	desc = agg.DescAll(ix)
	if desc.CellFloat("Wt", minRow) > 0.76 || desc.CellFloat("Wt", maxRow) < 0.84 {
		t.Errorf("SPct: %g\t Wt Min and Max should be < 0.66, > 0.74 not: %g, %g\n", spct, desc.CellFloat("Wt", minRow), desc.CellFloat("Wt", maxRow))
	}
	if desc.CellFloat("Wt", meanRow) < 0.79 || desc.CellFloat("Wt", meanRow) > 0.81 {
		t.Errorf("SPct: %g\t Wt Mean should be > 0.65, < 0.75 not: %g\n", spct, desc.CellFloat("Wt", meanRow))
	}
	if desc.CellFloat("SWt", minRow) < 0.76 || desc.CellFloat("SWt", maxRow) > 0.83 {
		t.Errorf("SPct: %g\t SWt Min and Max should be < 0.76, > 0.83, not: %g, %g\n", spct, desc.CellFloat("SWt", minRow), desc.CellFloat("SWt", maxRow))
	}
	// b.Reset()
	// desc.WriteCSV(b, etable.Tab, etable.Headers)
	// fmt.Printf("%s\n", string(b.Bytes()))
}

func TestSWtLinLearn(t *testing.T) {
	pj := &PrjnParams{}
	pj.Defaults()
	sy := &Synapse{}

	nt := NewNetwork("test")
	nt.SetRndSeed(1)

	/////////////////////////////////////////////
	mean := float32(0.1)
	vr := float32(0.05)
	spct := float32(0.0)
	dwt := float32(0.1)
	pj.SWt.Init.Var = vr
	pj.SWt.Adapt.SigGain = 1
	nlrn := 10
	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)

	pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
	// fmt.Printf("Wt: %g\t LWt: %g\t SWt: %g\n", sy.Wt, sy.LWt, sy.SWt)
	for i := 0; i < nlrn; i++ {
		sy.DWt = dwt
		pj.SWt.WtFmDWt(&sy.DWt, &sy.Wt, &sy.LWt, sy.SWt)
		// fmt.Printf("Wt: %g\t LWt: %g\t SWt: %g\n", sy.Wt, sy.LWt, sy.SWt)
	}
	if sy.Wt != 1 {
		t.Errorf("SPct: %g\t Wt should be 1 not: %g\n", spct, sy.Wt)
	}
	if sy.LWt != 1 {
		t.Errorf("SPct: %g\t LWt should be 1 not: %g\n", spct, sy.LWt)
	}
	if sy.SWt != 0.5 {
		t.Errorf("SPct: %g\t SWt should be 0.5 not: %g\n", spct, sy.SWt)
	}
}

*/
