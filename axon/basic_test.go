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
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
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

func newTestNet() *Network {
	var testNet Network
	testNet.InitName(&testNet, "testNet")
	inLay := testNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), BackPrjn)

	ctx := NewContext()

	testNet.Build()
	testNet.Defaults()
	testNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWts()                                          // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

func TestSynVals(t *testing.T) {
	testNet := newTestNet()
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
	for i := range out {
		if mat32.IsNaN(out[1]) {
			t.Errorf("%v err: out: %v is NaN, index: %v\n", msg, out[i], i)
		}
		dif := mat32.Abs(out[i] - cor[i])
		if dif > TOLERANCE { // allow for small numerical diffs
			t.Errorf("%v err: out: %v, cor: %v, dif: %v index: %v\n", msg, out[i], cor[i], dif, i)
			t.Errorf("%v out: %v, cor: %v", msg, out, cor)
		}
	}
}

func TestSpikeProp(t *testing.T) {
	net := NewNetwork("SpikeNet")
	inLay := net.AddLayer("Input", []int{1, 1}, InputLayer)
	hidLay := net.AddLayer("Hidden", []int{1, 1}, SuperLayer)

	prj := net.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), ForwardPrjn)

	net.Build()
	net.Defaults()
	net.ApplyParams(ParamSets[0].Sheets["Network"], false)

	net.InitExt()

	ctx := NewContext()

	pat := etensor.NewFloat32([]int{1, 1}, nil, []string{"Y", "X"})
	pat.Set([]int{0, 0}, 1)

	for del := 0; del <= 4; del++ {
		prj.Params.Com.Delay = uint32(del)
		prj.Params.Com.MaxDelay = uint32(del) // now need to ensure that >= Delay
		net.InitWts()                         // resets Gbuf
		net.NewState(ctx)

		inLay.ApplyExt(pat)

		net.NewState(ctx)
		ctx.NewState(etime.Train)

		inCyc := 0
		hidCyc := 0
		for cyc := 0; cyc < 100; cyc++ {
			net.Cycle(ctx)
			ctx.CycleInc()

			if inLay.Neurons[0].Spike > 0 {
				inCyc = cyc
			}

			ge := hidLay.Neurons[0].Ge
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
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	// vgpu.Debug = true
	NetActTest(t, true)
}

func NetActTest(t *testing.T, gpu bool) {
	testNet := newTestNet()
	testNet.InitExt()
	inPats := newInPats()

	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	ctx := NewContext()

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
	}

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.6944439, 0, 0, 0}
	qtr0HidGes := []float32{0.31093338, 0, 0, 0}
	qtr0HidGis := []float32{0.1547833, 0.1547833, 0.1547833, 0.1547833}
	qtr0OutActs := []float32{0.55552065, 0, 0, 0}
	qtr0OutGes := []float32{0.3789059, 0, 0, 0}
	qtr0OutGis := []float32{0.20974194, 0.20974194, 0.20974194, 0.20974194}

	qtr3HidActs := []float32{0.53240955, 0, 0, 0}
	qtr3HidGes := []float32{0.36792937, 0, 0, 0}
	qtr3HidGis := []float32{0.21772972, 0.21772972, 0.21772972, 0.21772972}
	qtr3OutActs := []float32{0.7293362, 0, 0, 0}
	qtr3OutGes := []float32{0.8, 0, 0, 0}
	qtr3OutGis := []float32{0.42606226, 0.42606226, 0.42606226, 0.4260622}

	p1qtr0HidActs := []float32{1.1965988e-10, 0.50503737, 0, 0}
	p1qtr0HidGes := []float32{0.0045430386, 0.5417977, 0, 0}
	p1qtr0HidGis := []float32{0.3178829, 0.3178829, 0.3178829, 0.3178829}
	p1qtr0OutActs := []float32{1.6391945e-10, 0.3799649, 0, 0}
	p1qtr0OutGes := []float32{0.0037496479, 0.113552466, 0, 0}
	p1qtr0OutGis := []float32{0.1000951, 0.1000951, 0.1000951, 0.1000951}

	p1qtr3HidActs := []float32{2.653303e-39, 0.5264462, 0, 0}
	p1qtr3HidGes := []float32{0.0008803774, 0.50142866, 0, 0}
	p1qtr3HidGis := []float32{0.3444711, 0.3444711, 0.3444711, 0.3444711}
	p1qtr3OutActs := []float32{3.6347e-39, 0.7274741, 0, 0}
	p1qtr3OutGes := []float32{0, 0.8, 0, 0}
	p1qtr3OutGis := []float32{0.473608, 0.473608, 0.473608, 0.473608}

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
		testNet.InitExt()
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)
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
				cmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				cmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				cmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				cmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				cmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				cmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				cmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				cmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				cmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				cmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				cmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				cmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
			}
			if pi == 1 && qtr == 0 {
				cmprFloats(hidActs, p1qtr0HidActs, "p1 qtr 0 hidActs", t)
				cmprFloats(hidGes, p1qtr0HidGes, "p1 qtr 0 hidGes", t)
				cmprFloats(hidGis, p1qtr0HidGis, "p1 qtr 0 hidGis", t)
				cmprFloats(outActs, p1qtr0OutActs, "p1 qtr 0 outActs", t)
				cmprFloats(outGes, p1qtr0OutGes, "p1 qtr 0 outGes", t)
				cmprFloats(outGis, p1qtr0OutGis, "p1 qtr 0 outGis", t)
			}
			if pi == 1 && qtr == 3 {
				cmprFloats(hidActs, p1qtr3HidActs, "p1 qtr 3 hidActs", t)
				cmprFloats(hidGes, p1qtr3HidGes, "p1 qtr 3 hidGes", t)
				cmprFloats(hidGis, p1qtr3HidGis, "p1 qtr 3 hidGis", t)
				cmprFloats(outActs, p1qtr3OutActs, "p1 qtr 3 outActs", t)
				cmprFloats(outGes, p1qtr3OutGes, "p1 qtr 3 outGes", t)
				cmprFloats(outGis, p1qtr3OutGis, "p1 qtr 3 outGis", t)
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
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	NetTestLearn(t, true)
}

func NetTestLearn(t *testing.T, gpu bool) {
	testNet := newTestNet()
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
	qtr3HidCaP := []float32{0.48164067, 0.47525364, 0.4646372, 0.4758791}
	qtr3HidCaD := []float32{0.45075783, 0.4384308, 0.43476036, 0.43747732}
	qtr3OutCaP := []float32{0.5401089, 0.54058266, 0.5389495, 0.54185855}
	qtr3OutCaD := []float32{0.45136902, 0.43564644, 0.4355862, 0.4358528}

	q3hidCaP := make([]float32, 4*NLrnPars)
	q3hidCaD := make([]float32, 4*NLrnPars)
	q3outCaP := make([]float32, 4*NLrnPars)
	q3outCaD := make([]float32, 4*NLrnPars)

	hidDwts := []float32{0.0015201505, 0.0017171452, 0.00143093, 0.0017795337}
	outDwts := []float32{0.0067295646, 0.009379843, 0.007949657, 0.009496575}
	hidWts := []float32{0.50912, 0.51030153, 0.50858474, 0.5106757}
	outWts := []float32{0.5402921, 0.5560492, 0.5475578, 0.55674076}

	if gpu {
		hidDwts = []float32{0.0015176951, 0.0017110581, 0.0014290133, 0.0017731723}
		outDwts = []float32{0.0065535065, 0.008962215, 0.007896948, 0.009082118}
		hidWts = []float32{0.5091053, 0.510265, 0.50857323, 0.51063746}
		outWts = []float32{0.53924257, 0.5535727, 0.5472443, 0.554284}
	}

	/* these are for closed-form exp SynCa, CPU
	hidDwts := []float32{0.0019419516, 0.0021907063, 0.001831886, 0.0022705847}
	outDwts := []float32{0.008427641, 0.010844037, 0.009319315, 0.011078155}
	hidWts := []float32{0.51164985, 0.51314133, 0.5109896, 0.5136202}
	outWts := []float32{0.55039877, 0.56470954, 0.55569035, 0.56609094}

	// these are for closed-form exp SynCa, GPU
	hidDwts := []float32{0.0019396556, 0.0021842457, 0.0018292944, 0.0022638545}
	outDwts := []float32{0.008366027, 0.010467653, 0.009353173, 0.010706494}
	hidWts := []float32{0.5116358, 0.5131027, 0.5109739, 0.51357985}
	outWts := []float32{0.55003285, 0.56248677, 0.555891, 0.5638975}
	*/

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
		testNet.InitWts()
		testNet.InitExt()

		ctx := NewContext()

		if gpu {
			testNet.ConfigGPUnoGUI(ctx)
		}

		for pi := 0; pi < 4; pi++ {
			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt()
			inLay.ApplyExt(inpat)
			outLay.ApplyExt(inpat)
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

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr 3 hidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr 3 hidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr 3 outCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr 3 outCaD", t)

	cmprFloats(hiddwt, hidDwts, "hid DWt", t)
	cmprFloats(outdwt, outDwts, "out DWt", t)
	cmprFloats(hidwt, hidWts, "hid Wt", t)
	cmprFloats(outwt, outWts, "out Wt", t)

	testNet.GPU.Destroy()
}

func TestGPURLRate(t *testing.T) {
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	NetTestRLRate(t, true)
}

func TestNetRLRate(t *testing.T) {
	NetTestRLRate(t, false)
}

func NetTestRLRate(t *testing.T, gpu bool) {
	testNet := newTestNet()
	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	// allp := testNet.AllParams()
	// os.WriteFile("test_net_act_all_pars.txt", []byte(allp), 0664)

	printCycs := false
	printQtrs := false

	patHidRLRates := []float32{0.0019666697, 5.0000002e-05, 5.0000002e-05, 5.0000002e-05,
		0.00016045463, 0.0039996677, 5.0000002e-05, 5.0000002e-05,
		5.0000002e-05, 0.00017801004, 0.0018803712, 5.0000002e-05,
		5.0000002e-05, 5.0000002e-05, 0.00018263639, 0.0041628983}

	// these are organized by pattern within and then by test iteration (params) outer
	// only the single active synapse is represented -- one per pattern
	// if there are differences, they will multiply over patterns and layers..

	qtr3HidCaP := []float32{0.48164067, 0.47525364, 0.4646372, 0.4758791}
	qtr3HidCaD := []float32{0.45075783, 0.4384308, 0.43476036, 0.43747732}
	qtr3OutCaP := []float32{0.5401089, 0.54058266, 0.5389495, 0.54185855}
	qtr3OutCaD := []float32{0.45136902, 0.43564644, 0.4355862, 0.4358528}

	q3hidCaP := make([]float32, 4*NLrnPars)
	q3hidCaD := make([]float32, 4*NLrnPars)
	q3outCaP := make([]float32, 4*NLrnPars)
	q3outCaD := make([]float32, 4*NLrnPars)

	hidDwts := []float32{2.989634e-06, 6.8680106e-06, 2.6906796e-06, 7.4080176e-06}
	outDwts := []float32{0.0067295646, 0.009379843, 0.007949657, 0.009496575}
	hidWts := []float32{0.5000179, 0.5000411, 0.5000161, 0.50004435}
	outWts := []float32{0.5402921, 0.5560492, 0.5475578, 0.55674076}

	/* these are for closed-form exp SynCa, CPU:
	hidDwts := []float32{3.8191774e-06, 8.762097e-06, 3.4446257e-06, 9.452213e-06}
	outDwts := []float32{0.008427641, 0.010844037, 0.009319315, 0.011078155}
	hidWts := []float32{0.5000229, 0.5000526, 0.50002074, 0.50005686}
	outWts := []float32{0.55039877, 0.56470954, 0.55569035, 0.56609094}

	// closed-form exp SynCa GPU:
	hidDwts := []float32{3.8191774e-06, 8.762097e-06, 3.4446257e-06, 9.452213e-06}
	outDwts := []float32{0.008366027, 0.010467653, 0.009353173, 0.010706494}
	hidWts := []float32{0.5000229, 0.5000526, 0.50002074, 0.50005686}
	outWts := []float32{0.55003285, 0.56248677, 0.555891, 0.5638975}
	*/

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
		testNet.InitWts()
		testNet.InitExt()

		ctx := NewContext()

		for pi := 0; pi < 4; pi++ {
			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt()
			inLay.ApplyExt(inpat)
			outLay.ApplyExt(inpat)
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

	cmprFloats(hidrlrs, patHidRLRates, "hid RLRate", t)

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr 3 hidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr 3 hidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr 3 outCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr 3 outCaD", t)

	cmprFloats(hiddwt, hidDwts, "hid DWt", t)
	cmprFloats(outdwt, outDwts, "out DWt", t)
	cmprFloats(hidwt, hidWts, "hid Wt", t)
	cmprFloats(outwt, outWts, "out Wt", t)

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

	InhibNet.Build()
	InhibNet.Defaults()
	InhibNet.ApplyParams(ParamSets[0].Sheets["Network"], false)
	InhibNet.ApplyParams(ParamSets[0].Sheets["InhibOff"], false)
	InhibNet.InitWts() // get GScale
	InhibNet.NewState(ctx)

	InhibNet.InitWts()
	InhibNet.InitExt()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.8761159, 0, 0, 0}
	qtr0HidGes := []float32{0.91799927, 0, 0, 0}
	qtr0HidGis := []float32{0.09300988, 0, 0, 0}
	qtr0OutActs := []float32{0.793471, 0, 0, 0}
	qtr0OutGes := []float32{0.81241286, 0, 0, 0}
	qtr0OutGis := []float32{0, 0, 0, 0}

	qtr3HidActs := []float32{0.91901356, 0, 0, 0}
	qtr3HidGes := []float32{1.1383185, 0, 0, 0}
	qtr3HidGis := []float32{0.09305171, 0, 0, 0}
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
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)

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
				cmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				cmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				cmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				cmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				cmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				cmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				cmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				cmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				cmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				cmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				cmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				cmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
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

func TestSWtInit(t *testing.T) {
	pj := &PrjnParams{}
	pj.Defaults()
	sy := &Synapse{}

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
		pj.SWt.InitWtsSyn(&nt.Rand, sy, mean, spct)
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
