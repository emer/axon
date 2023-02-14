// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/goki/mat32"
)

func init() {
	// must lock main thread for gpu!
	runtime.LockOSThread()
}

func TestGPUAct(t *testing.T) {
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	testNet := newTestNet()
	testNet.InitExt()
	inPats := newInPats()

	inLay := testNet.LayerByName("Input").(*Layer)
	hidLay := testNet.LayerByName("Hidden").(*Layer)
	outLay := testNet.LayerByName("Output").(*Layer)

	ctx := NewContext()

	// vgpu.Debug = true
	testNet.ConfigGPUnoGUI(ctx)

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

	inExts := []float32{}
	inGes := []float32{}
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
				testNet.GPU.SyncNeuronsFmGPU()

				if printCycs {
					inLay.UnitVals(&inExts, "Ext")
					inLay.UnitVals(&inGes, "Ge")
					inLay.UnitVals(&inActs, "Spike")
					hidLay.UnitVals(&hidActs, "Spike")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					outLay.UnitVals(&outActs, "Act")
					outLay.UnitVals(&outGes, "Ge")
					outLay.UnitVals(&outGis, "Gi")
					fmt.Printf("pat: %v qtr: %v cyc: %v\nin exts: %v\nin ges: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, cyc, inExts, inGes, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
				}
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
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

// tolerance version -- learning diffs across GPU can end up being significant
func cmprFloatsTol(out, cor []float32, msg string, t *testing.T, tol float32) {
	t.Helper()
	for i := range out {
		if mat32.IsNaN(out[1]) {
			t.Errorf("%v err: out: %v is NaN, index: %v\n", msg, out[i], i)
		}
		dif := mat32.Abs(out[i] - cor[i])
		if dif > tol { // allow for small numerical diffs
			t.Errorf("%v err: out: %v, cor: %v, dif: %v index: %v\n", msg, out[i], cor[i], dif, i)
			t.Errorf("%v out: %v, cor: %v", msg, out, cor)
		}
	}
}

func TestGPULearn(t *testing.T) {
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	testNet := newTestNet()
	inPats := newInPats()
	inLay := testNet.LayerByName("Input").(*Layer)
	hidLay := testNet.LayerByName("Hidden").(*Layer)
	outLay := testNet.LayerByName("Output").(*Layer)

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

	hidDwts := []float32{0.0019396556, 0.0021842457, 0.0018292944, 0.0022638545}
	// 0.0019419516, 0.0021907063, 0.001831886, 0.0022705847
	outDwts := []float32{0.008366027, 0.010467653, 0.009353173, 0.010706494}
	// 0.008427641, 0.010844037, 0.009319315, 0.011078155}
	hidWts := []float32{0.5116358, 0.5131027, 0.5109739, 0.51357985}
	// 0.51164985, 0.51314133, 0.5109896, 0.5136202}
	outWts := []float32{0.55003285, 0.56248677, 0.555891, 0.5638975}
	// 0.55039877, 0.56470954, 0.55569035, 0.56609094}

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

		testNet.ConfigGPUnoGUI(ctx)

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

			didx := ti*4 + pi

			testNet.DWt(ctx)
			testNet.GPU.SyncSynapsesFmGPU()

			// note: gotta grab dwt while they exist
			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctx)
			testNet.GPU.SyncSynapsesFmGPU()

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)

		}
	}

	tol := float32(1.0e-6) // these can end up diverging significantly over time..
	// the synaptic ca integration etc.

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr 3 hidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr 3 hidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr 3 outCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr 3 outCaD", t)

	cmprFloatsTol(hiddwt, hidDwts, "hid DWt", t, tol)
	cmprFloatsTol(outdwt, outDwts, "out DWt", t, tol)
	cmprFloatsTol(hidwt, hidWts, "hid Wt", t, tol)
	cmprFloatsTol(outwt, outWts, "out Wt", t, tol)

	testNet.GPU.Destroy()
}

func TestGPURLRate(t *testing.T) {
	if os.Getenv("TEST_GPU") == "" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	testNet := newTestNet()
	inPats := newInPats()
	inLay := testNet.LayerByName("Input").(*Layer)
	hidLay := testNet.LayerByName("Hidden").(*Layer)
	outLay := testNet.LayerByName("Output").(*Layer)

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

	hidDwts := []float32{3.8191774e-06, 8.762097e-06, 3.4446257e-06, 9.452213e-06}
	outDwts := []float32{0.008366027, 0.010467653, 0.009353173, 0.010706494}
	// 0.008427641, 0.010844037, 0.009319315, 0.011078155}
	hidWts := []float32{0.5000229, 0.5000526, 0.50002074, 0.50005686}
	outWts := []float32{0.55003285, 0.56248677, 0.555891, 0.5638975}
	// 0.55039877, 0.56470954, 0.55569035, 0.56609094}

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

		testNet.ConfigGPUnoGUI(ctx)

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
			testNet.GPU.SyncNeuronsFmGPU() // RLRate updated after plus

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			hidLay.UnitVals(&hidRLRate, "RLRate")
			ridx := ti*4*4 + pi*4
			copy(hidrlrs[ridx:ridx+4], hidRLRate)

			testNet.DWt(ctx)
			testNet.GPU.SyncSynapsesFmGPU()

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctx)
			testNet.GPU.SyncSynapsesFmGPU()

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)
		}
	}

	tol := float32(1.0e-6) // these can end up diverging significantly over time..
	// the synaptic ca integration etc.

	cmprFloats(hidrlrs, patHidRLRates, "hid RLRate", t)

	cmprFloats(q3hidCaP, qtr3HidCaP, "qtr 3 hidCaP", t)
	cmprFloats(q3hidCaD, qtr3HidCaD, "qtr 3 hidCaD", t)
	cmprFloats(q3outCaP, qtr3OutCaP, "qtr 3 outCaP", t)
	cmprFloats(q3outCaD, qtr3OutCaD, "qtr 3 outCaD", t)

	cmprFloatsTol(hiddwt, hidDwts, "hid DWt", t, tol)
	cmprFloatsTol(outdwt, outDwts, "out DWt", t, tol)
	cmprFloatsTol(hidwt, hidWts, "hid Wt", t, tol)
	cmprFloatsTol(outwt, outWts, "out Wt", t, tol)

	testNet.GPU.Destroy()
}
