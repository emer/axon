// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package deep

import (
	"fmt"
	"testing"

	"github.com/Astera-org/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

// TOLERANCE is the numerical difference tolerance for comparing vs. target values
const TOLERANCE = float32(1.0e-8)

// number of distinct sets of learning parameters to test
const NLrnPars = 1

// Note: subsequent params applied after Base
var ParamSets = params.Sets{
	{Name: "Base", Desc: "base testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Act.Gbar.L":      "0.2",
					"Layer.Learn.RLrate.On": "false",
				}},
			{Sel: "Prjn", Desc: "for reproducibility, identical weights",
				Params: params.Params{
					"Prjn.SWt.Init.Var": "0",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
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
			{Sel: ".Inhib", Desc: "weaker inhib",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
		},
	}},
}

func newTestNet() *Network {
	var testNet Network
	testNet.InitName(&testNet, "TestNet")
	inLay := testNet.AddLayer("Input", []int{4, 1}, emer.Input)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, emer.Hidden)
	outLay := testNet.AddLayer("Output", []int{4, 1}, emer.Target)

	testNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), emer.Forward)
	testNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), emer.Forward)
	testNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), emer.Back)

	testNet.Defaults()
	testNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.Build()
	testNet.InitWts()
	testNet.NewState() // get GScale

	return &testNet
	// var buf bytes.Buffer
	// testNet.WriteWtsJSON(&buf)
	// wb := buf.Bytes()
	// fmt.Printf("TestNet Weights:\n\n%v\n", string(wb))
	//
	// fp, err := os.Create("testdata/testnet.wts")
	// defer fp.Close()
	// if err != nil {
	// 	t.Error(err)
	// }
	// fp.Write(wb)
}

func TestSynVals(t *testing.T) {
	testNet := newTestNet()
	hidLay := testNet.LayerByName("Hidden").(*axon.Layer)
	fmIn := hidLay.RcvPrjns.SendName("Input").(*axon.Prjn)

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

func newInPats() (inPats *etensor.Float32) {
	inPats = etensor.NewFloat32([]int{4, 4, 1}, nil, []string{"pat", "Y", "X"})
	for pi := 0; pi < 4; pi++ {
		inPats.Set([]int{pi, pi, 0}, 1)
	}
	return
}

func cmprFloats(out, cor []float32, msg string, t *testing.T) {
	t.Helper()
	for i := range out {
		dif := mat32.Abs(out[i] - cor[i])
		if dif > TOLERANCE { // allow for small numerical diffs
			t.Errorf("%v err: out: %v, cor: %v, dif: %v\n", msg, out[i], cor[i], dif)
		}
	}
}

func TestSpikeProp(t *testing.T) {
	net := NewNetwork("SpikeNet")
	inLay := net.AddLayer("Input", []int{1, 1}, emer.Input).(*axon.Layer)
	hidLay := net.AddLayer("Hidden", []int{1, 1}, emer.Hidden).(*axon.Layer)

	prj := net.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), emer.Forward).(*axon.Prjn)

	net.Defaults()
	net.ApplyParams(ParamSets[0].Sheets["Network"], false)
	net.Build()

	net.InitExt()

	ctime := axon.NewTime()

	pat := etensor.NewFloat32([]int{1, 1}, nil, []string{"Y", "X"})
	pat.Set([]int{0, 0}, 1)

	for del := 0; del <= 4; del++ {
		prj.Com.Delay = del
		net.InitWts()  // resets Gbuf
		net.NewState() // get GScale

		inLay.ApplyExt(pat)

		net.NewState()
		ctime.NewState("Train")

		inCyc := 0
		hidCyc := 0
		for cyc := 0; cyc < 100; cyc++ {
			net.Cycle(ctime)
			ctime.CycleInc()

			if inLay.Neurons[0].Spike > 0 {
				inCyc = cyc
			}

			ge := hidLay.Neurons[0].Ge
			if ge > 0 {
				hidCyc = cyc
				break
			}
		}
		if hidCyc-inCyc != del {
			t.Errorf("SpikeProp error -- delay: %d  actual: %d\n", del, hidCyc-inCyc)
		}
	}
}

func TestNetAct(t *testing.T) {
	testNet := newTestNet()
	testNet.InitExt()

	inPats := newInPats()

	inLay := testNet.LayerByName("Input").(*axon.Layer)
	hidLay := testNet.LayerByName("Hidden").(*axon.Layer)
	outLay := testNet.LayerByName("Output").(*axon.Layer)

	ctime := axon.NewTime()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.72165483, 0, 0, 0}
	qtr0HidGes := []float32{0.49048427, 0, 0, 0}
	qtr0HidGis := []float32{0.13094407, 0.13094407, 0.13094407, 0.13094407}
	qtr0OutActs := []float32{0.63028616, 0, 0, 0}
	qtr0OutGes := []float32{0.3984899, 0, 0, 0}
	qtr0OutGis := []float32{0.07549549, 0.07549549, 0.07549549, 0.07549549}

	qtr3HidActs := []float32{0.6532401, 0, 0, 0}
	qtr3HidGes := []float32{0.7066621, 0, 0, 0}
	qtr3HidGis := []float32{0.24389985, 0.24389985, 0.24389985, 0.24389985}
	qtr3OutActs := []float32{0.69444436, 0, 0, 0}
	qtr3OutGes := []float32{0.6, 0, 0, 0}
	qtr3OutGis := []float32{0.20638072, 0.20638072, 0.20638072, 0.20638072}

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

		testNet.NewState()
		ctime.NewState("Train")

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				testNet.Cycle(ctime)
				ctime.CycleInc()

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
				testNet.MinusPhase(ctime)
				ctime.NewPhase(false)
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
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctime.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
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
		testNet.PlusPhase(ctime)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}

func TestNetLearn(t *testing.T) {
	testNet := newTestNet()
	inLay := testNet.LayerByName("Input").(*axon.Layer)
	hidLay := testNet.LayerByName("Hidden").(*axon.Layer)
	outLay := testNet.LayerByName("Output").(*axon.Layer)

	inPats := newInPats()

	printCycs := false
	printQtrs := false

	qtr0HidSpkCaP := []float32{0.5697344, 0.054755755, 0.054755755, 0.054755755}
	qtr0HidSpkCaD := []float32{0.30031276, 0.10729555, 0.10729555, 0.10729555}
	qtr0OutSpkCaP := []float32{0.40196124, 0.054755755, 0.054755755, 0.054755755}
	qtr0OutSpkCaD := []float32{0.21776359, 0.10729555, 0.10729555, 0.10729555}

	qtr3HidSpkCaP := []float32{0.89222527, 0.0012329844, 0.0012329844, 0.0012329844}
	qtr3HidSpkCaD := []float32{0.8385092, 0.0070280116, 0.0070280116, 0.0070280116}
	qtr3OutSpkCaP := []float32{0.8830335, 0.0012329844, 0.0012329844, 0.0012329844}
	qtr3OutSpkCaD := []float32{0.7841259, 0.0070280116, 0.0070280116, 0.0070280116}

	// these are organized by pattern within and then by test iteration (params) outer
	hidDwts := []float32{0.0036074303, 0.0068956804, 0.0009886026, 0.0009886026}
	outDwts := []float32{0.009111724, 0.01502119, 0.007415774, 0.007415774}
	hidWts := []float32{0.5, 0.5, 0.5, 0.5} // todo: not clear why not updating..
	outWts := []float32{0.55445933, 0.5891899, 0.54438084, 0.54438084}

	hiddwt := make([]float32, 4*NLrnPars)
	outdwt := make([]float32, 4*NLrnPars)
	hidwt := make([]float32, 4*NLrnPars)
	outwt := make([]float32, 4*NLrnPars)

	hidAct := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	hidSpkCaM := []float32{}
	hidSpkCaP := []float32{}
	hidSpkCaD := []float32{}
	outSpkCaP := []float32{}
	outSpkCaD := []float32{}

	cycPerQtr := 50

	for ti := 0; ti < NLrnPars; ti++ {
		testNet.Defaults()
		testNet.ApplyParams(ParamSets[0].Sheets["Network"], false)  // always apply base
		testNet.ApplyParams(ParamSets[ti].Sheets["Network"], false) // then specific
		testNet.InitWts()
		testNet.InitExt()

		ctime := axon.NewTime()

		for pi := 0; pi < 4; pi++ {
			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			inLay.ApplyExt(inpat)
			outLay.ApplyExt(inpat)

			testNet.NewState()
			ctime.NewState("Train")
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					testNet.Cycle(ctime)
					ctime.CycleInc()

					hidLay.UnitVals(&hidAct, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					hidLay.UnitVals(&hidSpkCaM, "SpkCaM")
					hidLay.UnitVals(&hidSpkCaP, "SpkCaP")
					hidLay.UnitVals(&hidSpkCaD, "SpkCaD")

					outLay.UnitVals(&outSpkCaP, "SpkCaP")
					outLay.UnitVals(&outSpkCaD, "SpkCaD")

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctime.Cycle, hidAct, hidGes, hidGis, hidSpkCaM, hidSpkCaP, hidSpkCaD, outSpkCaP, outSpkCaD)
					}
				}
				if qtr == 2 {
					testNet.MinusPhase(ctime)
					ctime.NewPhase(false)
				}

				hidLay.UnitVals(&hidSpkCaP, "SpkCaP")
				hidLay.UnitVals(&hidSpkCaD, "SpkCaD")

				outLay.UnitVals(&outSpkCaP, "SpkCaP")
				outLay.UnitVals(&outSpkCaD, "SpkCaD")

				if printQtrs {
					fmt.Printf("pat: %v qtr: %v cyc: %v\nhid avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctime.Cycle, hidSpkCaP, hidSpkCaD, outSpkCaP, outSpkCaD)
				}

				if pi == 0 && qtr == 0 {
					cmprFloats(hidSpkCaP, qtr0HidSpkCaP, "qtr 0 hidSpkCaP", t)
					cmprFloats(hidSpkCaD, qtr0HidSpkCaD, "qtr 0 hidSpkCaD", t)
					cmprFloats(outSpkCaP, qtr0OutSpkCaP, "qtr 0 outSpkCaP", t)
					cmprFloats(outSpkCaD, qtr0OutSpkCaD, "qtr 0 outSpkCaD", t)
				}
				if pi == 0 && qtr == 3 {
					cmprFloats(hidSpkCaP, qtr3HidSpkCaP, "qtr 3 hidSpkCaP", t)
					cmprFloats(hidSpkCaD, qtr3HidSpkCaD, "qtr 3 hidSpkCaD", t)
					cmprFloats(outSpkCaP, qtr3OutSpkCaP, "qtr 3 outSpkCaP", t)
					cmprFloats(outSpkCaD, qtr3OutSpkCaD, "qtr 3 outSpkCaD", t)
				}
			}
			testNet.PlusPhase(ctime)

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			testNet.DWt(ctime)

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctime)

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)
		}
	}

	cmprFloats(hiddwt, hidDwts, "hid DWt", t)
	cmprFloats(outdwt, outDwts, "out DWt", t)
	cmprFloats(hidwt, hidWts, "hid Wt", t)
	cmprFloats(outwt, outWts, "out Wt", t)

	// var buf bytes.Buffer
	// testNet.WriteWtsJSON(&buf)
	// wb := buf.Bytes()
	// fmt.Printf("TestNet Trained Weights:\n\n%v\n", string(wb))

	// fp, err := os.Create("testdata/testnet_train.wts")
	// defer fp.Close()
	// if err != nil {
	// 	t.Error(err)
	// }
	// fp.Write(wb)
}

func TestInhibAct(t *testing.T) {
	var InhibNet Network
	InhibNet.InitName(&InhibNet, "InhibNet")

	inPats := newInPats()

	inLay := InhibNet.AddLayer("Input", []int{4, 1}, emer.Input).(*axon.Layer)
	hidLay := InhibNet.AddLayer("Hidden", []int{4, 1}, emer.Hidden).(*axon.Layer)
	outLay := InhibNet.AddLayer("Output", []int{4, 1}, emer.Target).(*axon.Layer)

	InhibNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), emer.Forward)
	InhibNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), emer.Inhib)
	InhibNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), emer.Forward)
	InhibNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), emer.Back)

	InhibNet.Defaults()
	InhibNet.ApplyParams(ParamSets[0].Sheets["Network"], false)
	InhibNet.ApplyParams(ParamSets[0].Sheets["InhibOff"], false)
	InhibNet.Build()
	InhibNet.InitWts()
	InhibNet.NewState() // get GScale

	InhibNet.InitWts()
	InhibNet.InitExt()

	ctime := axon.NewTime()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.80708516, 0, 0, 0}
	qtr0HidGes := []float32{0.66517055, 0, 0, 0}
	qtr0HidGis := []float32{0.05214787, 0, 0, 0}
	qtr0OutActs := []float32{0.92420554, 0, 0, 0}
	qtr0OutGes := []float32{0.4682724, 0, 0, 0}
	qtr0OutGis := []float32{0, 0, 0, 0}

	qtr3HidActs := []float32{0.9086095, 0, 0, 0}
	qtr3HidGes := []float32{0.9144331, 0, 0, 0}
	qtr3HidGis := []float32{0.05217979, 0, 0, 0}
	qtr3OutActs := []float32{0.7936507, 0, 0, 0}
	qtr3OutGes := []float32{0.6, 0, 0, 0}
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

		InhibNet.NewState()
		ctime.NewState("Train")
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				InhibNet.Cycle(ctime)
				ctime.CycleInc()

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
				InhibNet.MinusPhase(ctime)
				ctime.NewPhase(false)
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
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctime.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
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
		InhibNet.PlusPhase(ctime)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}
