// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

// difTol is the numerical difference tolerance for comparing vs. target values
const difTol = float32(1.0e-8)

var TestNet Network
var InPats *etensor.Float32

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

func TestMakeNet(t *testing.T) {
	TestNet.InitName(&TestNet, "TestNet")
	inLay := TestNet.AddLayer("Input", []int{4, 1}, emer.Input)
	hidLay := TestNet.AddLayer("Hidden", []int{4, 1}, emer.Hidden)
	outLay := TestNet.AddLayer("Output", []int{4, 1}, emer.Target)

	TestNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), emer.Forward)
	TestNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), emer.Forward)
	TestNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), emer.Back)

	TestNet.Defaults()
	TestNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	TestNet.Build()
	TestNet.InitWts()
	TestNet.NewState() // get GScale

	// var buf bytes.Buffer
	// TestNet.WriteWtsJSON(&buf)
	// wb := buf.Bytes()
	// // fmt.Printf("TestNet Weights:\n\n%v\n", string(wb))
	//
	// fp, err := os.Create("testdata/testnet.wts")
	// defer fp.Close()
	// if err != nil {
	// 	t.Error(err)
	// }
	// fp.Write(wb)
}

func TestSynVals(t *testing.T) {
	TestNet.InitWts()
	hidLay := TestNet.LayerByName("Hidden").(*axon.Layer)
	fmIn := hidLay.RcvPrjns.SendName("Input").(*axon.Prjn)

	bfWt := fmIn.SynVal("Wt", 1, 1)
	if mat32.IsNaN(bfWt) {
		t.Errorf("Wt syn var not found")
	}
	bfLWt := fmIn.SynVal("LWt", 1, 1)

	fmIn.SetSynVal("Wt", 1, 1, .15)

	afWt := fmIn.SynVal("Wt", 1, 1)
	afLWt := fmIn.SynVal("LWt", 1, 1)

	CmprFloats([]float32{bfWt, bfLWt, afWt, afLWt}, []float32{0.5, 0.5, 0.15, 0.42822415}, "syn val setting test", t)

	// fmt.Printf("SynVals: before wt: %v, lwt: %v  after wt: %v, lwt: %v\n", bfWt, bfLWt, afWt, afLWt)
}

func TestInPats(t *testing.T) {
	InPats = etensor.NewFloat32([]int{4, 4, 1}, nil, []string{"pat", "Y", "X"})
	for pi := 0; pi < 4; pi++ {
		InPats.Set([]int{pi, pi, 0}, 1)
	}
}

func CmprFloats(out, cor []float32, msg string, t *testing.T) {
	for i := range out {
		dif := mat32.Abs(out[i] - cor[i])
		if dif > difTol { // allow for small numerical diffs
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

	ltime := axon.NewTime()

	pat := etensor.NewFloat32([]int{1, 1}, nil, []string{"Y", "X"})
	pat.Set([]int{0, 0}, 1)

	for del := 0; del <= 4; del++ {
		prj.Com.Delay = del
		net.InitWts()  // resets Gbuf
		net.NewState() // get GScale

		inLay.ApplyExt(pat)

		net.NewState()
		ltime.NewState("Train")

		inCyc := 0
		hidCyc := 0
		for cyc := 0; cyc < 100; cyc++ {
			net.Cycle(ltime)
			ltime.CycleInc()

			if inLay.Neurons[0].Spike > 0 {
				inCyc = cyc
			}

			// if inCyc > 0 {
			// 	fmt.Printf("del: %d   cyc %d   inCyc: %d   Zi: %d  gbuf: %v\n", del, cyc, inCyc, prj.Gidx.Zi, prj.Gbuf)
			// }
			ge := hidLay.Neurons[0].Ge
			if ge > 0 {
				hidCyc = cyc
				break
			}
		}
		// fmt.Printf("del: %d   inCyc: %d   hidCyc: %d\n", del, inCyc, hidCyc)
		if hidCyc-inCyc != del+1 {
			t.Errorf("SpikeProp error -- delay: %d  actual: %d\n", del+1, hidCyc-inCyc)
		}
	}
}

func TestNetAct(t *testing.T) {
	TestNet.InitWts()
	TestNet.InitExt()

	inLay := TestNet.LayerByName("Input").(*axon.Layer)
	hidLay := TestNet.LayerByName("Hidden").(*axon.Layer)
	outLay := TestNet.LayerByName("Output").(*axon.Layer)

	ltime := axon.NewTime()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.6291466, 0, 0, 0}
	qtr0HidGes := []float32{0.5408259, 0, 0, 0}
	qtr0HidGis := []float32{0.12581292, 0.12581292, 0.12581292, 0.12581292}
	qtr0OutActs := []float32{0.5448547, 0, 0, 0}
	qtr0OutGes := []float32{0.17616737, 0, 0, 0}
	qtr0OutGis := []float32{0.056886084, 0.056886084, 0.056886084, 0.056886084}

	qtr3HidActs := []float32{0.6275331, 0, 0, 0}
	qtr3HidGes := []float32{0.50843596, 0, 0, 0}
	qtr3HidGis := []float32{0.18011907, 0.18011907, 0.18011907, 0.18011907}
	qtr3OutActs := []float32{0.66682786, 0, 0, 0}
	qtr3OutGes := []float32{0.6, 0, 0, 0}
	qtr3OutGis := []float32{0.19600207, 0.19600207, 0.19600207, 0.19600207}

	inActs := []float32{}
	hidActs := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	outActs := []float32{}
	outGes := []float32{}
	outGis := []float32{}

	cycPerQtr := 50

	for pi := 0; pi < 4; pi++ {
		inpat, err := InPats.SubSpaceTry([]int{pi})
		if err != nil {
			t.Error(err)
		}
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)

		TestNet.NewState()
		ltime.NewState("Train")

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				TestNet.Cycle(ltime)
				ltime.CycleInc()

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
				TestNet.MinusPhase(ltime)
				ltime.NewPhase(false)
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
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ltime.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				CmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				CmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				CmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				CmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				CmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				CmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				CmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				CmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				CmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				CmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				CmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				CmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
			}
		}
		TestNet.PlusPhase(ltime)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}

func TestNetLearn(t *testing.T) {
	inLay := TestNet.LayerByName("Input").(*axon.Layer)
	hidLay := TestNet.LayerByName("Hidden").(*axon.Layer)
	outLay := TestNet.LayerByName("Output").(*axon.Layer)

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
	hidDwts := []float32{0.0015839314, 0.0018261874, 0.0018425429, 0.0018425632}
	outDwts := []float32{0.00233706, 0.0025052512, 0.0024995792, 0.0024995983}
	hidWts := []float32{0.5, 0.5, 0.5, 0.5} // todo: not clear why not updating..
	outWts := []float32{0.5140186, 0.5150271, 0.51499313, 0.51499313}

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
		TestNet.Defaults()
		TestNet.ApplyParams(ParamSets[0].Sheets["Network"], false)  // always apply base
		TestNet.ApplyParams(ParamSets[ti].Sheets["Network"], false) // then specific
		TestNet.InitWts()
		TestNet.InitExt()

		ltime := axon.NewTime()

		for pi := 0; pi < 4; pi++ {
			inpat, err := InPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			inLay.ApplyExt(inpat)
			outLay.ApplyExt(inpat)

			TestNet.NewState()
			ltime.NewState("Train")
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					TestNet.Cycle(ltime)
					ltime.CycleInc()

					hidLay.UnitVals(&hidAct, "Act")
					hidLay.UnitVals(&hidGes, "Ge")
					hidLay.UnitVals(&hidGis, "Gi")
					hidLay.UnitVals(&hidSpkCaM, "SpkCaM")
					hidLay.UnitVals(&hidSpkCaP, "SpkCaP")
					hidLay.UnitVals(&hidSpkCaD, "SpkCaD")

					outLay.UnitVals(&outSpkCaP, "SpkCaP")
					outLay.UnitVals(&outSpkCaD, "SpkCaD")

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ltime.Cycle, hidAct, hidGes, hidGis, hidSpkCaM, hidSpkCaP, hidSpkCaD, outSpkCaP, outSpkCaD)
					}
				}
				if qtr == 2 {
					TestNet.MinusPhase(ltime)
					ltime.NewPhase(false)
				}

				hidLay.UnitVals(&hidSpkCaP, "SpkCaP")
				hidLay.UnitVals(&hidSpkCaD, "SpkCaD")

				outLay.UnitVals(&outSpkCaP, "SpkCaP")
				outLay.UnitVals(&outSpkCaD, "SpkCaD")

				if printQtrs {
					fmt.Printf("pat: %v qtr: %v cyc: %v\nhid avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ltime.Cycle, hidSpkCaP, hidSpkCaD, outSpkCaP, outSpkCaD)
				}

				if pi == 0 && qtr == 0 {
					CmprFloats(hidSpkCaP, qtr0HidSpkCaP, "qtr 0 hidSpkCaP", t)
					CmprFloats(hidSpkCaD, qtr0HidSpkCaD, "qtr 0 hidSpkCaD", t)
					CmprFloats(outSpkCaP, qtr0OutSpkCaP, "qtr 0 outSpkCaP", t)
					CmprFloats(outSpkCaD, qtr0OutSpkCaD, "qtr 0 outSpkCaD", t)
				}
				if pi == 0 && qtr == 3 {
					CmprFloats(hidSpkCaP, qtr3HidSpkCaP, "qtr 3 hidSpkCaP", t)
					CmprFloats(hidSpkCaD, qtr3HidSpkCaD, "qtr 3 hidSpkCaD", t)
					CmprFloats(outSpkCaP, qtr3OutSpkCaP, "qtr 3 outSpkCaP", t)
					CmprFloats(outSpkCaD, qtr3OutSpkCaD, "qtr 3 outSpkCaD", t)
				}
			}
			TestNet.PlusPhase(ltime)

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			// fmt.Printf("hid cosdif stats: %v\nhid avgl:   %v\nhid avgllrn: %v\n", hidLay.CosDiff, hidAvgL, hidAvgLLrn)
			// fmt.Printf("out cosdif stats: %v\nout avgl:   %v\nout avgllrn: %v\n", outLay.CosDiff, outAvgL, outAvgLLrn)

			TestNet.DWt(ltime)

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			TestNet.WtFmDWt(ltime)

			hidwt[didx] = hidLay.RcvPrjns[0].SynVal("Wt", pi, pi)
			outwt[didx] = outLay.RcvPrjns[0].SynVal("Wt", pi, pi)
		}
	}

	//	fmt.Printf("hid dwt: %v\nout dwt: %v\nhid norm: %v\n hid moment: %v\nout norm: %v\nout moment: %v\nhid wt: %v\nout wt: %v\n", hiddwt, outdwt, hidnorm, hidmoment, outnorm, outmoment, hidwt, outwt)

	CmprFloats(hiddwt, hidDwts, "hid DWt", t)
	CmprFloats(outdwt, outDwts, "out DWt", t)
	CmprFloats(hidwt, hidWts, "hid Wt", t)
	CmprFloats(outwt, outWts, "out Wt", t)

	// var buf bytes.Buffer
	// TestNet.WriteWtsJSON(&buf)
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

	ltime := axon.NewTime()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.6019061, 0, 0, 0}
	qtr0HidGes := []float32{0.6206174, 0, 0, 0}
	qtr0HidGis := []float32{0.06083918, 0, 0, 0}
	qtr0OutActs := []float32{0.6408267, 0, 0, 0}
	qtr0OutGes := []float32{0.2103362, 0, 0, 0}
	qtr0OutGis := []float32{0, 0, 0, 0}

	qtr3HidActs := []float32{0.91008276, 0, 0, 0}
	qtr3HidGes := []float32{0.8942689, 0, 0, 0}
	qtr3HidGis := []float32{0.060876425, 0, 0, 0}
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
		inpat, err := InPats.SubSpaceTry([]int{pi})
		if err != nil {
			t.Error(err)
		}
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)

		InhibNet.NewState()
		ltime.NewState("Train")
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				InhibNet.Cycle(ltime)
				ltime.CycleInc()

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
				InhibNet.MinusPhase(ltime)
				ltime.NewPhase(false)
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
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ltime.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				CmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				CmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				CmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				CmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				CmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				CmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				CmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				CmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				CmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				CmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				CmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				CmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
			}
		}
		TestNet.PlusPhase(ltime)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}
