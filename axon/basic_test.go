// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build multinet

package axon

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
	"golang.org/x/exp/maps"
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
					"Layer.Acts.Gbar.L":     "0.2",
					"Layer.Learn.RLRate.On": "false",
					"Layer.Inhib.Layer.FB":  "0.5",
				}},
			{Sel: "Prjn", Desc: "for reproducibility, identical weights",
				Params: params.Params{
					"Prjn.SWts.Init.Var": "0",
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
		},
		"InhibOff": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Gbar.L":    "0.2",
					"Layer.Inhib.Layer.On": "false",
				}},
			{Sel: ".InhibPrjn", Desc: "weaker inhib",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
		},
	}},
	{Name: "FullDecay", Desc: "decay state completely for ndata testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Decay.Act":   "1",
					"Layer.Acts.Decay.Glong": "1",
					"Layer.Acts.Decay.AHP":   "1",
				}},
		},
	}},
	{Name: "SubMean", Desc: "submean on Prjn dwt", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "submean used in some models but not by default",
				Params: params.Params{
					"Prjn.Learn.Trace.SubMean": "1",
				}},
		},
	}},
}

func newTestNet(ctx *Context, nData int) *Network {
	var testNet Network
	testNet.InitName(&testNet, "testNet")
	testNet.SetRndSeed(42) // critical for ActAvg values
	testNet.MaxData = uint32(nData)

	inLay := testNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), BackPrjn)

	testNet.Build(ctx)
	ctx.NetIdxs.NData = uint32(nData)
	testNet.Defaults()
	testNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWts(ctx)                                       // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

// full connectivity
func newTestNetFull(ctx *Context, nData int) *Network {
	var testNet Network
	testNet.InitName(&testNet, "testNet")
	testNet.SetRndSeed(42) // critical for ActAvg values
	testNet.MaxData = uint32(nData)

	inLay := testNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	full := prjn.NewFull()
	testNet.ConnectLayers(inLay, hidLay, full, ForwardPrjn)
	testNet.ConnectLayers(hidLay, outLay, full, ForwardPrjn)
	testNet.ConnectLayers(outLay, hidLay, full, BackPrjn)

	testNet.Build(ctx)
	ctx.NetIdxs.NData = uint32(nData)
	testNet.Defaults()
	testNet.ApplyParams(ParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWts(ctx)                                       // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

func TestSynVals(t *testing.T) {
	ctx := NewContext()
	testNet := newTestNet(ctx, 1)
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

// StructVals adds field vals to given vals map
func StructVals(obj any, vals map[string]float32, key string) {
	v := kit.NonPtrValue(reflect.ValueOf(obj))
	typ := v.Type()
	for i := 0; i < v.NumField(); i++ {
		ft := typ.Field(i)
		if !ft.IsExported() {
			continue
		}
		fv := v.Field(i)
		kk := key + fmt.Sprintf("\t%s", ft.Name)
		vals[kk], _ = kit.ToFloat32(fv.Interface())
	}
}

// TestInitWts tests that initializing the weights results in same state
func TestInitWts(t *testing.T) {
	nData := 4
	ctx := NewContext()
	testNet := newTestNet(ctx, nData)
	inPats := newInPats()

	valMapA := make(map[string]float32)
	valMapB := make(map[string]float32)

	inLay := testNet.AxonLayerByName("Input")
	outLay := testNet.AxonLayerByName("Output")

	var vals []float32

	valMap := valMapA
	for wi := 0; wi < 2; wi++ {
		if wi == 1 {
			valMap = valMapB
		}
		testNet.SetRndSeed(42) // critical for ActAvg values
		testNet.InitWts(ctx)
		testNet.InitExt(ctx)
		for ni := 0; ni < 4; ni++ {
			for li := 0; li < 3; li++ {
				ly := testNet.Layers[li]
				for di := 0; di < nData; di++ {
					key := fmt.Sprintf("Layer: %s\tUnit: %d\tDi: %d", ly.Nm, ni, di)
					for _, vnm := range NeuronVarNames {
						ly.UnitVals(&vals, vnm, di)
						vkey := key + fmt.Sprintf("\t%s", vnm)
						valMap[vkey] = vals[ni]
					}
				}
			}
		}
		for li := 0; li < 3; li++ {
			ly := testNet.Layers[li]
			for di := 0; di < nData; di++ {
				lpl := ly.Pool(0, uint32(di))
				lnm := fmt.Sprintf("%s: di: %d", ly.Nm, di)
				StructVals(&lpl.Inhib, valMap, lnm)
			}
		}

		for pi := 0; pi < 4; pi++ {
			ctx.NewState(etime.Train)
			testNet.NewState(ctx)

			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt(ctx)
			for di := 0; di < nData; di++ {
				inLay.ApplyExt(ctx, uint32(di), inpat)
				outLay.ApplyExt(ctx, uint32(di), inpat)
			}
			testNet.ApplyExts(ctx) // key now for GPU

			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < 50; cyc++ {
					testNet.Cycle(ctx)
					ctx.CycleInc()
				}
				if qtr == 2 {
					testNet.MinusPhase(ctx)
					ctx.NewPhase(false)
					testNet.PlusPhaseStart(ctx)
				}
			}
			testNet.PlusPhase(ctx)
			testNet.DWt(ctx)
			testNet.WtFmDWt(ctx)
		}
	}
	ReportValDiffs(t, valMapA, valMapB, "init1", "init2", nil)
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

// NetActTest runs an activation test on the network and checks
// for key values relative to known standards.
// Note: use NetActDebug for printf debugging of all values -- "this is only a test"
func NetActTest(t *testing.T, gpu bool) {
	ctx := NewContext()
	testNet := newTestNet(ctx, 1)
	testNet.InitExt(ctx)
	inPats := newInPats()

	inLay := testNet.AxonLayerByName("Input")
	hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
		// testNet.GPU.RecFunTimes = true // alt modes
		// testNet.GPU.CycleByCycle = true // alt modes
	}

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
		testNet.NewState(ctx)
		ctx.NewState(etime.Train)

		inpat, err := inPats.SubSpaceTry([]int{pi})
		if err != nil {
			t.Fatal(err)
		}
		testNet.InitExt(ctx)
		inLay.ApplyExt(ctx, 0, inpat)
		outLay.ApplyExt(ctx, 0, inpat)
		testNet.ApplyExts(ctx) // key now for GPU

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < cycPerQtr; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()
				if gpu {
					testNet.GPU.SyncNeuronsFmGPU()
				}
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				testNet.PlusPhaseStart(ctx)
			}

			inLay.UnitVals(&inActs, "Act", 0)
			hidLay.UnitVals(&hidActs, "Act", 0)
			hidLay.UnitVals(&hidGes, "Ge", 0)
			hidLay.UnitVals(&hidGis, "Gi", 0)
			outLay.UnitVals(&outActs, "Act", 0)
			outLay.UnitVals(&outGes, "Ge", 0)
			outLay.UnitVals(&outGis, "Gi", 0)

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
	}

	testNet.GPU.Destroy()
}

func TestGPUDiffs(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	nonGPUVals := NetDebugAct(t, false, false, 1, false)
	gpuVals := NetDebugAct(t, false, true, 1, false)
	ReportValDiffs(t, nonGPUVals, gpuVals, "CPU", "GPU", nil)
}

func TestDebugAct(t *testing.T) {
	t.Skip("skipped in regular testing")
	NetDebugAct(t, true, false, 1, false)
}

func TestDebugGPUAct(t *testing.T) {
	t.Skip("skipped in regular testing")
	NetDebugAct(t, true, true, 1, false)
}

func TestNDataDiffs(t *testing.T) {
	nd1Vals := NetDebugAct(t, false, false, 1, true)
	nd4Vals := NetDebugAct(t, false, false, 4, true)
	ReportValDiffs(t, nd1Vals, nd4Vals, "nData = 1", "nData = 4", nil)
}

func TestGPUNDataDiffs(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	nd1Vals := NetDebugAct(t, false, true, 1, true)
	nd4Vals := NetDebugAct(t, false, true, 4, true)
	ReportValDiffs(t, nd1Vals, nd4Vals, "nData = 1", "nData = 4", nil)
}

// ReportValDiffs
func ReportValDiffs(t *testing.T, va, vb map[string]float32, aLabel, bLabel string, exclude []string) {
	const TOLERANCE = float32(1.0e-4) // GPU Nmda has genuine diffs beyond e-5, accumulate over time..
	keys := maps.Keys(va)
	sort.Strings(keys)
	nerrs := 0
	for _, k := range keys {
		hasEx := false
		for _, ex := range exclude {
			if strings.Contains(k, ex) {
				hasEx = true
				break
			}
		}
		if hasEx {
			continue
		}
		av := va[k]
		bv := vb[k]
		dif := mat32.Abs(av - bv)
		if dif > TOLERANCE { // allow for small numerical diffs
			if nerrs == 0 {
				t.Errorf("Diffs found between two runs (10 max): A = %s  B = %s\n", aLabel, bLabel)
			}
			fmt.Printf("%s\tA: %g\tB: %g\tDiff: %g\n", k, av, bv, dif)
			nerrs++
			if nerrs > 100 {
				fmt.Printf("Max diffs exceeded, increase for more\n")
				break
			}
		}
	}
}

// NetDebugAct prints selected values (if printVals),
// and also returns a map of all values and variables that can be used for a more
// fine-grained diff test, e.g., see the GPU version.
func NetDebugAct(t *testing.T, printVals bool, gpu bool, nData int, initWts bool) map[string]float32 {
	ctx := NewContext()
	testNet := newTestNet(ctx, nData)

	testNet.ApplyParams(ParamSets.SetByName("FullDecay").Sheets["Network"], false)

	testNet.InitExt(ctx)
	ctx.NetIdxs.NData = uint32(nData)

	valMap := make(map[string]float32)

	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	// hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")
	_, _ = inLay, outLay

	var vals []float32

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
		// testNet.GPU.RecFunTimes = true
		testNet.GPU.CycleByCycle = true // key for printing results cycle-by-cycle
	}

	// these control what is printed.
	// the whole thing is run and returned in the valMap
	valsPerRow := 8
	nQtrs := 1     // max 4
	cycPerQtr := 5 // max 50
	nPats := 2     // max 4
	stLayer := 1   // max 2
	edLayer := 2   // max 3
	nNeurs := 1    // max 4 -- number of neuron values to print

	for pi := 0; pi < 4; pi++ {
		if initWts {
			testNet.SetRndSeed(42) // critical for ActAvg values
			testNet.InitWts(ctx)
		} else {
			testNet.NewState(ctx)
		}
		ctx.NewState(etime.Train)

		testNet.InitExt(ctx)
		for di := 0; di < nData; di++ {
			ppi := (pi + di) % 4
			inpat, err := inPats.SubSpaceTry([]int{ppi})
			if err != nil {
				t.Fatal(err)
			}
			_ = inpat
			inLay.ApplyExt(ctx, uint32(di), inpat)
			outLay.ApplyExt(ctx, uint32(di), inpat)
		}

		testNet.ApplyExts(ctx) // key now for GPU

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < 50; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()
				if gpu {
					testNet.GPU.SyncNeuronsFmGPU()
				}

				for ni := 0; ni < 4; ni++ {
					for li := 0; li < 3; li++ {
						ly := testNet.Layers[li]
						for di := 0; di < nData; di++ {
							ppi := (pi + di) % 4
							key := fmt.Sprintf("pat: %d\tqtr: %d\tcyc: %02d\tLayer: %s\tUnit: %d", ppi, qtr, cyc, ly.Nm, ni)
							doPrint := (printVals && pi < nPats && qtr < nQtrs && cyc < cycPerQtr && ni < nNeurs && li >= stLayer && li < edLayer)
							if doPrint {
								fmt.Println(key)
							}
							for nvi, vnm := range NeuronVarNames {
								ly.UnitVals(&vals, vnm, di)
								vkey := key + fmt.Sprintf("\t%s", vnm)
								valMap[vkey] = vals[ni]
								if doPrint {
									fmt.Printf("\t%-10s%7.4f", vnm, vals[ni])
									if (int(nvi)+1)%valsPerRow == 0 {
										fmt.Printf("\n")
									}
								}
							}
							if doPrint {
								fmt.Printf("\n")
							}
						}
					}
				}
				for li := 0; li < 3; li++ {
					ly := testNet.Layers[li]
					for di := 0; di < nData; di++ {
						lpl := ly.Pool(0, uint32(di))
						lnm := fmt.Sprintf("%s: di: %d", ly.Nm, di)
						StructVals(&lpl.Inhib, valMap, lnm)
					}
				}
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				testNet.PlusPhaseStart(ctx)
			}
		}

		testNet.PlusPhase(ctx)
		pi += nData - 1
	}

	testNet.GPU.Destroy()
	return valMap
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

	testNet := newTestNet(ctx, 1)

	// fmt.Printf("synbanks: %d\n", ctx.NetIdxs.NSynCaBanks)

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
			// testNet.GPU.RecFunTimes = true // alt forms
			// testNet.GPU.CycleByCycle = true //
		}

		for pi := 0; pi < 4; pi++ {
			ctx.NewState(etime.Train)
			testNet.NewState(ctx)

			inpat, err := inPats.SubSpaceTry([]int{pi})
			if err != nil {
				t.Error(err)
			}
			testNet.InitExt(ctx)
			inLay.ApplyExt(ctx, 0, inpat)
			outLay.ApplyExt(ctx, 0, inpat)
			testNet.ApplyExts(ctx) // key now for GPU

			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					testNet.Cycle(ctx)
					ctx.CycleInc()
					if gpu {
						testNet.GPU.SyncNeuronsFmGPU()
					}

					hidLay.UnitVals(&hidAct, "Act", 0)
					hidLay.UnitVals(&hidGes, "Ge", 0)
					hidLay.UnitVals(&hidGis, "Gi", 0)
					hidLay.UnitVals(&hidCaM, "NrnCaM", 0)
					hidLay.UnitVals(&hidCaP, "NrnCaP", 0)
					hidLay.UnitVals(&hidCaD, "NrnCaD", 0)

					outLay.UnitVals(&outCaP, "NrnCaP", 0)
					outLay.UnitVals(&outCaD, "NrnCaD", 0)

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAct, hidGes, hidGis, hidCaM, hidCaP, hidCaD, outCaP, outCaD)
					}
				}
				if qtr == 2 {
					testNet.MinusPhase(ctx)
					ctx.NewPhase(false)
					testNet.PlusPhaseStart(ctx)
				}

				hidLay.UnitVals(&hidCaP, "NrnCaP", 0)
				hidLay.UnitVals(&hidCaD, "NrnCaD", 0)

				outLay.UnitVals(&outCaP, "NrnCaP", 0)
				outLay.UnitVals(&outCaD, "NrnCaD", 0)

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
				testNet.GPU.SyncSynCaFmGPU()
			}

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RcvPrjns[0].SynVal("DWt", pi, pi)
			outdwt[didx] = outLay.RcvPrjns[0].SynVal("DWt", pi, pi)

			testNet.WtFmDWt(ctx)
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
				testNet.GPU.SyncSynCaFmGPU()
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

	testNet := newTestNet(ctx, 1)
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

					hidLay.UnitVals(&hidAct, "Act", 0)
					hidLay.UnitVals(&hidGes, "Ge", 0)
					hidLay.UnitVals(&hidGis, "Gi", 0)
					hidLay.UnitVals(&hidCaM, "NrnCaM", 0)
					hidLay.UnitVals(&hidCaP, "NrnCaP", 0)
					hidLay.UnitVals(&hidCaD, "NrnCaD", 0)

					outLay.UnitVals(&outCaP, "NrnCaP", 0)
					outLay.UnitVals(&outCaD, "NrnCaD", 0)

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAct, hidGes, hidGis, hidCaM, hidCaP, hidCaD, outCaP, outCaD)
					}
				}
				if qtr == 2 {
					testNet.MinusPhase(ctx)
					ctx.NewPhase(false)
					testNet.PlusPhaseStart(ctx)
				}

				hidLay.UnitVals(&hidCaP, "NrnCaP", 0)
				hidLay.UnitVals(&hidCaD, "NrnCaD", 0)

				outLay.UnitVals(&outCaP, "NrnCaP", 0)
				outLay.UnitVals(&outCaD, "NrnCaD", 0)

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

			hidLay.UnitVals(&hidRLRate, "RLRate", 0)
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

// NetDebugLearn prints selected values (if printVals),
// and also returns a map of all values and variables that can be used for a more
// fine-grained diff test, e.g., see the GPU version.
func NetDebugLearn(t *testing.T, printVals bool, gpu bool, nData int, initWts, submean, slowAdapt bool) map[string]float32 {
	ctx := NewContext()
	var testNet *Network
	rand.Seed(1337)

	if submean {
		testNet = newTestNetFull(ctx, nData) // otherwise no effect
	} else {
		testNet = newTestNet(ctx, nData)
	}

	testNet.SetRndSeed(42) // critical for ActAvg values

	testNet.ApplyParams(ParamSets.SetByName("FullDecay").Sheets["Network"], false)

	if submean {
		testNet.ApplyParams(ParamSets.SetByName("SubMean").Sheets["Network"], false)
	}

	testNet.InitExt(ctx)
	ctx.NetIdxs.NData = uint32(nData)

	valMap := make(map[string]float32)

	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	// hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")
	_, _ = inLay, outLay

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
		testNet.GPU.CycleByCycle = true // key for printing results cycle-by-cycle
	}

	// these control what is printed.
	// the whole thing is run and returned in the valMap
	valsPerRow := 8
	nPats := 4   // max 4
	stLayer := 1 // max 2
	edLayer := 2 // max 3
	nNeurs := 4  // max 4 -- number of neuron values to print
	var vals []float32

	syncAfterWt := false // shows slow adapt errors earlier if true

	for pi := 0; pi < 4; pi++ {
		if initWts {
			testNet.SetRndSeed(42) // critical for ActAvg values
			testNet.InitWts(ctx)
		} else {
			testNet.NewState(ctx)
		}
		ctx.NewState(etime.Train)

		testNet.InitExt(ctx)
		for di := 0; di < nData; di++ {
			ppi := (pi + di) % 4
			inpat, err := inPats.SubSpaceTry([]int{ppi})
			if err != nil {
				t.Fatal(err)
			}
			_ = inpat
			inLay.ApplyExt(ctx, uint32(di), inpat)
			outLay.ApplyExt(ctx, uint32(di), inpat)
		}

		testNet.ApplyExts(ctx) // key now for GPU

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < 50; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				testNet.PlusPhaseStart(ctx)
			}
		}

		testNet.PlusPhase(ctx)
		testNet.DWt(ctx)

		if syncAfterWt {
			testNet.WtFmDWt(ctx)
			if slowAdapt {
				testNet.GPU.SyncSynCaFmGPU() // will be sent back and forth
				testNet.SlowAdapt(ctx)
			}
		}
		if gpu {
			testNet.GPU.SyncSynapsesFmGPU()
			testNet.GPU.SyncSynCaFmGPU()
		}

		for ni := 0; ni < 4; ni++ {
			for li := 1; li < 3; li++ {
				ly := testNet.Layers[li]
				for di := 0; di < nData; di++ {
					ppi := (pi + di) % 4
					key := fmt.Sprintf("pat: %d\tLayer: %s\tUnit: %d", ppi, ly.Nm, ni)
					doPrint := (printVals && pi < nPats && ni < nNeurs && li >= stLayer && li < edLayer)
					if doPrint {
						fmt.Println(key + fmt.Sprintf("  di: %d", di))
					}
					for nvi, vnm := range NeuronVarNames {
						ly.UnitVals(&vals, vnm, di)
						vkey := key + fmt.Sprintf("\t%s", vnm)
						valMap[vkey] = vals[ni]
						if doPrint {
							fmt.Printf("\t%-10s%7.4f", vnm, vals[ni])
							if (int(nvi)+1)%valsPerRow == 0 {
								fmt.Printf("\n")
							}
						}
					}
					lnm := fmt.Sprintf("%s: di: %d", ly.Nm, di)
					lpl := ly.Pool(0, uint32(di))
					StructVals(&lpl.Inhib, valMap, lnm)
					lval := ly.LayerVals(uint32(di))
					StructVals(&lval, valMap, lnm)
					if doPrint {
						fmt.Printf("\n")
					}
					for svi, snm := range SynapseVarNames {
						val := ly.RcvPrjns[0].SynValDi(snm, ni, ni, di)
						vkey := key + fmt.Sprintf("\t%s", snm)
						valMap[vkey] = val
						if doPrint {
							fmt.Printf("\t%-10s%7.4f", snm, val)
							if (int(svi)+1)%valsPerRow == 0 {
								fmt.Printf("\n")
							}
						}
					}
					if doPrint {
						fmt.Printf("\n")
					}
				}
			}
		}

		if !syncAfterWt {
			testNet.WtFmDWt(ctx)
			if slowAdapt {
				testNet.SlowAdapt(ctx)
			}
			if gpu {
				testNet.GPU.SyncSynapsesFmGPU()
				testNet.GPU.SyncSynCaFmGPU()
			}
		}

		pi += nData - 1
	}

	testNet.GPU.Destroy()
	return valMap
}

func TestDebugLearn(t *testing.T) {
	t.Skip("skipped in regular testing")
	NetDebugLearn(t, true, false, 2, true, false, false)
}

func TestNDataLearn(t *testing.T) {
	nd1Vals := NetDebugLearn(t, false, false, 1, true, false, false)
	nd4Vals := NetDebugLearn(t, false, false, 4, true, false, false)
	ReportValDiffs(t, nd1Vals, nd4Vals, "nData = 1", "nData = 4", []string{"DWt", "ActAvg", "DTrgAvg"})
}

func TestGPULearnDiff(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	// fmt.Printf("\n#############\nCPU\n")
	cpuVals := NetDebugLearn(t, false, false, 1, false, false, false)
	// fmt.Printf("\n#############\nGPU\n")
	gpuVals := NetDebugLearn(t, false, true, 1, false, false, false)
	ReportValDiffs(t, cpuVals, gpuVals, "CPU", "GPU", nil)
}

func TestGPUSubMeanLearn(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	// fmt.Printf("\n#############\nCPU\n")
	cpuVals := NetDebugLearn(t, false, false, 1, false, true, false)
	// fmt.Printf("\n#############\nGPU\n")
	gpuVals := NetDebugLearn(t, false, true, 1, false, true, false)
	ReportValDiffs(t, cpuVals, gpuVals, "CPU", "GPU", nil)
}

func TestGPUSlowAdaptLearn(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	// fmt.Printf("\n#############\nCPU\n")
	cpuVals := NetDebugLearn(t, false, false, 1, false, false, true)
	// fmt.Printf("\n#############\nGPU\n")
	gpuVals := NetDebugLearn(t, false, true, 1, false, false, true)
	ReportValDiffs(t, cpuVals, gpuVals, "CPU", "GPU", nil)
}

func TestInhibAct(t *testing.T) {
	// t.Skip("Skipping TestInhibAct for now until stable")
	inPats := newInPats()
	var InhibNet Network
	InhibNet.InitName(&InhibNet, "InhibNet")
	InhibNet.SetRndSeed(42) // critical for ActAvg values

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
					inLay.UnitVals(&inActs, "Act", 0)
					hidLay.UnitVals(&hidActs, "Act", 0)
					hidLay.UnitVals(&hidGes, "Ge", 0)
					hidLay.UnitVals(&hidGis, "Gi", 0)
					outLay.UnitVals(&outActs, "Act", 0)
					outLay.UnitVals(&outGes, "Ge", 0)
					outLay.UnitVals(&outGis, "Gi", 0)
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

			inLay.UnitVals(&inActs, "Act", 0)
			hidLay.UnitVals(&hidActs, "Act", 0)
			hidLay.UnitVals(&hidGes, "Ge", 0)
			hidLay.UnitVals(&hidGis, "Gi", 0)
			outLay.UnitVals(&outActs, "Act", 0)
			outLay.UnitVals(&outGes, "Ge", 0)
			outLay.UnitVals(&outGis, "Gi", 0)

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

func TestGlobalIdxs(t *testing.T) {
	ctx := NewContext()
	nData := uint32(5)
	ctx.PVLV.Drive.NActive = 4
	ctx.PVLV.Drive.NNegUSs = 3
	net := newTestNet(ctx, int(nData))
	val := float32(0)

	// fmt.Printf("MaxData: %d  NActive: %d  NNegUSs: %d  NetIdxs: GvVTAOff: %d  Stride: %d  USnegOff: %d  DriveOff: %d  DriveStride: %d\n", ctx.NetIdxs.MaxData, ctx.PVLV.Drive.NActive, ctx.PVLV.Drive.NNegUSs, ctx.NetIdxs.GvVTAOff, ctx.NetIdxs.GvVTAStride, ctx.NetIdxs.GvUSnegOff, ctx.NetIdxs.GvDriveOff, ctx.NetIdxs.GvDriveStride)

	for vv := GvRew; vv < GvVtaDA; vv++ {
		for di := uint32(0); di < nData; di++ {
			SetGlbV(ctx, di, vv, val)
			val += 1
		}
	}
	for vv := GvVtaDA; vv < GvUSneg; vv++ {
		for vt := GvVtaRaw; vt < GlobalVTATypeN; vt++ {
			for di := uint32(0); di < nData; di++ {
				SetGlbVTA(ctx, di, vt, vv, val)
				val += 1
			}
		}
	}
	for ui := uint32(0); ui < ctx.PVLV.Drive.NNegUSs; ui++ {
		for di := uint32(0); di < nData; di++ {
			SetGlbUSneg(ctx, di, ui, val)
			val += 1
		}
	}
	for vv := GvDrives; vv < GlobalVarsN; vv++ {
		for ui := uint32(0); ui < ctx.PVLV.Drive.NActive; ui++ {
			for di := uint32(0); di < nData; di++ {
				SetGlbDrvV(ctx, di, ui, vv, val)
				val += 1
			}
		}
	}
	if int(val) != len(net.Globals) {
		t.Errorf("Globals len: %d != val count: %d\n", len(net.Globals), int(val))
	}
	for i, gv := range net.Globals {
		if gv != float32(i) {
			t.Errorf("Global at index: %d != index val: %g\n", i, gv)
		}
	}
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
	pj.SWts.Init.Var = vr

	nt := NewNetwork("test")
	nt.SetRndSeed(1)

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWts.InitWtsSyn(ctx, &nt.Rand, sy, mean, spct)
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
	pj.SWts.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWts.InitWtsSyn(&nt.Rand, sy, mean, spct)
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
	pj.SWts.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWts.InitWtsSyn(&nt.Rand, sy, mean, spct)
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
	pj.SWts.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWts.InitWtsSyn(&nt.Rand, sy, mean, spct)
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
	pj.SWts.Init.Var = vr

	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)
	for i := 0; i < nsamp; i++ {
		pj.SWts.InitWtsSyn(&nt.Rand, sy, mean, spct)
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
	pj.SWts.Init.Var = vr
	pj.SWts.Adapt.SigGain = 1
	nlrn := 10
	// fmt.Printf("Wts Mean: %g\t Var: %g\t SPct: %g\n", mean, vr, spct)

	pj.SWts.InitWtsSyn(&nt.Rand, sy, mean, spct)
	// fmt.Printf("Wt: %g\t LWt: %g\t SWt: %g\n", sy.Wt, sy.LWt, sy.SWt)
	for i := 0; i < nlrn; i++ {
		sy.DWt = dwt
		pj.SWts.WtFmDWt(&sy.DWt, &sy.Wt, &sy.LWt, sy.SWt)
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
