// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/emer/emergent/etime"
)

func init() {
	// must lock main thread for gpu!
	runtime.LockOSThread()
}

func TestGPUAct(t *testing.T) {
	testNet := newTestNet()
	testNet.InitExt()
	inPats := newInPats()

	inLay := testNet.LayerByName("Input").(*Layer)
	hidLay := testNet.LayerByName("Hidden").(*Layer)
	outLay := testNet.LayerByName("Output").(*Layer)

	ctx := NewContext()

	testNet.GPUOnNoGUI(ctx)

	printCycs := true
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

	for pi := 0; pi < 1; pi++ {
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
				testNet.GPU.CopyNeuronsFromGPU(ctx, testNet)

				if printCycs {
					inLay.UnitVals(&inExts, "Ext")
					inLay.UnitVals(&inGes, "Ge")
					inLay.UnitVals(&inActs, "Act")
					hidLay.UnitVals(&hidActs, "Act")
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

			if true {

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
		}
		testNet.PlusPhase(ctx)

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}
