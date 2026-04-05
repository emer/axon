// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"testing"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/table"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/paths"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	shape1D = 2
)

// TestMultithreading
func TestMultithreading(t *testing.T) {
	pats := generateRandomPatterns(100, 42)
	// launch many goroutines to increase odds of finding race conditions
	netS, netP := buildIdenticalNetworks(t, pats, 16)

	assertNeuronsSynsEqual(t, netS, netP)

	fun := func(net *Network) {
		net.Cycle(true)
	}

	runFunEpochs(pats, netS, fun, 2)
	runFunEpochs(pats, netP, fun, 2)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netS, netP)
	// sanity check
	assert.True(t, neuronsSynsAreEqual(netS, netP))

	// runFunEpochs(pats, netS, fun, 1)
	// assert.False(t, neuronsSynsAreEqual(netS, netP))
}

func TestCollectAndSetDWts(t *testing.T) {
	// t.SkipNow()
	patsA := generateRandomPatterns(1, 1337)
	patsB := generateRandomPatterns(1, 42)
	shape := []int{shape1D, shape1D}

	netA := buildNet(t, 1, shape...)
	netA.Context().SlowInterval = 10000
	netB := buildNet(t, 1, shape...)
	netB.Context().SlowInterval = 10000

	runCycle := func(net *Network, pats *table.Table) {
		inPats := pats.Column("Input").SubSpace(0)
		outPats := pats.Column("Output").SubSpace(0)
		inputLayer := net.LayerByName("Input")
		outputLayer := net.LayerByName("Output")
		// we train on a single pattern
		input := inPats.SubSpace(0)
		output := outPats.SubSpace(0)

		net.ThetaCycleStart(etime.Train, false)
		net.MinusPhaseStart()
		net.InitExt()
		inputLayer.ApplyExt(0, input)
		outputLayer.ApplyExt(0, output)
		net.ApplyExts()

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < 50; cyc++ {
				net.Cycle(true)
			}
			if qtr == 2 {
				net.MinusPhaseEnd()
				net.PlusPhaseStart()
			}
		}
		net.PlusPhaseEnd()
		// for _, ly := range net.Layers {
		// 	fmt.Printf("ly: %s  actm: %g  actp: %g\n", ly.Name, ly.Pool(0, 0).AvgMax.CaD.Minus.Max, ly.Pool(0, 0).AvgMax.CaD.Plus.Max)
		// }
		net.DWt()
	}

	runCycle(netA, patsA)
	runCycle(netB, patsB)

	// No DWt applied, hence networks still equal
	assert.Equal(t, netA.WeightsHash(), netB.WeightsHash())
	var dwts []float32

	// for debugging
	netA.CollectDWts(&dwts) // important to collect DWt before applying it
	netB.SetDWts(dwts, 1)
	// if CompareWtsAll(netA, netB) {
	// 	t.Errorf("CollectDWts -> SetDWts failed\n")
	// }
	// if CompareNeursAll(netA, netB) {
	// 	t.Errorf("CollectDWts -> SetDWts failed\n")
	// }

	netA.WtFromDWt()
	assert.False(t, netA.WeightsHash() == netB.WeightsHash())

	netB.WtFromDWt()
	// if CompareWtsAll(netA, netB) {
	// 	t.Errorf("WtFromDWt failed\n")
	// }
	// todo: fixme:
	// assert.True(t, netA.WeightsHash() == netB.WeightsHash())

	netA.SlowAdapt()
	netB.SlowAdapt()
	// if CompareWtsAll(netA, netB) {
	// 	t.Errorf("SlowAdapt failed\n")
	// }

	// And again (as a sanity check), but without syncing DWt -> Models should diverge
	runCycle(netA, patsA)
	runCycle(netB, patsB)
	netA.WtFromDWt()
	netA.SlowAdapt()
	assert.False(t, netA.WeightsHash() == netB.WeightsHash())
	// netB is trained on a different pattern, hence different DWt, hence different Wt
	netB.WtFromDWt()
	netB.SlowAdapt()
	assert.False(t, netA.WeightsHash() == netB.WeightsHash())
}

// returns true if a diff
func CompareWtsAll(netA, netB *Network) bool {
	fmt.Printf("SWt:\n")
	diff := CompareWts(netA, netB, SWt)
	fmt.Printf("DWt:\n")
	diff = CompareWts(netA, netB, DWt) || diff
	fmt.Printf("DSWt:\n")
	diff = CompareWts(netA, netB, DSWt) || diff
	fmt.Printf("LWt:\n")
	diff = CompareWts(netA, netB, LWt) || diff
	fmt.Printf("Wt:\n")
	diff = CompareWts(netA, netB, Wt) || diff
	return diff
}

func CompareWts(netA, netB *Network, synvar SynapseVars) bool {
	var valsA, valsB []float32
	netA.SynsSlice(&valsA, synvar)
	netB.SynsSlice(&valsB, synvar)
	diff := false
	for i := range valsA {
		if valsA[i] != valsB[i] {
			fmt.Printf("%d  %g != %g\n", i, valsA[i], valsB[i])
			diff = true
		}
	}
	return diff
}

// returns true if a diff
func CompareNeursAll(netA, netB *Network) bool {
	fmt.Printf("ActAvg:\n")
	diff := CompareNeurs(netA, netB, "ActAvg")
	fmt.Printf("ActAvg:\n")
	diff = CompareNeurs(netA, netB, "DTrgAvg") || diff
	return diff
}

func CompareNeurs(netA, netB *Network, nrnVar string) bool {
	var valsA, valsB []float32
	netA.NeuronsSlice(&valsA, nrnVar, 0)
	netB.NeuronsSlice(&valsB, nrnVar, 0)
	diff := false
	for i := range valsA {
		if valsA[i] != valsB[i] {
			fmt.Printf("%d  %g != %g\n", i, valsA[i], valsB[i])
			diff = true
		}
	}
	return diff
}

// Make sure that training is deterministic, as long as the network is the same
// at the beginning of the test.
func TestDeterministicSingleThreadedTraining(t *testing.T) {
	pats := generateRandomPatterns(10, 42)
	netA, netB := buildIdenticalNetworks(t, pats, 1)

	fun := func(net *Network) {
		net.Cycle(true)
		net.WtFromDWt()
	}

	// by splitting the epochs into three parts for netB, we make sure that the
	// training is fully deterministic and not dependent on the `rand` package,
	// as we re-set the seed at the beginning of runFunEpochs.
	runFunEpochs(pats, netB, fun, 1)
	runFunEpochs(pats, netB, fun, 2)
	runFunEpochs(pats, netA, fun, 5)
	runFunEpochs(pats, netB, fun, 2)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netA, netB)
	// assert.Equal(t, netA.WeightsHash(), netB.WeightsHash())

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netA, netB))
	runFunEpochs(pats, netA, fun, 1)
	// assert.False(t, neuronsSynsAreEqual(netA, netB))
	assert.False(t, netA.WeightsHash() == netB.WeightsHash())
}

// assert that all Neuron fields and all Synapse fields are bit-equal between
// the two networks
func assertNeuronsSynsEqual(t *testing.T, netS *Network, netP *Network) {
	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]

		// check Neuron fields
		for lni := uint32(0); lni < layerS.NNeurons; lni++ {
			for _, fn := range NeuronVarNames {
				if fn == "LearnNow" {
					continue
				}
				vidx, _ := layerS.UnitVarIndex(fn)
				vS := layerS.UnitValue1D(vidx, int(lni), 0)
				vP := layerP.UnitValue1D(vidx, int(lni), 0)
				if math32.IsNaN(vS) && math32.IsNaN(vP) {
					continue
				}
				require.Equal(t, vS, vP,
					"Neuron %d, field %s, single thread: %f, multi thread: %f",
					lni, fn, vS, vP)
			}
		}
	}

	// check Synapse fields after all neurons are validated -- neuron diffs primary
	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]

		for pi := range layerS.SendPaths {
			pathS := layerS.SendPaths[pi]
			pathP := layerP.SendPaths[pi]
			for sni := uint32(0); sni < pathS.NSyns; sni++ {
				for fi := 0; fi < int(SynapseVarsN); fi++ {
					synS := pathS.SynValue1D(fi, int(sni))
					synP := pathP.SynValue1D(fi, int(sni))
					require.Equal(t, synS, synP,
						"Synapse %d, field %s, single thread: %f, multi thread: %f",
						sni, SynapseVars(fi).String(), synS, synP)
				}
			}
		}
	}
}

// This implements the same logic as assertNeuronsEqual, but returns a bool
// to allow writing tests that are expected to fail (eg assert that two networks are not equal)
func neuronsSynsAreEqual(netS *Network, netP *Network) bool {
	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]

		// check Neuron fields
		for lni := uint32(0); lni < layerS.NNeurons; lni++ {
			for _, fn := range NeuronVarNames {
				if fn == "LearnNow" {
					continue
				}
				vidx, _ := layerS.UnitVarIndex(fn)
				vS := layerS.UnitValue1D(vidx, int(lni), 0)
				vP := layerP.UnitValue1D(vidx, int(lni), 0)
				if math32.IsNaN(vS) && math32.IsNaN(vP) {
					continue
				}
				if vS != vP {
					return false
				}
			}
		}
	}

	// check Synapse fields after all neurons are validated -- neuron diffs primary
	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]

		for pi := range layerS.SendPaths {
			pathS := layerS.SendPaths[pi]
			pathP := layerP.SendPaths[pi]
			for sni := uint32(0); sni < pathS.NSyns; sni++ {
				for fi := 0; fi < int(SynapseVarsN); fi++ {
					synS := pathS.SynValue1D(fi, int(sni))
					synP := pathP.SynValue1D(fi, int(sni))
					if synS != synP {
						return false
					}
				}
			}
		}
	}
	return true
}

func generateRandomPatterns(nPats int, seed int64) *table.Table {
	shape := []int{shape1D, shape1D}
	pats := table.New()
	pats.AddStringColumn("Name")
	pats.AddFloat32Column("Input", shape...)
	pats.AddFloat32Column("Output", shape...)
	pats.SetNumRows(nPats)
	numOn := max((shape[0]*shape[1])/4, 1) // ensure min at least 1
	patterns.NewRand(seed)
	patterns.PermutedBinaryMinDiff(pats.Columns.Values[1], numOn, 1, 0, numOn/2)
	patterns.PermutedBinaryMinDiff(pats.Columns.Values[2], numOn, 1, 0, numOn/2)
	// fmt.Printf("%v\n", pats.Columns[1].(*tensor.Float32).Values)
	return pats
}

// buildIdenticalNetworks builds two identical nets, one single-threaded and one
// multi-threaded (parallel). They are seeded with the same RNG, so they are identical.
// Returns two networks: (sequential, parallel)
func buildIdenticalNetworks(t *testing.T, pats *table.Table, nthrs int) (*Network, *Network) {
	shape := []int{shape1D, shape1D}

	// Create both networks. Ideally we'd create one network, run for a few cycles
	// to get more interesting state, and then duplicate it. But currently we have
	// no good way of storing and restoring the full network state, so we just
	// initialize two equal networks from scratch

	// single-threaded network
	netS := buildNet(t, 1, shape...)
	// multi-threaded network
	netM := buildNet(t, nthrs, shape...)

	assertNeuronsSynsEqual(t, netS, netM)

	return netS, netM
}

func buildNet(t *testing.T, nthrs int, shape ...int) *Network {
	net := NewNetwork("MTTest")
	/*
	 * Input -> Hidden -> Hidden3 -> Output
	 *       -> Hidden2 -^
	 */
	net.Rand.Seed(1337)
	inputLayer := net.AddLayer("Input", InputLayer, shape...)
	hiddenLayer := net.AddLayer("Hidden", SuperLayer, shape...)
	hiddenLayer2 := net.AddLayer("Hidden2", SuperLayer, shape...)
	hiddenLayer3 := net.AddLayer("Hidden3", SuperLayer, shape...)
	outputLayer := net.AddLayer("Output", TargetLayer, shape...)
	net.ConnectLayers(inputLayer, hiddenLayer, paths.NewFull(), ForwardPath)
	net.ConnectLayers(inputLayer, hiddenLayer2, paths.NewFull(), ForwardPath)
	net.BidirConnectLayers(hiddenLayer, hiddenLayer3, paths.NewFull())
	net.BidirConnectLayers(hiddenLayer2, hiddenLayer3, paths.NewFull())
	net.BidirConnectLayers(hiddenLayer3, outputLayer, paths.NewFull())

	if err := net.Build(); err != nil {
		t.Fatal(err)
	}
	net.Defaults() // Initializes threading defaults, but we override below
	net.InitWeights()
	net.SetNThreads(nthrs)
	return net
}

// runFunEpochs runs the given function for the given number of iterations over the
// dataset. The random seed is set once at the beginning of the function.
func runFunEpochs(pats *table.Table, net *Network, fun func(*Network), epochs int) {
	nCycles := 150

	inPats := pats.Column("Input")
	outPats := pats.Column("Output")
	inputLayer := net.LayerByName("Input")
	outputLayer := net.LayerByName("Output")
	for epoch := 0; epoch < epochs; epoch++ {
		for pi := range pats.NumRows() {
			input := inPats.SubSpace(pi)
			output := outPats.SubSpace(pi)

			net.ThetaCycleStart(etime.Train, false)
			net.MinusPhaseStart()
			net.InitExt()
			inputLayer.ApplyExt(0, input)
			outputLayer.ApplyExt(0, output)
			net.ApplyExts()
			for range nCycles {
				fun(net)
			}
		}
	}
}
