package axon

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	shape1D = 8
)

// TestMultithreadingCycleFun tests the whole net.Cycle() function, turning on
// multithreading for all functions.
func TestMultithreadingCycleFun(t *testing.T) {
	pats := generateRandomPatterns(100, 42)
	// launch many goroutines to increase odds of finding race conditions
	netS, netP := buildIdenticalNetworks(t, pats, 16, 16, 16)

	fun := func(net *Network, ctx *Context) {
		net.Cycle(ctx)
	}

	runFunEpochs(pats, netS, fun, 2)
	runFunEpochs(pats, netP, fun, 2)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netS, netP)
	// sanity check
	assert.True(t, neuronsSynsAreEqual(netS, netP))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netS, netP))

	// check that the number of threads used is correct
	assert.Equal(t, 1, netS.Threads.Neurons)
	assert.Equal(t, 1, netS.Threads.SendSpike)
	assert.Equal(t, 1, netS.Threads.SynCa)
	assert.Equal(t, 16, netP.Threads.Neurons)
	assert.Equal(t, 16, netP.Threads.SendSpike)
	assert.Equal(t, 16, netP.Threads.SynCa)
}

func TestCollectAndSetDWts(t *testing.T) {
	// TODO: This should be moved out of threads_test.go, but all the useful
	// helper functions are here.

	patsA := generateRandomPatterns(1, 1337)
	patsB := generateRandomPatterns(1, 42)
	shape := []int{shape1D, shape1D}

	rand.Seed(1337)
	netA := buildNet(t, shape, 1, 1, 1)
	netA.SlowInterval = 1
	rand.Seed(1337)
	netB := buildNet(t, shape, 1, 1, 1)
	netB.SlowInterval = 1

	runCycle := func(net *Network, ctx *Context, pats *etable.Table) {
		inPats := pats.ColByName("Input").(*etensor.Float32).SubSpace([]int{0})
		outPats := pats.ColByName("Output").(*etensor.Float32).SubSpace([]int{0})
		inputLayer := net.AxonLayerByName("Input")
		outputLayer := net.AxonLayerByName("Output")
		// we train on a single pattern
		input := inPats.SubSpace([]int{0})
		output := outPats.SubSpace([]int{0})

		inputLayer.ApplyExt(input)
		outputLayer.ApplyExt(output)

		net.NewState(ctx)
		ctx.NewState(etime.Train)
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < 25; cyc++ {
				net.Cycle(ctx)
				ctx.CycleInc()
			}
			if qtr == 2 {
				net.MinusPhase(ctx)
				ctx.NewPhase(true)
				net.PlusPhaseStart(ctx)
			}
		}
		net.PlusPhase(ctx)
		net.DWt(ctx)
	}

	rand.Seed(42)
	ctxA := NewContext()
	runCycle(netA, ctxA, patsA)

	rand.Seed(42)
	ctxB := NewContext()
	runCycle(netB, ctxB, patsB)

	// No DWt applied, hence networks still equal
	assert.Equal(t, netA.WtsHash(), netB.WtsHash())

	var dwts []float32
	netA.CollectDWts(&dwts) // important to collect DWt before applying it
	netA.WtFmDWt(ctxA)
	netA.SlowAdapt(ctxA)
	assert.False(t, netA.WtsHash() == netB.WtsHash())

	netB.SetDWts(dwts, 1)
	netB.WtFmDWt(ctxB)
	netB.SlowAdapt(ctxB)
	assert.True(t, netA.WtsHash() == netB.WtsHash())

	// And again (as a sanity check), but without syncing DWt -> Models should diverge
	runCycle(netA, ctxA, patsA)
	runCycle(netB, ctxB, patsB)
	netA.WtFmDWt(ctxA)
	netA.SlowAdapt(ctxA)
	assert.False(t, netA.WtsHash() == netB.WtsHash())
	// netB is trained on a different pattern, hence different DWt, hence different Wt
	netB.WtFmDWt(ctxB)
	netB.SlowAdapt(ctxB)
	assert.False(t, netA.WtsHash() == netB.WtsHash())
}

// Make sure that training is deterministic, as long as the network is the same
// at the beginning of the test.
func TestDeterministicSingleThreadedTraining(t *testing.T) {
	pats := generateRandomPatterns(10, 42)
	netA, netB := buildIdenticalNetworks(t, pats, 1, 1, 1)

	fun := func(net *Network, ctx *Context) {
		net.Cycle(ctx)
		net.WtFmDWt(ctx)
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
	assert.Equal(t, netA.WtsHash(), netB.WtsHash())

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netA, netB))
	runFunEpochs(pats, netA, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netA, netB))
	assert.False(t, netA.WtsHash() == netB.WtsHash())
}

func TestMultithreadedSendSpike(t *testing.T) {
	pats := generateRandomPatterns(10, 42)
	netS, netP := buildIdenticalNetworks(t, pats, 1, 16, 1)

	fun := func(net *Network, ctx *Context) {
		net.Cycle(ctx)
	}

	runFunEpochs(pats, netP, fun, 2)
	runFunEpochs(pats, netS, fun, 2)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netS, netP)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netS, netP))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netS, netP))
}

func TestMultithreadedNeuronFun(t *testing.T) {
	pats := generateRandomPatterns(10, 42)
	netS, netP := buildIdenticalNetworks(t, pats, 16, 1, 1)

	fun := func(net *Network, ctx *Context) {
		net.Cycle(ctx)
	}

	runFunEpochs(pats, netP, fun, 3)
	runFunEpochs(pats, netS, fun, 3)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netS, netP)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netS, netP))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netS, netP))
}

func TestMultithreadedSynCa(t *testing.T) {
	pats := generateRandomPatterns(10, 42)
	netS, netP := buildIdenticalNetworks(t, pats, 16, 1, 16)

	fun := func(net *Network, ctx *Context) {
		net.Cycle(ctx)
	}

	runFunEpochs(pats, netP, fun, 3)
	runFunEpochs(pats, netS, fun, 3)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netS, netP)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netS, netP))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netS, netP))
}

// assert that all Neuron fields and all Synapse fields are equal between
// the two networks
func assertNeuronsSynsEqual(t *testing.T, netS *Network, netP *Network) {
	maxDiff := struct {
		diff      float64
		field     string
		neuronIdx int
		valS      float64
		valP      float64
	}{}

	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]

		// check Neuron fields
		for ni := range layerS.Neurons {
			nrnS := reflect.ValueOf(layerS.Neurons[ni])
			nrnP := reflect.ValueOf(layerP.Neurons[ni])
			for fi := 0; fi < nrnS.NumField(); fi++ {
				fieldS := nrnS.Field(fi)
				fieldP := nrnP.Field(fi)
				if fieldS.Kind() == reflect.Float32 {
					// Notice: We check for full bit-equality here, because there is no reason
					// why the trivially parallelizable functions should produce different results
					assert.Equal(t, fieldS.Float(), fieldP.Float(),
						"Neuron %d, field %s, single thread: %f, multi thread: %f",
						ni, nrnS.Type().Field(fi).Name, fieldS.Float(), fieldP.Float())
					if math.Abs(fieldS.Float()-fieldP.Float()) > maxDiff.diff {
						maxDiff.diff = math.Abs(fieldS.Float() - fieldP.Float())
						maxDiff.field = nrnS.Type().Field(fi).Name
						maxDiff.neuronIdx = ni
						maxDiff.valS = fieldS.Float()
						maxDiff.valP = fieldP.Float()
					}
				} else if fieldS.Kind() == reflect.Int32 {
					assert.Equal(t, fieldS.Int(), fieldP.Int(),
						"Neuron %d, field %s, single thread: %d, multi thread: %d",
						ni, nrnS.Type().Field(fi).Name, fieldS.Int(), fieldP.Int())
				}
			}
		}

		// check Synapse fields
		for pi := range layerS.SndPrjns {
			prjnS := layerS.SndPrjns[pi]
			prjnP := layerP.SndPrjns[pi]
			for si := range prjnS.Syns {
				synS := reflect.ValueOf(prjnS.Syns[si])
				synP := reflect.ValueOf(prjnP.Syns[si])
				for fi := 0; fi < synS.NumField(); fi++ {
					fieldS := synS.Field(fi)
					fieldP := synP.Field(fi)
					if fieldS.Kind() == reflect.Float32 {
						assert.Equal(t, fieldS.Float(), fieldP.Float(),
							"Synapse %d, field %s, single thread: %f, multi thread: %f",
							si, synS.Type().Field(fi).Name, fieldS.Float(), fieldP.Float())
					} else if fieldS.Kind() == reflect.Int32 {
						assert.Equal(t, fieldS.Int(), fieldP.Int(),
							"Synapse %d, field %s, single thread: %d, multi thread: %d",
							si, synS.Type().Field(fi).Name, fieldS.Int(), fieldP.Int())
					}
				}
			}
		}
	}

	require.Equalf(t, 0.0, maxDiff.diff,
		"Max difference (floats only): %f, field %s, neuron %d, single thread: %f, multi thread: %f",
		maxDiff.diff, maxDiff.field, maxDiff.neuronIdx, maxDiff.valS, maxDiff.valP)
}

// This implements the same logic as assertNeuronsEqual, but returns a bool
// to allow writing tests that are expected to fail (eg assert that two networks are not equal)
func neuronsSynsAreEqual(netS *Network, netP *Network) bool {
	for li := range netS.Layers {
		layerS := netS.Layers[li]
		layerP := netP.Layers[li]
		for ni := range layerS.Neurons {
			nrnS := reflect.ValueOf(layerS.Neurons[ni])
			nrnP := reflect.ValueOf(layerP.Neurons[ni])
			for fi := 0; fi < nrnS.NumField(); fi++ {
				fieldS := nrnS.Field(fi)
				fieldP := nrnP.Field(fi)
				if fieldS.Kind() == reflect.Float32 {
					if fieldS.Float() != fieldP.Float() {
						return false
					}
				} else if fieldS.Kind() == reflect.Int32 {
					if fieldS.Int() != fieldP.Int() {
						return false
					}
				}
			}
		}
	}
	return true
}

func generateRandomPatterns(nPats int, seed int64) *etable.Table {
	shape := []int{shape1D, shape1D}

	rand.Seed(seed)

	pats := &etable.Table{}
	pats.SetFromSchema(etable.Schema{
		{Name: "Name", Type: etensor.STRING, CellShape: nil, DimNames: nil},
		{Name: "Input", Type: etensor.FLOAT32, CellShape: shape, DimNames: []string{"Y", "X"}},
		{Name: "Output", Type: etensor.FLOAT32, CellShape: shape, DimNames: []string{"Y", "X"}},
	}, nPats)
	numOn := shape[0] * shape[1] / 8
	patgen.PermutedBinaryRows(pats.Cols[1], numOn, 1, 0)
	patgen.PermutedBinaryRows(pats.Cols[2], numOn, 1, 0)
	return pats
}

// buildIdenticalNetworks builds two identical nets, one single-threaded and one
// multi-threaded (parallel). They are seeded with the same RNG, so they are identical.
// Returns two networks: (sequential, parallel)
func buildIdenticalNetworks(t *testing.T, pats *etable.Table, tNeuron, tSendSpike, tSynCa int) (*Network, *Network) {
	shape := []int{shape1D, shape1D}

	// Create both networks. Ideally we'd create one network, run for a few cycles
	// to get more interesting state, and then duplicate it. But currently we have
	// no good way of storing and restoring the full network state, so we just
	// initialize two equal networks from scratch

	// single-threaded network
	rand.Seed(1337)
	netS := buildNet(t, shape, 1, 1, 1)
	// multi-threaded network
	rand.Seed(1337)
	netM := buildNet(t, shape, tNeuron, tSendSpike, tSynCa)

	// The below code doesn't work, because we have no clean way of storing and restoring
	// the full state of a network.
	//
	// run for a few cycles to get more interesting state on the single-threaded net
	// rand.Seed(1337)
	// inPats := pats.ColByName("Input").(*etensor.Float32)
	// outPats := pats.ColByName("Output").(*etensor.Float32)
	// inputLayer := netS.AxonLayerByName("Input").(*Layer)
	// outputLayer := netS.AxonLayerByName("Output").(*Layer)
	// input := inPats.SubSpace([]int{0})
	// output := outPats.SubSpace([]int{0})
	// inputLayer.ApplyExt(input)
	// outputLayer.ApplyExt(output)
	// netS.NewState()
	// ctx := NewContext()
	// ctx.NewState("train")
	// for i := 0; i < 150; i++ {
	// 	netS.Cycle(ctx)
	// }

	// // sync the weights
	// filename := t.TempDir() + "/netS.json"
	// // write Synapse weights to file
	// fh, err := os.Create(filename)
	// require.NoError(t, err)
	// bw := bufio.NewWriter(fh)
	// require.NoError(t, netS.WriteWtsJSON(bw))
	// require.NoError(t, bw.Flush())
	// require.NoError(t, fh.Close())
	// // read Synapse weights from file
	// fh, err = os.Open(filename)
	// require.NoError(t, err)
	// br := bufio.NewReader(fh)
	// require.NoError(t, netM.ReadWtsJSON(br))
	// require.NoError(t, fh.Close())
	// // sync Neuron weights as well (TODO: we need a better way to do this)
	// copy(netM.Neurons, netS.Neurons)

	// todo: this should be uncommented!
	// assert.True(t, assertneuronsAreEqual(netS, netM))

	return netS, netM
}

func buildNet(t *testing.T, shape []int, tNeuron, tSendSpike, tSynCa int) *Network {
	net := NewNetwork("MTTest")

	/*
	 * Input -> Hidden -> Hidden3 -> Output
	 *       -> Hidden2 -^
	 */
	inputLayer := net.AddLayer("Input", shape, InputLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, SuperLayer)
	hiddenLayer2 := net.AddLayer("Hidden2", shape, SuperLayer)
	hiddenLayer3 := net.AddLayer("Hidden3", shape, SuperLayer)
	outputLayer := net.AddLayer("Output", shape, TargetLayer)
	net.ConnectLayers(inputLayer, hiddenLayer, prjn.NewFull(), ForwardPrjn)
	net.ConnectLayers(inputLayer, hiddenLayer2, prjn.NewFull(), ForwardPrjn)
	net.BidirConnectLayers(hiddenLayer, hiddenLayer3, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer2, hiddenLayer3, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer3, outputLayer, prjn.NewFull())

	if err := net.Build(); err != nil {
		t.Fatal(err)
	}
	net.Defaults() // Initializes threading defaults, but we override below
	net.InitWts()

	if err := net.Threads.Set(tNeuron, tSendSpike, tSynCa); err != nil {
		t.Fatal(err)
	}
	return net
}

// runFunEpochs runs the given function for the given number of iterations over the
// dataset. The random seed is set once at the beginning of the function.
func runFunEpochs(pats *etable.Table, net *Network, fun func(*Network, *Context), epochs int) {
	rand.Seed(42)
	nCycles := 150

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)
	inputLayer := net.AxonLayerByName("Input")
	outputLayer := net.AxonLayerByName("Output")
	ctx := NewContext()
	for epoch := 0; epoch < epochs; epoch++ {
		for pi := 0; pi < pats.NumRows(); pi++ {
			input := inPats.SubSpace([]int{pi})
			output := outPats.SubSpace([]int{pi})

			inputLayer.ApplyExt(input)
			outputLayer.ApplyExt(output)

			net.NewState(ctx)
			ctx.NewState(etime.Train)
			for cycle := 0; cycle < nCycles; cycle++ {
				fun(net, ctx)
			}
		}
	}
}
