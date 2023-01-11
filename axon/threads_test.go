package axon

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/emer/emergent/emer"
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
	pats := generateRandomPatterns(100)
	// launch many goroutines to increase odds of finding race conditions
	netS, netP := buildIdenticalNetworks(t, pats, 16, 16, 16)

	fun := func(net *Network, ltime *Time) {
		net.Cycle(ltime)
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

// Make sure that training is deterministic, as long as the network is the same
// at the beginning of the test.
func TestDeterministicSingleThreadedTraining(t *testing.T) {
	pats := generateRandomPatterns(10)
	netA, netB := buildIdenticalNetworks(t, pats, 1, 1, 1)

	fun := func(net *Network, ltime *Time) {
		net.Cycle(ltime)
	}

	// by splitting the epochs into three parts for netB, we make sure that the
	// training is fully deterministic and not dependent on the `rand` package,
	// as we re-set the seed at the beginning of runFunEpochs.
	runFunEpochs(pats, netB, fun, 1)
	runFunEpochs(pats, netB, fun, 2)
	runFunEpochs(pats, netA, fun, 10)
	runFunEpochs(pats, netB, fun, 7)

	// compare the resulting networks
	assertNeuronsSynsEqual(t, netA, netB)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsSynsAreEqual(netA, netB))
	runFunEpochs(pats, netA, fun, 1)
	assert.False(t, neuronsSynsAreEqual(netA, netB))
}

func TestMultithreadedSendSpike(t *testing.T) {
	pats := generateRandomPatterns(10)
	netS, netP := buildIdenticalNetworks(t, pats, 1, 16, 1)

	fun := func(net *Network, ltime *Time) {
		net.Cycle(ltime)
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
	pats := generateRandomPatterns(10)
	netS, netP := buildIdenticalNetworks(t, pats, 16, 1, 1)

	fun := func(net *Network, ctime *Time) {
		net.Cycle(ctime)
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
	pats := generateRandomPatterns(10)
	netS, netP := buildIdenticalNetworks(t, pats, 16, 1, 16)

	fun := func(net *Network, ltime *Time) {
		net.Cycle(ltime)
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
		layerS := netS.Layers[li].(AxonLayer).AsAxon()
		layerP := netP.Layers[li].(AxonLayer).AsAxon()

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
		for pi := range *layerS.SendPrjns() {
			prjnS := (*layerS.SendPrjns())[pi].(*Prjn)
			prjnP := (*layerP.SendPrjns())[pi].(*Prjn)
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
		layerS := netS.Layers[li].(AxonLayer).AsAxon()
		layerP := netP.Layers[li].(AxonLayer).AsAxon()
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

func generateRandomPatterns(nPats int) *etable.Table {
	shape := []int{shape1D, shape1D}

	rand.Seed(42)

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
	netP := buildNet(t, shape, tNeuron, tSendSpike, tSynCa)

	assert.True(t, neuronsSynsAreEqual(netS, netP))

	return netS, netP
}

func buildNet(t *testing.T, shape []int, tNeuron, tSendSpike, tSynCa int) *Network {
	net := NewNetwork("MTTest")

	/*
	 * Input -> Hidden -> Hidden3 -> Output
	 *       -> Hidden2 -^
	 */
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden).(AxonLayer)
	hiddenLayer2 := net.AddLayer("Hidden2", shape, emer.Hidden).(AxonLayer)
	hiddenLayer3 := net.AddLayer("Hidden3", shape, emer.Hidden).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)
	net.ConnectLayers(inputLayer, hiddenLayer, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(inputLayer, hiddenLayer2, prjn.NewFull(), emer.Forward)
	net.BidirConnectLayers(hiddenLayer, hiddenLayer3, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer2, hiddenLayer3, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer3, outputLayer, prjn.NewFull())

	net.Defaults() // Initializes threading defaults, but we override below
	if err := net.Build(); err != nil {
		t.Fatal(err)
	}
	net.InitWts()

	if err := net.Threads.Set(tNeuron, tSendSpike, tSynCa); err != nil {
		t.Fatal(err)
	}
	return net
}

// runFunEpochs runs the given function for the given number of iterations over the
// dataset. The random seed is set once at the beginning of the function.
func runFunEpochs(pats *etable.Table, net *Network, fun func(*Network, *Time), epochs int) {
	rand.Seed(42)
	nCycles := 150

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)
	inputLayer := net.LayerByName("Input").(*Layer)
	outputLayer := net.LayerByName("Output").(*Layer)
	ltime := NewTime()
	for epoch := 0; epoch < epochs; epoch++ {
		for pi := 0; pi < pats.NumRows(); pi++ {
			input := inPats.SubSpace([]int{pi})
			output := outPats.SubSpace([]int{pi})

			inputLayer.ApplyExt(input)
			outputLayer.ApplyExt(output)

			net.NewState()
			ltime.NewState("Train")
			for cycle := 0; cycle < nCycles; cycle++ {
				fun(net, ltime)
			}
		}
	}
}
