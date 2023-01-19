package axon

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/emer/emergent/emer"
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
	pats := generateRandomPatterns(100)
	netS, netM := buildIdenticalNetworks(t, pats, 16, 16, 16)

	fun := func(net *Network, ctxt *Context) {
		net.Cycle(ctxt)
	}

	runFunEpochs(pats, netS, fun, 2)
	runFunEpochs(pats, netM, fun, 2)

	// compare the resulting networks
	assertNeuronsEqual(t, netS, netM)
	// sanity check
	assert.True(t, neuronsAreEqual(netS, netM))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsAreEqual(netS, netM))
}

// Make sure that training is deterministic, as long as the network is the same
// at the beginning of the test.
func TestDeterministicSingleThreadedTraining(t *testing.T) {
	pats := generateRandomPatterns(10)
	netA, netB := buildIdenticalNetworks(t, pats, 1, 1, 1)

	fun := func(net *Network, ctxt *Context) {
		net.Cycle(ctxt)
	}

	// by splitting the epochs into three parts for netB, we make sure that the
	// training is fully deterministic and not dependent on the `rand` package,
	// as we re-set the seed at the beginning of runFunEpochs.
	runFunEpochs(pats, netB, fun, 1)
	runFunEpochs(pats, netB, fun, 2)
	runFunEpochs(pats, netA, fun, 10)
	runFunEpochs(pats, netB, fun, 7)

	// compare the resulting networks
	assertNeuronsEqual(t, netA, netB)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsAreEqual(netA, netB))
	runFunEpochs(pats, netA, fun, 1)
	assert.False(t, neuronsAreEqual(netA, netB))
}

func TestMultithreadedSendSpike(t *testing.T) {
	pats := generateRandomPatterns(10)
	// run an absurd amount of threads, to make sure we encounter race conditions
	// goroutines are cheap, they barely cost any memory
	netS, netM := buildIdenticalNetworks(t, pats, 1, 16, 1)

	fun := func(net *Network, ctxt *Context) {
		net.Cycle(ctxt)
	}

	runFunEpochs(pats, netM, fun, 2)
	runFunEpochs(pats, netS, fun, 2)

	// compare the resulting networks
	assertNeuronsEqual(t, netS, netM)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsAreEqual(netS, netM))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsAreEqual(netS, netM))
}

func TestMultithreadedNeuronFun(t *testing.T) {
	pats := generateRandomPatterns(10)
	netS, netM := buildIdenticalNetworks(t, pats, 16, 1, 1)

	fun := func(net *Network, ctxt *Context) {
		net.Cycle(ctxt)
	}

	runFunEpochs(pats, netM, fun, 3)
	runFunEpochs(pats, netS, fun, 3)

	// compare the resulting networks
	assertNeuronsEqual(t, netS, netM)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsAreEqual(netS, netM))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsAreEqual(netS, netM))
}

func TestMultithreadedSynCa(t *testing.T) {
	pats := generateRandomPatterns(10)
	netS, netM := buildIdenticalNetworks(t, pats, 16, 1, 16)

	fun := func(net *Network, ctxt *Context) {
		net.Cycle(ctxt)
	}

	runFunEpochs(pats, netM, fun, 3)
	runFunEpochs(pats, netS, fun, 3)

	// compare the resulting networks
	assertNeuronsEqual(t, netS, netM)

	// sanity check, to make sure we're not accidentally sharing pointers etc.
	assert.True(t, neuronsAreEqual(netS, netM))
	runFunEpochs(pats, netS, fun, 1)
	assert.False(t, neuronsAreEqual(netS, netM))
}

// assert that neuron fields are equal. We could also check for Synapse equality,
// but over time different Synapse weights will lead to different Neuron fields.
func assertNeuronsEqual(t *testing.T, netS *Network, netM *Network) {
	maxDiff := struct {
		diff      float64
		field     string
		neuronIdx int
		valShould float64
		valIs     float64
	}{}

	for li := range netS.Layers {
		layerS := netS.Layers[li].(AxonLayer).AsAxon()
		layerM := netM.Layers[li].(AxonLayer).AsAxon()

		// check Neuron fields
		for ni := range layerS.Neurons {
			nrnS := reflect.ValueOf(layerS.Neurons[ni])
			nrnM := reflect.ValueOf(layerM.Neurons[ni])
			for fi := 0; fi < nrnS.NumField(); fi++ {
				fieldS := nrnS.Field(fi)
				fieldM := nrnM.Field(fi)
				if fieldS.Kind() == reflect.Float32 {
					// Notice: We check for full bit-equality here, because there is no reason
					// why the trivially parallelizable functions should produce different results
					assert.Equal(t, fieldS.Float(), fieldM.Float(),
						"Neuron %d, field %s, single thread: %f, multi thread: %f",
						ni, nrnS.Type().Field(fi).Name, fieldS.Float(), fieldM.Float())
					if math.Abs(fieldS.Float()-fieldM.Float()) > maxDiff.diff {
						maxDiff.diff = math.Abs(fieldS.Float() - fieldM.Float())
						maxDiff.field = nrnS.Type().Field(fi).Name
						maxDiff.neuronIdx = ni
						maxDiff.valShould = fieldS.Float()
						maxDiff.valIs = fieldM.Float()
					}
				} else if fieldS.Kind() == reflect.Int32 {
					assert.Equal(t, fieldS.Int(), fieldM.Int(),
						"Neuron %d, field %s, single thread: %d, multi thread: %d",
						ni, nrnS.Type().Field(fi).Name, fieldS.Int(), fieldM.Int())
				}
			}
		}

		// check Synapse fields
		for pi := range *layerS.SendPrjns() {
			prjnS := (*layerS.SendPrjns())[pi].(*Prjn)
			prjnM := (*layerM.SendPrjns())[pi].(*Prjn)
			for si := range prjnS.Syns {
				synS := reflect.ValueOf(prjnS.Syns[si])
				synM := reflect.ValueOf(prjnM.Syns[si])
				for fi := 0; fi < synS.NumField(); fi++ {
					fieldS := synS.Field(fi)
					fieldM := synM.Field(fi)
					if fieldS.Kind() == reflect.Float32 {
						assert.Equal(t, fieldS.Float(), fieldM.Float(),
							"Synapse %d, field %s, single thread: %f, multi thread: %f",
							si, synS.Type().Field(fi).Name, fieldS.Float(), fieldM.Float())
					} else if fieldS.Kind() == reflect.Int32 {
						assert.Equal(t, fieldS.Int(), fieldM.Int(),
							"Synapse %d, field %s, single thread: %d, multi thread: %d",
							si, synS.Type().Field(fi).Name, fieldS.Int(), fieldM.Int())
					}
				}
			}
		}
	}

	require.Equalf(t, 0.0, maxDiff.diff,
		"Max difference (floats only): %f, field %s, neuron %d, single thread: %f, multi thread: %f",
		maxDiff.diff, maxDiff.field, maxDiff.neuronIdx, maxDiff.valShould, maxDiff.valIs)
}

// I search for a while for a good way to have expected test failures in Go, but
// couldn't find anything, so I wrote this additional function that returns a bool
// that I can check. Please let me know if there is a better way to do this.
func neuronsAreEqual(netS *Network, netM *Network) bool {
	for li := range netS.Layers {
		layerS := netS.Layers[li].(AxonLayer).AsAxon()
		layerM := netM.Layers[li].(AxonLayer).AsAxon()
		for ni := range layerS.Neurons {
			nrnS := reflect.ValueOf(layerS.Neurons[ni])
			nrnM := reflect.ValueOf(layerM.Neurons[ni])
			for fi := 0; fi < nrnS.NumField(); fi++ {
				fieldS := nrnS.Field(fi)
				fieldM := nrnM.Field(fi)
				if fieldS.Kind() == reflect.Float32 {
					if fieldS.Float() != fieldM.Float() {
						return false
					}
				} else if fieldS.Kind() == reflect.Int32 {
					if fieldS.Int() != fieldM.Int() {
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
// multi-threaded. They are seeded with the same RNG, so they are identical.
// Returns two networks: (single-threaded, multi-threaded)
func buildIdenticalNetworks(t *testing.T, pats *etable.Table, tNeuron, tSendSpike, tSynCa int) (*Network, *Network) {
	shape := []int{shape1D, shape1D}

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
	// inputLayer := netS.LayerByName("Input").(*Layer)
	// outputLayer := netS.LayerByName("Output").(*Layer)
	// input := inPats.SubSpace([]int{0})
	// output := outPats.SubSpace([]int{0})
	// inputLayer.ApplyExt(input)
	// outputLayer.ApplyExt(output)
	// netS.NewState()
	// ctxt := NewContext()
	// ctxt.NewState("train")
	// for i := 0; i < 150; i++ {
	// 	netS.Cycle(ctxt)
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

	assert.True(t, neuronsAreEqual(netS, netM))

	return netS, netM
}

func buildNet(t *testing.T, shape []int, tNeuron, tSendSpike, tSynCa int) *Network {
	net := NewNetwork("MTTest")
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden).(AxonLayer)
	hiddenLayer2 := net.AddLayer("Hidden2", shape, emer.Hidden).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)
	net.ConnectLayers(inputLayer, hiddenLayer, prjn.NewFull(), emer.Forward)
	net.BidirConnectLayers(hiddenLayer, hiddenLayer2, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer2, outputLayer, prjn.NewFull())

	net.Defaults() // Initializes threading defaults, but we override below
	if err := net.Build(); err != nil {
		t.Error(err)
	}
	net.InitWts()

	if err := net.Threads.Set(tNeuron, tSendSpike, tSynCa); err != nil {
		t.Error(err)
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
	inputLayer := net.LayerByName("Input").(*Layer)
	outputLayer := net.LayerByName("Output").(*Layer)
	ctxt := NewContext()
	for epoch := 0; epoch < epochs; epoch++ {
		for pi := 0; pi < pats.NumRows(); pi++ {
			input := inPats.SubSpace([]int{pi})
			output := outPats.SubSpace([]int{pi})

			inputLayer.ApplyExt(input)
			outputLayer.ApplyExt(output)

			net.NewState()
			ctxt.NewState(etime.Train)
			for cycle := 0; cycle < nCycles; cycle++ {
				fun(net, ctxt)
			}
		}
	}
}
