package axon

import (
	"bufio"
	"fmt"
	"os"
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

func TestMultithreadingNeuronFun(t *testing.T) {
	shape := []int{8, 8}
	nPats := 100

	netS := buildNet(shape, 1, t)
	netM := buildNet(shape, 5, t)

	// sync the randomly initialized weights
	// filename := t.TempDir() + "/netS.json"
	filename := "netS.json"
	// write Synapse weights to file
	fh, err := os.Create(filename)
	require.NoError(t, err)
	bw := bufio.NewWriter(fh)
	require.NoError(t, netS.WriteWtsJSON(bw))
	require.NoError(t, bw.Flush())
	require.NoError(t, fh.Close())
	// read Synapse weights from file
	fh, err = os.Open(filename)
	require.NoError(t, err)
	br := bufio.NewReader(fh)
	require.NoError(t, netM.ReadWtsJSON(br))
	require.NoError(t, fh.Close())
	// sync Neuron weights as well (TODO: we need a better way to do this)
	copy(netM.Neurons, netS.Neurons)

	// generate some random patterns
	pats := &etable.Table{}
	pats.SetFromSchema(etable.Schema{
		{Name: "Name", Type: etensor.STRING, CellShape: nil, DimNames: nil},
		{Name: "Input", Type: etensor.FLOAT32, CellShape: shape, DimNames: []string{"Y", "X"}},
		{Name: "Output", Type: etensor.FLOAT32, CellShape: shape, DimNames: []string{"Y", "X"}},
	}, nPats)
	numOn := shape[0] * shape[1] / 8
	patgen.PermutedBinaryRows(pats.Cols[1], numOn, 1, 0)
	patgen.PermutedBinaryRows(pats.Cols[2], numOn, 1, 0)
	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)

	runSomeCycles(pats, inPats, outPats, netS, 10)
	runSomeCycles(pats, inPats, outPats, netM, 10)

	// compare the results
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
					assert.InDelta(t, fieldS.Float(), fieldM.Float(), 0.0001,
						"Neuron %d, field %s, single thread: %f, multi thread: %f",
						ni, nrnS.Type().Field(fi).Name, fieldS.Float(), fieldM.Float())
				} else if fieldS.Kind() == reflect.Int32 {
					assert.Equal(t, fieldS.Int(), fieldM.Int(),
						"Neuron %d, field %s, single thread: %d, multi thread: %d",
						ni, nrnS.Type().Field(fi).Name, fieldS.Int(), fieldM.Int())
				}
			}
		}
	}
}

func buildNet(shape []int, nThreads int, t *testing.T) *Network {
	net := NewNetwork(fmt.Sprint("MTTest", nThreads))
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)
	net.ConnectLayers(inputLayer, hiddenLayer, prjn.NewFull(), emer.Forward)
	net.BidirConnectLayers(hiddenLayer, outputLayer, prjn.NewFull())

	net.Defaults()
	net.NThreads = nThreads
	if err := net.Build(); err != nil {
		t.Error(err)
	}
	net.InitWts()

	// override again, just to be safe
	net.Threads.Set(2, nThreads, nThreads, nThreads, nThreads)
	net.ThreadsAlloc()
	return net
}

func runSomeCycles(pats *etable.Table, inPats *etensor.Float32, outPats *etensor.Float32, net *Network, cycles int) {
	inputLayer := net.LayerByName("Input").(*Layer)
	outputLayer := net.LayerByName("Output").(*Layer)
	ltime := NewTime()
	for pi := 0; pi < pats.NumRows(); pi++ {
		input := inPats.SubSpace([]int{pi})
		output := outPats.SubSpace([]int{pi})

		inputLayer.ApplyExt(input)
		outputLayer.ApplyExt(output)

		net.NewState()
		ltime.NewState("train")
		for cycle := 0; cycle < cycles; cycle++ {
			net.NeuronFun(func(ly AxonLayer, ni int, nrn *Neuron) {
				ly.CycleNeuron(ni, nrn, ltime)
			},
				"CycleNeuron", true, true)
		}
	}
}
