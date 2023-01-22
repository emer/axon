package axon

import (
	"bufio"
	"os"
	"testing"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLayer(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)

	assert.True(t, inputLayer.IsInput())
	assert.False(t, outputLayer.IsInput())
	assert.True(t, outputLayer.IsTarget())

	assert.NoError(t, net.Build())

	// the layer.Neuron struct is empty before Build(), which may be surprising to the user?
	assert.Equal(t, 4, len(hiddenLayer.AsAxon().Neurons))

	// query the 'Spike' variable for all neurons of the layer
	tensor := etensor.NewFloat32([]int{2}, nil, nil)
	assert.Nil(t, hiddenLayer.AsAxon().UnitValsTensor(tensor, "Spike"))
	for i := 0; i < 4; i++ {
		// can't have spiked as we haven't run the network yet
		assert.Equal(t, float32(0.0), tensor.Values[i])
	}
	assert.Equal(t, []int{2, 2}, tensor.Shape.Shp)
}

func TestLayer_SendSpike(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{3, 3}
	inputLayer1 := net.AddLayer("Input1", shape, emer.Input).(AxonLayer)
	inputLayer2 := net.AddLayer("Input2", shape, emer.Input).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)
	net.ConnectLayers(inputLayer1, outputLayer, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(inputLayer2, outputLayer, prjn.NewFull(), emer.Forward)

	/*
	 * Input1 -> Output
	 * Input2 -^
	 */

	assert.NoError(t, net.Build())
	net.Defaults()
	net.InitWts()

	ctx := NewContext()
	net.NewState(ctx)

	// spike the first neuron. Do this after NewState(), so that the spike is not decayed away
	inputLayer1.AsAxon().Neurons[1].Spike = 1.0
	inputLayer2.AsAxon().Neurons[0].Spike = 1.0

	// set some of the weights
	in1pj0 := inputLayer1.SendPrjn(0).(*Prjn)
	in1pj0.Syns[in1pj0.SendConIdxStart[1]].Wt = 0.1
	in1pj0.Params.GScale.Scale = 6.6

	in2pj0 := inputLayer2.SendPrjn(0).(*Prjn)
	in2pj0.Syns[in2pj0.SendConIdxStart[0]+4].Wt = 3.0
	in2pj0.Params.GScale.Scale = 0.4

	net.SendSpikeFun(func(ly AxonLayer) { ly.SendSpike(ctx) },
		"SendSpike")

	// the neuron we spiked is connected to 9 neurons in the output layer
	// make sure they all received the spike
	recvBuffs := [][]float32{
		inputLayer1.AsAxon().SndPrjns[0].(*Prjn).GBuf,
		inputLayer2.AsAxon().SndPrjns[0].(*Prjn).GBuf,
	}
	for _, recvBuf := range recvBuffs {
		count := 0
		for _, g := range recvBuf {
			if g > 0.0 {
				count++
			}
		}
		assert.Equal(t, 9, count)
	}

	// spot-check two of the conductances
	delay := 3
	l1contrib := 6.6 * 0.1
	l2contrib := 3.0 * 0.4
	assert.Equal(t, float32(l1contrib), recvBuffs[0][0*delay+(delay-1)])
	assert.Equal(t, float32(l2contrib), recvBuffs[1][4*delay+(delay-1)])
}

func TestLayerToJson(t *testing.T) {
	shape := []int{2, 2}

	// create network has internal calls to Random number generators.
	// We test the JSON import and export by creating two networks (initally different)
	// and syncing them by dumping the weights from net A and loading the weights
	// from net B. TODO: Would be better if we ran a cycle first, to get more variance.
	net := createNetwork(shape, t)
	hiddenLayer := net.LayerByName("Hidden").(AxonLayer)
	ctx := NewContext()
	net.Cycle(ctx) // run one cycle to make the weights more different

	netC := createNetwork(shape, t)
	hiddenLayerC := netC.LayerByName("Hidden").(AxonLayer)

	// save to JSON
	filename := t.TempDir() + "/layer.json"
	fh, err := os.Create(filename)
	require.NoError(t, err)
	bw := bufio.NewWriter(fh)
	hiddenLayer.WriteWtsJSON(bw, 0)
	// t.Log(filename)
	assert.NoError(t, bw.Flush())
	assert.NoError(t, fh.Close())

	// load from JSON
	fh, err = os.Open(filename)
	require.NoError(t, err)
	br := bufio.NewReader(fh)
	assert.NoError(t, hiddenLayerC.ReadWtsJSON(br))
	assert.NoError(t, fh.Close())

	// make sure the synapse weights are the same
	origProj := hiddenLayer.AsAxon().RcvPrjns[0]
	copyProj := hiddenLayerC.AsAxon().RcvPrjns[0]
	varIdx, _ := origProj.SynVarIdx("Wt")
	assert.Equal(t, origProj.Syn1DNum(), copyProj.Syn1DNum())
	for idx := 0; idx < origProj.Syn1DNum(); idx++ {
		origWeight := origProj.SynVal1D(varIdx, idx)
		copyWeight := copyProj.SynVal1D(varIdx, idx)
		assert.InDelta(t, origWeight, copyWeight, 0.001)
	}

	nrns := hiddenLayer.AsAxon().Neurons
	nrnsC := hiddenLayerC.AsAxon().Neurons
	// right now only two of the Neuron variables are exported
	for i := range nrns {
		assert.Equal(t, nrns[i].TrgAvg, nrnsC[i].TrgAvg)
		assert.Equal(t, nrns[i].ActAvg, nrnsC[i].ActAvg)
	}

}

func createNetwork(shape []int, t *testing.T) *Network {
	net := NewNetwork("LayerTest")
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden)
	outputLayer := net.AddLayer("Output", shape, emer.Target)
	full := prjn.NewFull()
	net.ConnectLayers(inputLayer, hiddenLayer, full, emer.Forward)
	net.BidirConnectLayers(hiddenLayer, outputLayer, full)
	assert.NoError(t, net.Build())
	net.Defaults()
	net.InitWts()
	return net
}

func TestLayerBase_IsOff(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	inputLayer2 := net.AddLayer("Input2", shape, emer.Input).(AxonLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, emer.Hidden).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)

	full := prjn.NewFull()
	inToHid := net.ConnectLayers(inputLayer, hiddenLayer, full, emer.Forward)
	in2ToHid := net.ConnectLayers(inputLayer2, hiddenLayer, full, emer.Forward)
	hidToOut, outToHid := net.BidirConnectLayers(hiddenLayer, outputLayer, full)

	assert.NoError(t, net.Build())
	net.Defaults()

	assert.False(t, inputLayer.IsOff())

	inputLayer.SetOff(true)
	assert.True(t, inputLayer.IsOff())
	assert.False(t, hiddenLayer.IsOff())
	assert.True(t, inToHid.IsOff())
	assert.False(t, in2ToHid.IsOff())

	inputLayer2.SetOff(true)
	assert.True(t, inputLayer2.IsOff())
	assert.False(t, hiddenLayer.IsOff())
	assert.True(t, in2ToHid.IsOff())

	hiddenLayer.SetOff(true)
	assert.True(t, hiddenLayer.IsOff())
	assert.True(t, hidToOut.IsOff())
	assert.True(t, outToHid.IsOff())

	hiddenLayer.SetOff(false)
	assert.False(t, hiddenLayer.IsOff())
	assert.False(t, hidToOut.IsOff())
	assert.False(t, outToHid.IsOff())
	assert.True(t, inToHid.IsOff())
	assert.True(t, in2ToHid.IsOff())
}
