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
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)
	net.ConnectLayers(inputLayer, outputLayer, prjn.NewFull(), emer.Forward)
	net.Defaults()

	assert.NoError(t, net.Build())
	net.InitWts()

	net.NewState()
	ltime := NewTime()

	// spike the first neuron. Do this after NewState(), so that the spike is not decayed away
	inputLayer.AsAxon().Neurons[0].Spike = 1.0
	net.SendSpikeFun(func(ly AxonLayer, ni int, nrn *Neuron) { ly.SendSpike(ni, nrn, ltime) },
		"SendSpike", NoThread, Wait)

	// the neuron we spiked is connected to 4 neurons in the output layer
	// make sure they all received the spike
	conductBuf := inputLayer.AsAxon().SndPrjns[0].(*Prjn).GBuf
	count := 0
	for _, g := range conductBuf {
		if g > 0.0 {
			count++
		}
	}
	assert.Equal(t, 4, count)
}

func TestLayerToJson(t *testing.T) {
	shape := []int{2, 2}

	origNet := NewNetwork("LayerTest")
	inputLayer := origNet.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayer := origNet.AddLayer("Hidden", shape, emer.Hidden)
	outputLayer := origNet.AddLayer("Output", shape, emer.Target)
	full := prjn.NewFull()
	origNet.ConnectLayers(inputLayer, hiddenLayer, full, emer.Forward)
	origNet.BidirConnectLayers(hiddenLayer, outputLayer, full)
	origNet.Defaults()
	assert.NoError(t, origNet.Build())
	origNet.InitWts()

	// save to json
	filename := "layer.json"
	fh, err := os.Create(filename)
	require.NoError(t, err)
	bw := bufio.NewWriter(fh)
	hiddenLayer.WriteWtsJSON(bw, 0)
	t.Log(filename)
	assert.NoError(t, bw.Flush())
	assert.NoError(t, fh.Close())

	copyNet := NewNetwork("LayerTest")
	inputLayerC := copyNet.AddLayer("Input", shape, emer.Input).(AxonLayer)
	hiddenLayerC := copyNet.AddLayer("Hidden", shape, emer.Hidden)
	outputLayerC := copyNet.AddLayer("Output", shape, emer.Target)
	copyNet.ConnectLayers(inputLayerC, hiddenLayerC, full, emer.Forward)
	copyNet.BidirConnectLayers(hiddenLayerC, outputLayerC, full)
	copyNet.Defaults()
	assert.NoError(t, copyNet.Build())
	copyNet.InitWts()

	// load from json
	fh, err = os.Open(filename)
	require.NoError(t, err)
	br := bufio.NewReader(fh)
	assert.NoError(t, hiddenLayerC.ReadWtsJSON(br))
	assert.NoError(t, fh.Close())

	// make sure the syn weights are the same
	origProj := hiddenLayer.(AxonLayer).AsAxon().RcvPrjns[0]
	copyProj := hiddenLayerC.(AxonLayer).AsAxon().RcvPrjns[0]
	varIdx, err := origProj.SynVarIdx("Wt")
	assert.Equal(t, origProj.Syn1DNum(), copyProj.Syn1DNum())
	for idx := 0; idx < origProj.Syn1DNum(); idx++ {
		origWeight := origProj.SynVal1D(varIdx, idx)
		copyWeight := copyProj.SynVal1D(varIdx, idx)
		assert.InDelta(t, origWeight, copyWeight, 0.001)
	}
}
