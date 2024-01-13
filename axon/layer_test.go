package axon

import (
	"bufio"
	"os"
	"testing"

	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/etable/v2/etensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLayer(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, InputLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, SuperLayer)
	outputLayer := net.AddLayer("Output", shape, TargetLayer)

	ctx := NewContext()
	assert.NoError(t, net.Build(ctx))

	assert.True(t, inputLayer.Params.IsInput())
	assert.False(t, outputLayer.Params.IsInput())
	assert.True(t, outputLayer.Params.IsTarget())

	// the layer.Neuron struct is empty before Build(), which may be surprising to the user?
	assert.Equal(t, uint32(4), hiddenLayer.NNeurons)

	// query the 'Spike' variable for all neurons of the layer
	tensor := etensor.NewFloat32([]int{2}, nil, nil)
	assert.Nil(t, hiddenLayer.UnitValsTensor(tensor, "Spike", 0))
	for i := 0; i < 4; i++ {
		// can't have spiked as we haven't run the network yet
		assert.Equal(t, float32(0.0), tensor.Values[i])
	}
	assert.Equal(t, []int{2, 2}, tensor.Shape.Shp)
}

/*
func TestLayer_SendSpike(t *testing.T) {
	t.Skip("skipping -- needs reorg to recv based")
	net := NewNetwork("LayerTest")
	shape := []int{3, 3}
	inputLayer1 := net.AddLayer("Input1", shape, InputLayer)
	inputLayer2 := net.AddLayer("Input2", shape, InputLayer)
	outputLayer := net.AddLayer("Output", shape, TargetLayer)
	net.ConnectLayers(inputLayer1, outputLayer, prjn.NewFull(), ForwardPrjn)
	net.ConnectLayers(inputLayer2, outputLayer, prjn.NewFull(), ForwardPrjn)


	// Input1 -> Output
	// Input2 -^


	assert.NoError(t, net.Build())
	net.Defaults()
	net.InitWts()

	ctx := NewContext()
	net.NewState(ctx)

	// spike the neurons. Do this after NewState(), so that the spike is not decayed away
	inputLayer1.Neurons[1].Spike = 1.0
	inputLayer2.Neurons[0].Spike = 1.0

	// set some of the weights
	const in1pj0_n1_to_n2_wt = 0.1
	const in1pj0_scale = 6.6
	in1pj0 := inputLayer1.SendPrjn(0).(*Prjn)
	// in1pj0.Syns[in1pj0.SendConIdxStart[1]].Wt = in1pj0_n1_to_n2_wt
	in1pj0.Params.GScale.Scale = in1pj0_scale

	const in2pj0_n0_to_n4_wt = 3.0
	const in2pj0_scale = 0.4
	in2pj0 := inputLayer2.SendPrjn(0).(*Prjn)
	in2pj0.Params.GScale.Scale = in2pj0_scale
	// in2pj0.Syns[in2pj0.SendConIdxStart[0]+4].Wt = in2pj0_n0_to_n4_wt

	net.SendSpikeFun(func(ly AxonLayer) { ly.SendSpike(ctx) },
		"SendSpike")

	// the neurons we spiked are connected to 9 neurons in the output layer
	// make sure they all received the spike
	recvBuffs := [][]float32{
		inputLayer1.SndPrjns[0].(*Prjn).GBuf,
		inputLayer2.SndPrjns[0].(*Prjn).GBuf,
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

	delayStride := in1pj0.Params.Com.Delay + 1
	assert.Equal(t, in1pj0.Params.Com.Delay, in2pj0.Params.Com.Delay) // sanity

	// spot-check two of the conductances
	l1contrib := float32(in1pj0_n1_to_n2_wt) * in1pj0_scale
	l2contrib := float32(in2pj0_n0_to_n4_wt) * in2pj0_scale
	assert.Equal(t, l1contrib, recvBuffs[0][0*delayStride+(delayStride-1)])
	assert.Equal(t, l2contrib, recvBuffs[1][4*delayStride+(delayStride-1)])
}
*/

func TestLayerToJson(t *testing.T) {
	shape := []int{2, 2}
	ctx := NewContext()
	ctxC := NewContext()

	// create network has internal calls to Random number generators.
	// We test the JSON import and export by creating two networks (initally different)
	// and syncing them by dumping the weights from net A and loading the weights
	// from net B. TODO: Would be better if we ran a cycle first, to get more variance.
	net := createNetwork(ctx, shape, t)
	hiddenLayer := net.AxonLayerByName("Hidden")
	net.Cycle(ctx) // run one cycle to make the weights more different

	netC := createNetwork(ctxC, shape, t)
	hiddenLayerC := netC.AxonLayerByName("Hidden")

	// save to JSON
	filename := t.TempDir() + "/layer.json"
	fh, err := os.Create(filename)
	require.NoError(t, err)
	bw := bufio.NewWriter(fh)
	hiddenLayer.WriteWtsJSON(bw, 0)
	assert.NoError(t, bw.Flush())
	assert.NoError(t, fh.Close())

	// load from JSON
	fh, err = os.Open(filename)
	require.NoError(t, err)
	br := bufio.NewReader(fh)
	assert.NoError(t, hiddenLayerC.ReadWtsJSON(br))
	assert.NoError(t, fh.Close())

	// make sure the synapse weights are the same
	origProj := hiddenLayer.RcvPrjns[0]
	copyProj := hiddenLayerC.RcvPrjns[0]
	varIdx, _ := origProj.SynVarIdx("Wt")
	assert.Equal(t, origProj.Syn1DNum(), copyProj.Syn1DNum())
	for idx := 0; idx < origProj.Syn1DNum(); idx++ {
		origWeight := origProj.SynVal1D(varIdx, idx)
		copyWeight := copyProj.SynVal1D(varIdx, idx)
		assert.InDelta(t, origWeight, copyWeight, 0.001)
	}

	trgavgs := []float32{}
	actavgs := []float32{}
	trgavgsC := []float32{}
	actavgsC := []float32{}
	hiddenLayer.UnitVals(&trgavgs, "TrgAvg", 0)
	hiddenLayer.UnitVals(&actavgs, "ActAvg", 0)
	hiddenLayerC.UnitVals(&trgavgsC, "TrgAvg", 0)
	hiddenLayerC.UnitVals(&actavgsC, "ActAvg", 0)

	for i := range trgavgs {
		assert.Equal(t, trgavgs[i], trgavgsC[i])
		assert.Equal(t, actavgs[i], actavgsC[i])
	}
}

func createNetwork(ctx *Context, shape []int, t *testing.T) *Network {
	net := NewNetwork("LayerTest")
	inputLayer := net.AddLayer("Input", shape, InputLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, SuperLayer)
	outputLayer := net.AddLayer("Output", shape, TargetLayer)
	full := prjn.NewFull()
	net.ConnectLayers(inputLayer, hiddenLayer, full, ForwardPrjn)
	net.BidirConnectLayers(hiddenLayer, outputLayer, full)
	assert.NoError(t, net.Build(ctx))
	net.Defaults()
	net.InitWts(ctx)
	return net
}

func TestLayerBase_IsOff(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, InputLayer)
	inputLayer2 := net.AddLayer("Input2", shape, InputLayer)
	hiddenLayer := net.AddLayer("Hidden", shape, SuperLayer)
	outputLayer := net.AddLayer("Output", shape, TargetLayer)

	full := prjn.NewFull()
	inToHid := net.ConnectLayers(inputLayer, hiddenLayer, full, ForwardPrjn)
	in2ToHid := net.ConnectLayers(inputLayer2, hiddenLayer, full, ForwardPrjn)
	hidToOut, outToHid := net.BidirConnectLayers(hiddenLayer, outputLayer, full)

	ctx := NewContext()

	assert.NoError(t, net.Build(ctx))
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
