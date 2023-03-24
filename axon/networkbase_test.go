package axon

import (
	"testing"

	"github.com/emer/emergent/prjn"
	"github.com/stretchr/testify/assert"
)

func TestAddLayer(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	layer := net.AddLayer("Input", shape, InputLayer)
	assert.Equal(t, 1, net.NLayers())
	assert.Same(t, layer, net.LayByName("Input"))
}

func TestDefaults(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{2, 2}
	input := net.AddLayer("Input", shape, InputLayer)
	hidden := net.AddLayer("Hidden", shape, SuperLayer)
	output := net.AddLayer("Output", shape, TargetLayer)

	full := prjn.NewFull()
	net.ConnectLayers(input, hidden, full, ForwardPrjn)
	net.BidirConnectLayers(hidden, output, full)

	assert.Nil(t, net.Build())
	net.Defaults()
	net.InitWts()

	assert.Equal(t, 100, net.SlowInterval)
	assert.Equal(t, 0, net.SlowCtr)
	assert.Equal(t, 12, len(net.Neurons))

	// test layer access
	assert.Equal(t, net.Layers[0], net.LayByName("Input"))
	assert.Equal(t, net.Layers[1], net.LayByName("Hidden"))
	assert.Equal(t, net.Layers[2], net.LayByName("Output"))
	assert.Nil(t, net.LayByName("DoesNotExist"))
	_, err := net.LayByNameTry("DoesNotExist")
	assert.Error(t, err)
	val := net.LayersByType(InputLayer)
	assert.Equal(t, 1, len(val))
	val = net.LayersByClass("InputLayer")
	assert.Equal(t, 1, len(val))

	for layerIdx, lyr := range net.Layers {
		assert.Equal(t, layerIdx, lyr.Index())
		assert.Equal(t, 4, len(lyr.Neurons))
		for neuronIdx := range lyr.Neurons {
			neuron := &lyr.Neurons[neuronIdx]
			assert.Equal(t, uint32(lyr.Index()), neuron.LayIdx)
		}
	}
}

// TODO: test initial weights somehow

func TestConnectLayers(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	input := net.AddLayer("Input", shape, InputLayer)
	output := net.AddLayer("Output", shape, TargetLayer)
	assert.Equal(t, 2, net.NLayers())
	net.ConnectLayers(input, output, prjn.NewFull(), ForwardPrjn)

	assert.Same(t, output, input.SendPrjn(0).RecvLay())
	assert.Same(t, input, output.RecvPrjn(0).SendLay())
}

func TestDelete(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	net.AddLayer("Input", shape, InputLayer)
	assert.Equal(t, 1, net.NLayers())
	net.DeleteAll()
	assert.Equal(t, 0, net.NLayers())
}
