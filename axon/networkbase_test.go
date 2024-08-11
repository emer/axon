package axon

import (
	"testing"

	"github.com/emer/emergent/v2/paths"
	"github.com/stretchr/testify/assert"
)

func TestAddLayer(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	layer := net.AddLayer("Input", shape, InputLayer)
	assert.Equal(t, 1, net.NLayers())
	assert.Same(t, layer, net.LayerByName("Input"))
}

func TestDefaults(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{2, 2}
	input := net.AddLayer("Input", shape, InputLayer)
	hidden := net.AddLayer("Hidden", shape, SuperLayer)
	output := net.AddLayer("Output", shape, TargetLayer)

	full := paths.NewFull()
	net.ConnectLayers(input, hidden, full, ForwardPath)
	net.BidirConnectLayers(hidden, output, full)

	ctx := NewContext()
	assert.Nil(t, net.Build(ctx))
	net.Defaults()
	net.InitWeights(ctx)

	assert.Equal(t, 100, int(ctx.SlowInterval))
	assert.Equal(t, 0, int(ctx.SlowCtr))
	assert.Equal(t, uint32(12), net.NNeurons)

	// test layer access
	assert.Equal(t, net.Layers[0], net.LayerByName("Input"))
	assert.Equal(t, net.Layers[1], net.LayerByName("Hidden"))
	assert.Equal(t, net.Layers[2], net.LayerByName("Output"))
	assert.Nil(t, net.LayerByName("DoesNotExist"))
	_, err := net.LayerByName("DoesNotExist")
	assert.Error(t, err)
	val := net.LayersByType(InputLayer)
	assert.Equal(t, 1, len(val))
	val = net.LayersByClass("InputLayer")
	assert.Equal(t, 1, len(val))

	for layerIndex, lyr := range net.Layers {
		assert.Equal(t, layerIndex, lyr.Index())
		assert.Equal(t, uint32(4), lyr.NNeurons)
		for lni := uint32(0); lni < lyr.NNeurons; lni++ {
			ni := lyr.NeurStIndex + lni
			li := NrnI(ctx, ni, NrnLayIndex)
			assert.Equal(t, uint32(lyr.Index()), li)
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
	net.ConnectLayers(input, output, paths.NewFull(), ForwardPath)

	assert.Same(t, output, input.SendPath(0).RecvLayer())
	assert.Same(t, input, output.RecvPath(0).SendLayer())
}

func TestDelete(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	net.AddLayer("Input", shape, InputLayer)
	assert.Equal(t, 1, net.NLayers())
	net.DeleteAll()
	assert.Equal(t, 0, net.NLayers())
}
