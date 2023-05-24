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
	assert.Same(t, layer, net.AxonLayerByName("Input"))
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

	ctx := NewContext()
	assert.Nil(t, net.Build(ctx))
	net.Defaults()
	net.InitWts(ctx)

	assert.Equal(t, 100, ctx.SlowInterval)
	assert.Equal(t, 0, ctx.SlowCtr)
	assert.Equal(t, uint32(12), net.NNeurons)

	// test layer access
	assert.Equal(t, net.Layers[0], net.AxonLayerByName("Input"))
	assert.Equal(t, net.Layers[1], net.AxonLayerByName("Hidden"))
	assert.Equal(t, net.Layers[2], net.AxonLayerByName("Output"))
	assert.Nil(t, net.AxonLayerByName("DoesNotExist"))
	_, err := net.LayByNameTry("DoesNotExist")
	assert.Error(t, err)
	val := net.LayersByType(InputLayer)
	assert.Equal(t, 1, len(val))
	val = net.LayersByClass("InputLayer")
	assert.Equal(t, 1, len(val))

	for layerIdx, lyr := range net.Layers {
		assert.Equal(t, layerIdx, lyr.Index())
		assert.Equal(t, uint32(4), lyr.NNeurons)
		for lni := uint32(0); lni < lyr.NNeurons; lni++ {
			ni := lyr.NeurStIdx + lni
			li := NrnI(ctx, ni, NrnLayIdx)
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
