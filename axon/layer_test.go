package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
	"github.com/stretchr/testify/assert"
	"testing"
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
