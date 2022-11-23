package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLayer(t *testing.T) {
	net := NewNetwork("LayerTest")
	shape := []int{2, 2}
	inputLayer := net.AddLayer("Input", shape, emer.Input).(AxonLayer)
	outputLayer := net.AddLayer("Output", shape, emer.Target).(AxonLayer)

	assert.True(t, inputLayer.IsInput())
	assert.False(t, outputLayer.IsInput())
	assert.True(t, outputLayer.IsTarget())

	// assert.Equal(t, 4, len(inputLayer.(Layer).Neurons))
}
