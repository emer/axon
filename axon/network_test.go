package axon

// TODO: should we make a network package?

import (
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/stretchr/testify/assert"
)

func TestNewNetwork(t *testing.T) {
	testNet := axon.NewNetwork("testNet")
	assert.Equal(t, "testNet", testNet.Name())
	assert.Equal(t, 0, testNet.NLayers())
	assert.IsType(t, &axon.Network{}, testNet)
}

func TestInterfaceType(t *testing.T) {
	testNet := axon.NewNetwork("testNet")
	_, ok := testNet.EmerNet.(emer.Network)
	assert.True(t, ok)
}
