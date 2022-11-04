package axon

import (
	"fmt"
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/stretchr/testify/assert"
)

func TestAddLayer(t *testing.T) {
	net := axon.NewNetwork("testNet")
	shape := []int{5, 5}
	layer := net.AddLayer("Input", shape, emer.Input)
	assert.Equal(t, 1, net.NLayers())
	assert.Same(t, layer, net.LayerByName("Input"))
}

func TestDefaults(t *testing.T) {
	net := axon.NewNetwork("testNet")
	net.Defaults()
	net.Build()
	net.InitWts()
	fmt.Println(net)

	assert.Equal(t, 100, net.SlowInterval)
	assert.Equal(t, 0, net.SlowCtr)
}

// TODO: test initial weights somehow

func TestConnectLayers(t *testing.T) {
	net := axon.NewNetwork("testNet")
	shape := []int{5, 5}
	input := net.AddLayer("Input", shape, emer.Input)
	output := net.AddLayer("Output", shape, emer.Target)
	assert.Equal(t, 2, net.NLayers())
	net.ConnectLayers(input, output, prjn.NewFull(), emer.Forward)

	assert.Same(t, output, input.SendPrjn(0).RecvLay())
	assert.Same(t, input, output.RecvPrjn(0).SendLay())
}

func TestDelete(t *testing.T) {
	net := axon.NewNetwork("testNet")
	shape := []int{5, 5}
	net.AddLayer("Input", shape, emer.Input)
	assert.Equal(t, 1, net.NLayers())
	net.DeleteAll()
	assert.Equal(t, 0, net.NLayers())
}
