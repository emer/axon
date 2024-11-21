// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"bytes"
	"testing"

	"github.com/emer/emergent/v2/paths"
	"github.com/stretchr/testify/assert"
)

func TestAddLayer(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	layer := net.AddLayer("Input", InputLayer, shape...)
	assert.Equal(t, 1, net.NumLayers())
	assert.Same(t, layer, net.LayerByName("Input"))
}

func TestDefaults(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{2, 2}
	input := net.AddLayer("Input", InputLayer, shape...)
	hidden := net.AddLayer("Hidden", SuperLayer, shape...)
	output := net.AddLayer("Output", TargetLayer, shape...)

	full := paths.NewFull()
	net.ConnectLayers(input, hidden, full, ForwardPath)
	net.BidirConnectLayers(hidden, output, full)

	assert.Nil(t, net.Build())
	net.Defaults()
	net.InitWeights()
	ctx := net.Context()

	assert.Equal(t, 100, int(ctx.SlowInterval))
	assert.Equal(t, 0, int(ctx.SlowCounter))
	assert.Equal(t, uint32(12), net.NetIxs().NNeurons)

	// test layer access
	assert.Equal(t, net.Layers[0], net.LayerByName("Input"))
	assert.Equal(t, net.Layers[1], net.LayerByName("Hidden"))
	assert.Equal(t, net.Layers[2], net.LayerByName("Output"))
	assert.Nil(t, net.LayerByName("DoesNotExist"))
	_, err := net.EmerLayerByName("DoesNotExist")
	assert.Error(t, err)
	val := net.LayersByType(InputLayer)
	assert.Equal(t, 1, len(val))
	val = net.LayersByClass("InputLayer")
	assert.Equal(t, 1, len(val))

	for layerIndex, lyr := range net.Layers {
		assert.Equal(t, layerIndex, lyr.Index)
		assert.Equal(t, uint32(4), lyr.NNeurons)
		for lni := uint32(0); lni < lyr.NNeurons; lni++ {
			// ni := lyr.NeurStIndex + lni
			// li := NeuronIxs[ni, NrnLayIndex]
			// assert.Equal(t, uint32(lyr.Index), li)
		}
	}
}

// TODO: test initial weights somehow

func TestConnectLayers(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	input := net.AddLayer("Input", InputLayer, shape...)
	output := net.AddLayer("Output", TargetLayer, shape...)
	assert.Equal(t, 2, net.NumLayers())
	net.ConnectLayers(input, output, paths.NewFull(), ForwardPath)

	assert.Same(t, output, input.SendPath(0).RecvLayer())
	assert.Same(t, input, output.RecvPath(0).SendLayer())
}

func TestDelete(t *testing.T) {
	net := NewNetwork("testNet")
	shape := []int{5, 5}
	net.AddLayer("Input", InputLayer, shape...)
	assert.Equal(t, 1, net.NumLayers())
	net.DeleteAll()
	assert.Equal(t, 0, net.NumLayers())
}

var stdWeights = `{
	"Network": "testNet",
	"Layers": [
		{
			"Layer": "Input",
			"MetaData": {
				"ActMAvg": "0.1",
				"ActPAvg": "0.1",
				"GiMult": "1"
			},
			"Paths": null
		},
		{
			"Layer": "Hidden",
			"MetaData": {
				"ActMAvg": "0.1",
				"ActPAvg": "0.1",
				"GiMult": "1"
			},
			"Units": {
				"ActAvg": [ 0.05, 0.15, 0.2, 0.1 ],
				"TrgAvg": [ 0.5, 1.5, 2, 1 ]
			},
			"Paths": [
				{
					"From": "Input",
					"Rs": [
						{
							"Ri": 0,
							"N": 1,
							"Si": [ 0 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 1,
							"N": 1,
							"Si": [ 1 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 2,
							"N": 1,
							"Si": [ 2 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 3,
							"N": 1,
							"Si": [ 3 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						}
					]
				},
				{
					"From": "Output",
					"Rs": [
						{
							"Ri": 0,
							"N": 1,
							"Si": [ 0 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 1,
							"N": 1,
							"Si": [ 1 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 2,
							"N": 1,
							"Si": [ 2 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 3,
							"N": 1,
							"Si": [ 3 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						}
					]
				}
			 ]
		},
		{
			"Layer": "Output",
			"MetaData": {
				"ActMAvg": "0.1",
				"ActPAvg": "0.1",
				"GiMult": "1"
			},
			"Paths": [
				{
					"From": "Hidden",
					"Rs": [
						{
							"Ri": 0,
							"N": 1,
							"Si": [ 0 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 1,
							"N": 1,
							"Si": [ 1 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 2,
							"N": 1,
							"Si": [ 2 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						},
						{
							"Ri": 3,
							"N": 1,
							"Si": [ 3 ],
							"Wt": [ 0.5 ],
							"Wt1": [ 0.5 ]
						}
					]
				}
			 ]
		}
	]
}
`

func TestSaveWeights(t *testing.T) {
	var b bytes.Buffer
	testNet := newTestNet(1)
	err := testNet.WriteWeightsJSON(&b)
	if err != nil {
		t.Error(err.Error())
	}
	assert.Equal(t, stdWeights, string(b.Bytes()))
	// fmt.Println(string(b.Bytes()))

	loadNet := newTestNet(1)
	err = loadNet.ReadWeightsJSON(&b)
	if err != nil {
		t.Error(err.Error())
	}
}
