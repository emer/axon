package axon

import (
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestNeuronVarStart(t *testing.T) {
	typ := reflect.TypeOf((*Neuron)(nil)).Elem()
	for i := NeuronVarStart; i < typ.NumField(); i++ {
		fld := typ.Field(i)
		assert.Equal(t, fld.Type.Kind(), reflect.Float32)
	}
}

func TestNeuron(t *testing.T) {
	nrn := Neuron{}
	assert.Contains(t, nrn.VarNames(), "Spike")
}
