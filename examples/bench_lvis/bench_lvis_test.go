package bench

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/stretchr/testify/assert"
)

const (
	convergenceTestEpochs = 10
	defaultNumEpochs      = 350
)

var maxProcs = flag.Int("maxProcs", 0, "GOMAXPROCS value to set -- 0 = use current default")
var threadsNeuron = flag.Int("thrNeuron", 0, "number of goroutines to launch for NeuronFun")
var threadsSendSpike = flag.Int("thrSendSpike", 0, "number of goroutines to launch for SendSpike")
var threadsSynCa = flag.Int("thrSynCa", 0, "number of goroutines to launch for SynCa")
var numEpochs = flag.Int("epochs", defaultNumEpochs, "number of epochs to run")
var numPats = flag.Int("pats", 10, "number of patterns per epoch")
var verbose = flag.Bool("verbose", true, "if false, only report the final time")

func BenchmarkBenchNetFull(b *testing.B) {
	inputShape := [2]int{10, 10}
	outputShape := [2]int{4, 4}

	if *maxProcs > 0 {
		runtime.GOMAXPROCS(*maxProcs)
	}

	if *verbose {
		fmt.Printf("Running bench with: %d neuronThreads, %d sendSpikeThreads, %d synCaThreads, %d epochs, %d pats, (%d, %d) input, (%d, %d) output\n",
			*threadsNeuron, *threadsSendSpike, *threadsSynCa,
			*numEpochs, *numPats, inputShape[0], inputShape[1], outputShape[0], outputShape[1])
	}

	rand.Seed(42)

	net := &axon.Network{}
	ConfigNet(net, *threadsNeuron, *threadsSendSpike, *threadsSynCa, *verbose)
	if *verbose {
		log.Println(net.SizeReport())
	}

	pats := &etable.Table{}
	ConfigPats(pats, *numPats, inputShape, outputShape)

	inLay := net.LayerByName("V1m16").(*axon.Layer)
	outLay := net.LayerByName("Output").(*axon.Layer)

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)

	// todo: is the input shape actually correct for a 4D layer?
	assert.Equal(b, inLay.Shp.Len(), inPats.Len())
	assert.Equal(b, outLay.Shp.Len(), outPats.Len())

	epcLog := &etable.Table{}
	ConfigEpcLog(epcLog)

	TrainNet(net, pats, epcLog, *numEpochs, *verbose)
}
