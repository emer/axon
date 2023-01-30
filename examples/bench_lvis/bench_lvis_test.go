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
var numUnits = flag.Int("units", 100, "number of units per layer -- uses NxN where N = sqrt(units)")
var verbose = flag.Bool("verbose", true, "if false, only report the final time")
var writeStats = flag.Bool("writestats", false, "whether to write network stats to a CSV file")

func BenchmarkBenchNetFull(b *testing.B) {
	if *maxProcs > 0 {
		runtime.GOMAXPROCS(*maxProcs)
	}

	if *verbose {
		fmt.Printf("Running bench with: %d neuronThreads, %d sendSpikeThreads, %d synCaThreads, %d epochs, %d pats, %d units\n", *threadsNeuron, *threadsSendSpike, *threadsSynCa, *numEpochs, *numPats, *numUnits)
	}

	rand.Seed(42)

	net := &axon.Network{}
	ConfigNet(net, *threadsNeuron, *threadsSendSpike, *threadsSynCa, *numUnits, *verbose)
	if *verbose {
		log.Println(net.SizeReport())
	}

	pats := &etable.Table{}
	ConfigPats(pats, *numPats, *numUnits)

	epcLog := &etable.Table{}
	ConfigEpcLog(epcLog)

	TrainNet(net, pats, epcLog, *numEpochs, *verbose)
}
