package bench

import (
	"flag"
	"fmt"
	"log"
	"math"
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
)

const (
	convergenceTestEpochs = 10
	defaultNumEpochs      = 350
)

var numThreads = flag.Int("threads", 1, "number of threads (goroutines) to use")
var numEpochs = flag.Int("epochs", defaultNumEpochs, "number of epochs to run")
var numPats = flag.Int("pats", 10, "number of patterns per epoch")
var numUnits = flag.Int("units", 100, "number of units per layer -- uses NxN where N = sqrt(units)")
var verbose = flag.Bool("verbose", true, "if false, only report the final time")
var writeStats = flag.Bool("writestats", false, "whether to write network stats to a CSV file")

func BenchmarkAxon(b *testing.B) {
	if *verbose {
		fmt.Printf("Running bench with: %d threads, %d epochs, %d pats, %d units\n", *numThreads, *numEpochs, *numPats, *numUnits)
	}

	net := &axon.Network{}
	ConfigNet(net, *numThreads, *numUnits, *verbose)
	if *verbose {
		log.Println(net.SizeReport())
	}

	pats := &etable.Table{}
	ConfigPats(pats, *numPats, *numUnits)

	epcLog := &etable.Table{}
	ConfigEpcLog(epcLog)

	TrainNet(net, pats, epcLog, *numEpochs, *verbose)

	if *writeStats {
		filename := fmt.Sprintf("bench_%d_units.csv", *numUnits)
		epcLog.SaveCSV(gi.FileName(filename), ',', etable.Headers)
	}

	if *numEpochs < defaultNumEpochs {
		b.Logf("skipping convergence test because numEpochs < %d", defaultNumEpochs)
		return
	}
	corSimSum := 0.0
	for epoch := *numEpochs - convergenceTestEpochs; epoch < *numEpochs; epoch++ {
		corSimSum += epcLog.CellFloat("CorSim", epoch)
		if math.IsNaN(corSimSum) {
			b.Errorf("CorSim for epoch %d is NaN", epoch)
		}
	}
	corSimAvg := corSimSum / float64(convergenceTestEpochs)
	if corSimAvg < 0.90 {
		b.Errorf("average of CorSim for last %d epochs too low. Want %v, got %v", convergenceTestEpochs, 0.95, corSimAvg)
	}
}
