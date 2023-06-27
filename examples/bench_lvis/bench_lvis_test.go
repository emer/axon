package bench

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/stretchr/testify/require"
)

var gpu = flag.Bool("gpu", false, "whether to run gpu or not")
var maxProcs = flag.Int("maxProcs", 0, "GOMAXPROCS value to set -- 0 = use current default -- better to set threads instead, as long as it is < GOMAXPROCS")
var threads = flag.Int("threads", 2, "number of goroutines for parallel processing -- 2, 4 give good results typically")
var ndata = flag.Int("ndata", 1, "number of inputs to run in parallel")
var numEpochs = flag.Int("epochs", 1, "number of epochs to run")
var numPats = flag.Int("pats", 10, "number of patterns per epoch")
var verbose = flag.Bool("verbose", true, "if false, only report the final time")
var inputNeurs = flag.Int("inputNeurs", 5, "input neurons per pool")
var inputPools = flag.Int("inputPools", 16, "key parameter: number of input pools, also determines number of hidden pools")
var pathways = flag.Int("pathways", 4, "number of separate pathways for different resolution / receptive field")
var hiddenNeurs = flag.Int("hiddenNeurs", 8, "key parameter: number of neurons per hidden pool -- 8 and 10 in basic lvis")
var outputDim = flag.Int("outputDim", 10, "output dimension")

func BenchmarkBenchNetFull(b *testing.B) {
	inputShape := [2]int{*inputNeurs * *inputPools, *inputNeurs * *inputPools}
	outputShape := [2]int{*outputDim, *outputDim}

	if *maxProcs > 0 {
		runtime.GOMAXPROCS(*maxProcs)
	}

	// if *verbose {
	fmt.Printf("Running bench with: %d Threads, %d MaxData, %d epochs, %d pats, (%d, %d) input, (%d) hidden, (%d, %d) output\n",
		*threads, *ndata, *numEpochs, *numPats, inputShape[0], inputShape[1], *hiddenNeurs, outputShape[0], outputShape[1])
	// }

	rand.Seed(42)

	ctx := axon.NewContext()
	net := &axon.Network{}
	ConfigNet(ctx, net, *inputNeurs, *inputPools, *pathways, *hiddenNeurs, *outputDim, *threads, *ndata, *verbose)
	// if *verbose {
	log.Println(net.SizeReport(false))
	// }

	pats := &etable.Table{}
	ConfigPats(pats, *numPats, inputShape, outputShape)

	inLay := net.LayerByName("V1_0").(*axon.Layer)
	outLay := net.LayerByName("Output").(*axon.Layer)

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)

	// todo: is the input shape actually correct for a 4D layer?
	require.Equal(b, inLay.Shp.Len(), inPats.Len() / *numPats)
	require.Equal(b, outLay.Shp.Len(), outPats.Len() / *numPats)

	epcLog := &etable.Table{}
	ConfigEpcLog(epcLog)

	TrainNet(ctx, net, pats, epcLog, *pathways, *numEpochs, *verbose, *gpu)
}

// TestGPUSynCa is a key test for large memory allocations
// as in the SynapseCas variables at high ndata levels (8, 16)
func TestGPUSynCa(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	rand.Seed(42)

	ctx := axon.NewContext()
	net := &axon.Network{}
	ConfigNet(ctx, net, *inputNeurs, *inputPools, *pathways, *hiddenNeurs, *outputDim, *threads, *ndata, *verbose)
	log.Println(net.SizeReport(false))

	// vgpu.Debug = true // definitely enable if failing!!
	net.ConfigGPUnoGUI(ctx)

	// on mac, only works up to ndata = 6 -- 7 fails
	fmt.Printf("ndata: %d   floats per: %X  banks: %d\n", ctx.NetIdxs.NData, ctx.NetIdxs.GPUMaxBuffFloats, ctx.NetIdxs.GPUSynCaBanks)

	passed := net.GPU.TestSynCa()
	if !passed {
		t.Errorf("GPU SynCa write failed\n")
	}
}
