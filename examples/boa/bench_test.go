//go:build !race

package main

import (
	"fmt"
	"testing"
)

// RunBench runs the boa benchmark
func RunBench(b *testing.B, gpu bool, ndata int) {
	fmt.Printf("bench: gpu: %v  ndata: %d\n", gpu, ndata)
	sim := &Sim{}

	sim.New()
	sim.Sim.NData = ndata

	sim.Config()

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 1)
	sim.Args.SetBool("epclog", false) // set to true to debug runs
	sim.Args.SetBool("runlog", false)
	sim.Args.SetBool("bench", true)
	sim.Args.SetBool("gpu", gpu)

	sim.RunNoGUI()
}

func BenchmarkGPUnData1(b *testing.B) {
	RunBench(b, true, 1)
}

func BenchmarkGPUnData2(b *testing.B) {
	RunBench(b, true, 2)
}

func BenchmarkGPUnData4(b *testing.B) {
	RunBench(b, true, 4)
}

func BenchmarkGPUnData8(b *testing.B) {
	RunBench(b, true, 8)
}

func BenchmarkGPUnData16(b *testing.B) {
	RunBench(b, true, 16)
}

func BenchmarkCPUnData1(b *testing.B) {
	RunBench(b, false, 1)
}

func BenchmarkCPUnData2(b *testing.B) {
	RunBench(b, false, 2)
}

func BenchmarkCPUnData4(b *testing.B) {
	RunBench(b, false, 4)
}

func BenchmarkCPUnData8(b *testing.B) {
	RunBench(b, false, 8)
}

func BenchmarkCPUnData16(b *testing.B) {
	RunBench(b, false, 16)
}
