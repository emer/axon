//go:build !race && not

package main

import (
	"fmt"
	"os"
	"testing"
)

// default number of threads for CPU
var defNThread = 8

// RunBench runs the boa benchmark
func RunBench(b *testing.B, gpu bool, ndata, nthread int) {
	fmt.Printf("bench: gpu: %v  ndata: %d  nthread: %d\n", gpu, ndata, nthread)
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config.Run.GPU = gpu
	sim.Config.Run.NData = ndata
	sim.Config.Run.NThreads = nthread
	sim.Config.Run.NRuns = 1
	sim.Config.Run.NEpochs = 1
	sim.Config.Log.Run = false
	sim.Config.Log.Epoch = false

	sim.ConfigAll()
	sim.RunNoGUI()
}

// GPU

func BenchmarkGPUnData1(b *testing.B) {
	if os.Getenv("TEST_GPU") != "true" {
		b.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunBench(b, true, 1, defNThread)
}

func BenchmarkGPUnData2(b *testing.B) {
	if os.Getenv("TEST_GPU") != "true" {
		b.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunBench(b, true, 2, defNThread)
}

func BenchmarkGPUnData4(b *testing.B) {
	if os.Getenv("TEST_GPU") != "true" {
		b.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunBench(b, true, 4, defNThread)
}

func BenchmarkGPUnData8(b *testing.B) {
	if os.Getenv("TEST_GPU") != "true" {
		b.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunBench(b, true, 8, defNThread)
}

func BenchmarkGPUnData16(b *testing.B) {
	if os.Getenv("TEST_GPU") != "true" {
		b.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunBench(b, true, 16, defNThread)
}

// CPU

func BenchmarkCPUnData1(b *testing.B) {
	RunBench(b, false, 1, defNThread)
}

func BenchmarkCPUnData2(b *testing.B) {
	RunBench(b, false, 2, defNThread)
}

func BenchmarkCPUnData4(b *testing.B) {
	RunBench(b, false, 4, defNThread)
}

func BenchmarkCPUnData8(b *testing.B) {
	RunBench(b, false, 8, defNThread)
}

func BenchmarkCPUnData16(b *testing.B) {
	RunBench(b, false, 16, defNThread)
}

// CPU Nthreads

func BenchmarkCPUnData1Nthr1(b *testing.B) {
	RunBench(b, false, 1, 1)
}

func BenchmarkCPUnData1Nthr2(b *testing.B) {
	RunBench(b, false, 1, 2)
}

func BenchmarkCPUnData1Nthr4(b *testing.B) {
	RunBench(b, false, 1, 4)
}

func BenchmarkCPUnData1Nthr8(b *testing.B) {
	RunBench(b, false, 1, 8)
}

func BenchmarkCPUnData16Nthr1(b *testing.B) {
	RunBench(b, false, 16, 1)
}

func BenchmarkCPUnData16Nthr2(b *testing.B) {
	RunBench(b, false, 16, 2)
}

func BenchmarkCPUnData16Nthr4(b *testing.B) {
	RunBench(b, false, 16, 4)
}

func BenchmarkCPUnData16Nthr8(b *testing.B) {
	RunBench(b, false, 16, 8)
}
