// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/emer/emergent/timer"
	"github.com/goki/ki/ints"
)

// Maps the given function across the [0, total) range of items, using
// nThreads goroutines.
func parallelRun(fun func(st, ed int), total int, nThreads int) {
	itemsPerThr := int(math.Ceil(float64(total) / float64(nThreads)))
	waitGroup := sync.WaitGroup{}
	for start := 0; start < total; start += itemsPerThr {
		start := start // capture into loop-local variable for closure
		end := ints.MinInt(start+itemsPerThr, total)
		waitGroup.Add(1) // todo: move out of loop
		go func() {
			fun(start, end)
			waitGroup.Done()
		}()
	}
	waitGroup.Wait()
}

// NetThreads parameterizes how many goroutines to use for each task
type NetThreads struct {
	Neurons   int `desc:"for basic neuron-level computation -- highly parallel and linear in memory -- should be able to use a lot of threads"`
	SendSpike int `desc:"for sending spikes per neuron -- very large memory footprint through synapses and is very sparse -- suffers from significant bandwidth limitations -- use fewer threads"`
	SynCa     int `desc:"for synaptic-level calcium updating -- very large memory footprint through synapses but linear in order -- use medium number of threads"`
}

// SetDefaults uses heuristics to determine the number of goroutines to use
// for each task: Neurons, SendSpike, SynCa.
func (nt *NetThreads) SetDefaults(nNeurons, nPrjns, nLayers int) {
	maxProcs := runtime.GOMAXPROCS(0) // query GOMAXPROCS

	// heuristics
	prjnMinThr := ints.MinInt(ints.MaxInt(nPrjns, 1), 4)
	synHeur := math.Ceil(float64(nNeurons) / float64(1000))
	neuronHeur := math.Ceil(float64(nNeurons) / float64(500))

	if err := nt.Set(
		ints.MinInt(maxProcs, int(neuronHeur)),
		ints.MinInt(maxProcs, int(synHeur)),
		ints.MinInt(maxProcs, int(prjnMinThr)),
	); err != nil {
		log.Fatal(err)
	}
}

// Set sets number of goroutines manually for each task
// This exists mainly for testing, just use SetDefaults in normal use,
// and GOMAXPROCS=1 to force single-threaded operation.
func (nt *NetThreads) Set(neurons, sendSpike, synCa int) error {
	if neurons < 1 || sendSpike < 1 || synCa < 1 {
		return fmt.Errorf("NetThreads: all values must be >= 1, got: %v, %v, %v", neurons, sendSpike, synCa)
	}
	nt.Neurons = neurons
	nt.SendSpike = sendSpike
	nt.SynCa = synCa
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Specialized parallel map functions that respect settings in NetworkBase.Threads

// SynCaFun applies function of given name to all projections, using
// NetThreads.SynCa number of goroutines.
func (nt *NetworkBase) SynCaFun(fun func(pj AxonPrjn), funame string) {
	nt.PrjnMapParallel(fun, funame, nt.Threads.SynCa)
}

// NeuronFun applies function of given name to all neurons, using
// NetThreads.Neurons number of goroutines.
func (nt *NetworkBase) NeuronFun(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string) {
	nt.NeuronMapParallel(fun, funame, nt.Threads.Neurons)
}

// SendSpikeFun applies function of given name to all layers
// using as many goroutines as configured in NetThreads.SendSpike
func (nt *NetworkBase) SendSpikeFun(fun func(ly AxonLayer), funame string) {
	nt.LayerMapParallel(fun, funame, nt.Threads.SendSpike)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Generic parallel map functions

// PrjnMapParallel applies function of given name to all projections
// using nThreads go routines if nThreads > 1, otherwise runs sequentially.
func (nt *NetworkBase) PrjnMapParallel(fun func(prjn AxonPrjn), funame string, nThreads int) {
	nt.FunTimerStart(funame)
	// run single-threaded, skipping the overhead of goroutines
	if nThreads <= 1 {
		nt.PrjnMapSeq(fun, funame)
	} else {
		parallelRun(func(st, ed int) {
			for pi := st; pi < ed; pi++ {
				fun(nt.Prjns[pi])
			}
		}, len(nt.Prjns), nThreads)
	}
	nt.FunTimerStop(funame)
}

// PrjnMapSeq applies function of given name to all projections sequentially.
func (nt *NetworkBase) PrjnMapSeq(fun func(pj AxonPrjn), funame string) {
	nt.FunTimerStart(funame)
	for _, pj := range nt.Prjns {
		fun(pj)
	}
	nt.FunTimerStop(funame)
}

// LayerMapParallel applies function of given name to all layers
// using nThreads go routines if nThreads > 1, otherwise runs sequentially.
func (nt *NetworkBase) LayerMapParallel(fun func(ly AxonLayer), funame string, nThreads int) {
	nt.FunTimerStart(funame)
	if nThreads <= 1 {
		nt.LayerMapSeq(fun, funame)
	} else {
		parallelRun(func(st, ed int) {
			for li := st; li < ed; li++ {
				fun(nt.Layers[li].(AxonLayer))
			}
		}, len(nt.Layers), nThreads)
	}
	nt.FunTimerStop(funame)
}

// LayerMapSeq applies function of given name to all layers sequentially.
func (nt *NetworkBase) LayerMapSeq(fun func(ly AxonLayer), funame string) {
	nt.FunTimerStart(funame)
	for _, ly := range nt.Layers {
		fun(ly.(AxonLayer))
	}
	nt.FunTimerStop(funame)
}

// NeuronMapParallel applies function of given name to all neurons
// using as many go routines as configured in NetThreads.Neurons.
func (nt *NetworkBase) NeuronMapParallel(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string, nThreads int) {
	nt.FunTimerStart(funame)
	if nThreads <= 1 {
		nt.NeuronMapSequential(fun, funame)
	} else {
		parallelRun(func(st, ed int) {
			for ni := st; ni < ed; ni++ {
				nrn := &nt.Neurons[ni]
				ly := nt.Layers[nrn.LayIdx].(AxonLayer)
				fun(ly, ni-ly.NeurStartIdx(), nrn)
			}
		}, len(nt.Neurons), nThreads)
	}
	nt.FunTimerStop(funame)
}

// NeuronMapSequential applies function of given name to all neurons sequentially.
func (nt *NetworkBase) NeuronMapSequential(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string) {
	nt.FunTimerStart(funame)
	for _, layer := range nt.Layers {
		lyr := layer.(AxonLayer)
		lyrNeurons := lyr.AsAxon().Neurons
		for nrnIdx := range lyrNeurons {
			nrn := &lyrNeurons[nrnIdx]
			fun(lyr, nrnIdx, nrn)
		}
	}
	nt.FunTimerStop(funame)
}

//////////////////////////////////////////////////////////////
// Timing reports

// TimerReport reports the amount of time spent in each function, and in each thread
func (nt *NetworkBase) TimerReport() {
	fmt.Printf("TimerReport: %v\n", nt.Nm)
	fmt.Printf("\t%13s \t%7s\t%7s\n", "Function Name", "Secs", "Pct")
	nfn := len(nt.FunTimes)
	fnms := make([]string, nfn)
	idx := 0
	for k := range nt.FunTimes {
		fnms[idx] = k
		idx++
	}
	sort.StringSlice(fnms).Sort()
	pcts := make([]float64, nfn)
	tot := 0.0
	for i, fn := range fnms {
		pcts[i] = nt.FunTimes[fn].TotalSecs()
		tot += pcts[i]
	}
	for i, fn := range fnms {
		fmt.Printf("\t%13s \t%7.3f\t%7.1f\n", fn, pcts[i], 100*(pcts[i]/tot))
	}
	fmt.Printf("\t%13s \t%7.3f\n", "Total", tot)
}

// FunTimerStart starts function timer for given function name -- ensures creation of timer
func (nt *NetworkBase) FunTimerStart(fun string) {
	if !nt.RecFunTimes {
		return
	}
	ft, ok := nt.FunTimes[fun]
	if !ok {
		ft = &timer.Time{}
		nt.FunTimes[fun] = ft
	}
	ft.Start()
}

// FunTimerStop stops function timer -- timer must already exist
func (nt *NetworkBase) FunTimerStop(fun string) {
	if !nt.RecFunTimes {
		return
	}
	ft := nt.FunTimes[fun]
	ft.Stop()
}
