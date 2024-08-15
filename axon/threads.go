// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"cogentcore.org/core/base/timer"
)

// Maps the given function across the [0, total) range of items, using
// nThreads goroutines.
func ParallelRun(fun func(st, ed uint32), total uint32, nThreads int) {
	itemsPerThr := uint32(math.Ceil(float64(total) / float64(nThreads)))
	wait := sync.WaitGroup{}
	for start := uint32(0); start < total; start += itemsPerThr {
		start := start // capture into loop-local variable for closure
		end := start + itemsPerThr
		if end > total {
			end = total
		}
		wait.Add(1) // todo: move out of loop
		go func() {
			fun(start, end)
			wait.Done()
		}()
	}
	wait.Wait()
}

// SetNThreads sets number of threads to use for CPU parallel processing.
// pass 0 to use a default heuristic number based on current GOMAXPROCS
// processors and the number of neurons in the network (call after building)
func (nt *Network) SetNThreads(nthr int) {
	maxProcs := runtime.GOMAXPROCS(0) // query GOMAXPROCS
	if nthr <= 0 {
		nneur := len(nt.Neurons)
		nthr = int(math.Ceil(float64(nneur) / (float64(10000) / float64(nt.MaxData))))
		if nthr < 1 { // shouldn't happen but justin..
			nthr = 1
		}
	}
	nt.NThreads = min(maxProcs, nthr)
}

// PathMapSeq applies function of given name to all pathways sequentially.
func (nt *Network) PathMapSeq(fun func(pj *Path), funame string) {
	nt.FunTimerStart(funame)
	for _, pj := range nt.Paths {
		fun(pj)
	}
	nt.FunTimerStop(funame)
}

// LayerMapSeq applies function of given name to all layers sequentially.
func (nt *Network) LayerMapSeq(fun func(ly *Layer), funame string) {
	nt.FunTimerStart(funame)
	for _, ly := range nt.Layers {
		fun(ly)
	}
	nt.FunTimerStop(funame)
}

// LayerMapPar applies function of given name to all layers
// using as many go routines as configured in NetThreads.Neurons.
func (nt *Network) LayerMapPar(fun func(ly *Layer), funame string) {
	if nt.NThreads <= 1 {
		nt.LayerMapSeq(fun, funame)
	} else {
		nt.FunTimerStart(funame)
		ParallelRun(func(st, ed uint32) {
			for li := st; li < ed; li++ {
				ly := nt.Layers[li]
				fun(ly)
			}
		}, uint32(len(nt.Layers)), nt.NThreads)
		nt.FunTimerStop(funame)
	}
}

// NeuronMapSeq applies function of given name to all neurons sequentially.
func (nt *Network) NeuronMapSeq(ctx *Context, fun func(ly *Layer, ni uint32), funame string) {
	nt.FunTimerStart(funame)
	for _, ly := range nt.Layers {
		for lni := uint32(0); lni < ly.NNeurons; lni++ {
			ni := ly.NeurStIndex + lni
			fun(ly, ni)
		}
	}
	nt.FunTimerStop(funame)
}

// NeuronMapPar applies function of given name to all neurons
// using as many go routines as configured in NetThreads.Neurons.
func (nt *Network) NeuronMapPar(ctx *Context, fun func(ly *Layer, ni uint32), funame string) {
	if nt.NThreads <= 1 {
		nt.NeuronMapSeq(ctx, fun, funame)
	} else {
		nt.FunTimerStart(funame)
		ParallelRun(func(st, ed uint32) {
			for ni := st; ni < ed; ni++ {
				li := NrnI(ctx, ni, NrnLayIndex)
				ly := nt.Layers[li]
				fun(ly, ni)
			}
		}, nt.NNeurons, nt.NThreads)
		nt.FunTimerStop(funame)
	}
}

//////////////////////////////////////////////////////////////
// Timing reports

// TimerReport reports the amount of time spent in each function, and in each thread
func (nt *Network) TimerReport() {
	fmt.Printf("TimerReport: %v  %d threads\n", nt.Name, nt.NThreads)
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
		pcts[i] = float64(nt.FunTimes[fn].Total)
		tot += pcts[i]
	}
	for i, fn := range fnms {
		fmt.Printf("\t%13s \t%7.3f\t%7.1f\n", fn, pcts[i], 100*(pcts[i]/tot))
	}
	fmt.Printf("\t%13s \t%7.3f\n", "Total", tot)
}

// FunTimerStart starts function timer for given function name -- ensures creation of timer
func (nt *Network) FunTimerStart(fun string) {
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
func (nt *Network) FunTimerStop(fun string) {
	if !nt.RecFunTimes {
		return
	}
	ft := nt.FunTimes[fun]
	ft.Stop()
}
