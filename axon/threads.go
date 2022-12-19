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

// NetThread specifies how to allocate threads & chunks to each task, and
// manages running those threads (goroutines)
type NetThread struct {
	nThreads  int `desc:"number of parallel threads to deploy"`
	WaitGroup sync.WaitGroup
}

// Set the number of threads to use for this task to a given value >= 1
func (th *NetThread) Set(nthr int) error {
	if nthr < 1 {
		th.nThreads = 1
		return fmt.Errorf("NetThread.NThreads must be > 0")
	}
	th.nThreads = nthr
	return nil
}

// Maps the given function across the [0, total) range of items, using
// the number of goroutines specified in the NetThread.
func (th *NetThread) Run(fun func(st, ed int), total int) {
	itemsPerThr := int(math.Ceil(float64(total) / float64(th.nThreads)))
	for start := 0; start < total; start += itemsPerThr {
		start := start // be extra sure with closure
		end := ints.MinInt(start+itemsPerThr, total)
		th.WaitGroup.Add(1) // todo: move out of loop
		go func(st, ed int) {
			fun(st, ed)
			th.WaitGroup.Done()
		}(start, end)
	}
	th.WaitGroup.Wait()
}

// NetThreads parameterizes how many goroutines to use for each task
type NetThreads struct {
	Neurons   NetThread `desc:"for basic neuron-level computation -- highly parallel and linear in memory -- should be able to use a lot of threads"`
	SendSpike NetThread `desc:"for sending spikes per neuron -- very large memory footprint through synapses and is very sparse -- suffers from significant bandwidth limitations -- use fewer threads"`
	SynCa     NetThread `desc:"for synaptic-level calcium updating -- very large memory footprint through synapses but linear in order -- use medium number of threads"`
	Prjn      NetThread `desc:"for projection-level learning: DWt and WtFmDWt -- very large memory footprint through synapses but linear in order -- use medium number of threads"`
	Layer     NetThread `desc:"for layer-level computation"`
}

// SetDefaults sets default allocation of threads based on number of neurons
// and projections.
// According to tests on the LVis model, basically only CycleNeuron scales
// beyond 4 threads..
func (nt *NetThreads) SetDefaults(nNeurons, nPrjns, nLayers int) {
	maxProcs := runtime.GOMAXPROCS(0)      // query GOMAXPROCS
	prjnMinThr := ints.MinInt(maxProcs, 4) // heuristic

	neuronHeur := math.Ceil(float64(nNeurons) / float64(500))
	synHeur := math.Ceil(float64(nNeurons) / float64(1000))

	if err := nt.Set(
		ints.MinInt(maxProcs, int(neuronHeur)),
		ints.MinInt(maxProcs, int(synHeur)),
		ints.MinInt(nPrjns, prjnMinThr),
		ints.MinInt(nPrjns, prjnMinThr),
		1,
	); err != nil {
		log.Fatal(err)
	}
}

// Set sets number of goroutines manually for each task
func (nt *NetThreads) Set(neurons, sendSpike, synCa, prjn, layer int) error {
	if err := nt.Neurons.Set(neurons); err != nil {
		return fmt.Errorf("NetThreads.Neurons: %v", err)
	}
	if err := nt.SendSpike.Set(sendSpike); err != nil {
		return fmt.Errorf("NetThreads.SendSpike: %v", err)
	}
	if err := nt.SynCa.Set(synCa); err != nil {
		return fmt.Errorf("NetThreads.SynCa: %v", err)
	}
	if err := nt.Prjn.Set(prjn); err != nil {
		return fmt.Errorf("NetThreads.Prjn: %v", err)
	}
	if err := nt.Layer.Set(layer); err != nil {
		return fmt.Errorf("NetThreads.Layer: %v", err)
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////
//  Compute Function Calling (threading)

// PrjnFun applies function of given name to all projections
// using Prjn threads (go routines) if thread is true and NThreads > 1.
func (nt *NetworkBase) PrjnFun(fun func(pj AxonPrjn), funame string) {
	nt.FunTimerStart(funame)
	// run single-threaded, skipping the overhead of goroutines
	if nt.Threads.Prjn.nThreads <= 1 {
		for _, pj := range nt.Prjns {
			fun(pj)
		}
	} else {
		nt.Threads.Prjn.Run(func(st, ed int) {
			for pi := st; pi < ed; pi++ {
				pj := nt.Prjns[pi]
				fun(pj)
			}
		}, len(nt.Prjns))
	}
	nt.FunTimerStop(funame)
}

// SynCaFun applies function of given name to all projections
// using SynCa threads (go routines) if thread is true and NThreads > 1.
// TODO: Merge with PrjnFun
func (nt *NetworkBase) SynCaFun(fun func(pj AxonPrjn), funame string) {
	nt.FunTimerStart(funame)
	// run single-threaded, skipping the overhead of goroutines
	if nt.Threads.SynCa.nThreads <= 1 {
		for _, pj := range nt.Prjns {
			fun(pj)
		}
	} else {
		nt.Threads.SynCa.Run(func(st, ed int) {
			for pi := st; pi < ed; pi++ {
				pj := nt.Prjns[pi]
				fun(pj)
			}
		}, len(nt.Prjns))
	}
	nt.FunTimerStop(funame)
}

// note: in practice, all LayerFun are called as NoThread -- probably can get rid of this..

// LayerFun applies function of given name to all layers
// using threading (go routines) if thread is true and NThreads > 1.
// many layer-level functions are not actually worth threading overhead
// so this should be benchmarked for each case.
func (nt *NetworkBase) LayerFun(fun func(ly AxonLayer), funame string) {
	nt.FunTimerStart(funame)
	if nt.Threads.Layer.nThreads <= 1 {
		for _, ly := range nt.Layers {
			fun(ly.(AxonLayer))
		}
	} else {
		nt.Threads.Layer.Run(func(st, ed int) {
			for li := st; li < ed; li++ {
				ly := nt.Layers[li].(AxonLayer)
				fun(ly)
			}
		}, len(nt.Layers))
	}
	nt.FunTimerStop(funame)
}

// NeuronFun applies function of given name to all neurons
// using Neurons threading (go routines) if thread is true and NThreads > 1.
func (nt *NetworkBase) NeuronFun(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string) {
	nt.FunTimerStart(funame)
	if nt.Threads.Neurons.nThreads <= 1 {
		for _, layer := range nt.Layers {
			lyr := layer.(AxonLayer)
			lyrNeurons := lyr.AsAxon().Neurons
			for nrnIdx := range lyrNeurons {
				nrn := &lyrNeurons[nrnIdx]
				fun(lyr, nrnIdx, nrn)
			}
		}
	} else {
		nt.Threads.Neurons.Run(func(st, ed int) {
			for ni := st; ni < ed; ni++ {
				nrn := &nt.Neurons[ni]
				ly := nt.Layers[nrn.LayIdx].(AxonLayer)
				fun(ly, ni-ly.NeurStartIdx(), nrn)
			}
		}, len(nt.Neurons))
	}
	nt.FunTimerStop(funame)
}

// SendSpikeFun applies function of given name to all neurons
// using SendSpike threading (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
// todo: merge with NeuronFun
func (nt *NetworkBase) SendSpikeFun(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string) {
	nt.FunTimerStart(funame)
	if nt.Threads.SendSpike.nThreads <= 1 {
		for _, layer := range nt.Layers {
			lyr := layer.(AxonLayer)
			lyrNeurons := lyr.AsAxon().Neurons
			for nrnIdx := range lyrNeurons {
				nrn := &lyrNeurons[nrnIdx] // loops over all neurons, same as NeuronFun
				fun(lyr, nrnIdx, nrn)
			}
		}
	} else {
		nt.Threads.SendSpike.Run(func(st, ed int) {
			for ni := st; ni < ed; ni++ {
				nrn := &nt.Neurons[ni]
				ly := nt.Layers[nrn.LayIdx].(AxonLayer)
				fun(ly, ni-ly.NeurStartIdx(), nrn)
			}
		}, len(nt.Neurons))
	}
	nt.FunTimerStop(funame)
}

//////////////////////////////////////////////////////////////
// Timing reports -- could move all this to NetThreads

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
