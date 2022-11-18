// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"runtime"
	"sort"

	"github.com/emer/emergent/timer"
	"github.com/goki/ki/ints"
)

// NetThread specifies how to allocate threads & chunks to each task, and
// manages running those threads (goroutines)
type NetThread struct {
	NThreads  int     `desc:"number of parallel threads to deploy"`
	ChunksPer int     `desc:"number of chunks per thread to use -- each thread greedily grabs chunks"`
	Work      WorkMgr `view:"-" desc:"work manager"`
}

func (th *NetThread) Set(nthr, chk int) {
	maxP := runtime.GOMAXPROCS(0)
	th.NThreads = nthr
	if th.NThreads < 1 {
		th.NThreads = 1
	}
	if th.NThreads > maxP {
		th.NThreads = maxP
	}
	th.ChunksPer = chk
}

func (th *NetThread) Alloc(tot int) {
	th.Work.Alloc(tot, th.NThreads, th.ChunksPer)
}

func (th *NetThread) Run(fun func(st, ed int)) {
	th.Work.Run(th.NThreads, fun)
}

// NetThreads parameterizes how many threads to use for each task
type NetThreads struct {
	Neurons   NetThread `desc:"for basic neuron-level computation -- highly parallel and linear in memory -- should be able to use a lot of threads"`
	SendSpike NetThread `desc:"for sending spikes per neuron -- very large memory footprint through synapses and is very sparse -- suffers from significant bandwidth limitations -- use fewer threads"`
	SynCa     NetThread `desc:"for synaptic-level calcium updating -- very large memory footprint through synapses but linear in order -- use medium number of threads"`
	Learn     NetThread `desc:"for learning: DWt and WtFmDWt -- very large memory footprint through synapses but linear in order -- use medium number of threads"`
}

// SetDefaults sets default allocation of threads based on number of neurons
// and projections.
// According to tests on the LVis model, basically only CycleNeuron scales
// beyond 4 threads..  ChunksPer = 2 is much better than 1, but 3 == 2
func (nt *NetThreads) SetDefaults(nNeurons, nPrjns int) {
	chk := 2
	nt.Neurons.Set(nNeurons/500, chk) // todo: this is all heuristic -- needs tuning!
	nt.SendSpike.Set(nNeurons/10000, chk)
	mxpj := ints.MinInt(nPrjns, 4) // > 4 generally not useful?
	nt.SynCa.Set(mxpj, chk)
	nt.Learn.Set(mxpj, chk)
}

// Set sets allocation of threads manually
func (nt *NetThreads) Set(chk, neurons, sendSpike, synCa, learn int) {
	nt.Neurons.Set(neurons, chk)
	nt.SendSpike.Set(sendSpike, chk)
	nt.SynCa.Set(synCa, chk)
	nt.Learn.Set(learn, chk)
}

// Alloc allocates work managers -- at Build
func (nt *NetThreads) Alloc(nNeurons, nPrjns int) {
	nt.Neurons.Alloc(nNeurons)
	nt.SendSpike.Alloc(nNeurons)
	nt.SynCa.Alloc(nPrjns)
	nt.Learn.Alloc(nPrjns)
	// maxP := runtime.GOMAXPROCS(0)
	// mpi.Printf("Threading: GOMAXPROCS: %d  Chunks: %d  Neurons: %d  SendSpike: %d  SynCa: %d  Learn: %d\n", maxP, nt.Neurons.ChunksPer, nt.Neurons.NThreads, nt.SendSpike.NThreads, nt.SynCa.NThreads, nt.Learn.NThreads)
}

// ThreadsAlloc allocates threads if thread numbers have been updated
// must be called *after* Build
func (nt *NetworkBase) ThreadsAlloc() {
	nt.Threads.Alloc(len(nt.Neurons), len(nt.Prjns))
}

///////////////////////////////////////////////////////////////////////////
//  Compute Function Calling (threading)

const (
	// Thread is named const for actually using threads
	Thread = true
	// NoThread is named const for not using threads
	NoThread = false
	// Wait is named const for waiting for all go routines
	Wait = true
	// NoWait is named const for NOT waiting for all go routines
	NoWait = false
)

// LearnFun applies function of given name to all projections
// using Learn threads (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
func (nt *NetworkBase) LearnFun(fun func(pj AxonPrjn), funame string, thread, wait bool) {
	if thread { // don't bother if not significant to thread
		nt.FunTimerStart(funame)
	}
	if !thread || nt.NThreads <= 1 {
		for _, pj := range nt.Prjns {
			fun(pj)
		}
		if thread {
			nt.FunTimerStop(funame)
		}
		return
	}
	// todo: ignoring wait for now..
	nt.Threads.Learn.Run(func(st, ed int) {
		for pi := st; pi < ed; pi++ {
			pj := nt.Prjns[pi]
			fun(pj)
		}
	})

	if thread {
		nt.FunTimerStop(funame)
	}
}

// SynCaFun applies function of given name to all projections
// using SynCa threads (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
func (nt *NetworkBase) SynCaFun(fun func(pj AxonPrjn), funame string, thread, wait bool) {
	if thread { // don't bother if not significant to thread
		nt.FunTimerStart(funame)
	}
	if !thread || nt.NThreads <= 1 {
		for _, pj := range nt.Prjns {
			fun(pj)
		}
		if thread {
			nt.FunTimerStop(funame)
		}
		return
	}
	// todo: ignoring wait for now..
	nt.Threads.SynCa.Run(func(st, ed int) {
		for pi := st; pi < ed; pi++ {
			pj := nt.Prjns[pi]
			fun(pj)
		}
	})

	if thread {
		nt.FunTimerStop(funame)
	}
}

// note: in practice, all LayerFun are called as NoThread -- probably can get rid of this..

// LayerFun applies function of given name to all layers
// using threading (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
// many layer-level functions are not actually worth threading overhead
// so this should be benchmarked for each case.
func (nt *NetworkBase) LayerFun(fun func(ly AxonLayer), funame string, thread, wait bool) {
	if thread { // don't bother if not significant to thread
		nt.FunTimerStart(funame)
	}
	if !thread || nt.NThreads <= 1 {
		for _, ly := range nt.Layers {
			fun(ly.(AxonLayer))
		}
	} else {
		for _, lyr := range nt.Layers {
			ly := lyr
			nt.WaitGp.Add(1)
			go func() {
				fun(ly.(AxonLayer))
				nt.WaitGp.Done()
			}()
		}
		if wait {
			nt.WaitGp.Wait()
		}
	}
	if thread {
		nt.FunTimerStop(funame)
	}
}

// NeuronFun applies function of given name to all neurons
// using Neurons threading (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
func (nt *NetworkBase) NeuronFun(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string, thread, wait bool) {
	if !thread || nt.NThreads <= 1 {
		for _, layer := range nt.Layers {
			lyr := layer.(AxonLayer)
			lyrNeurons := lyr.AsAxon().Neurons
			for nrnIdx := range lyrNeurons {
				nrn := &lyr.AsAxon().Neurons[nrnIdx]
				fun(lyr, nrnIdx, nrn)
			}
		}
	} else {
		nt.FunTimerStart(funame)
		// todo: ignoring wait for now..
		nt.Threads.Neurons.Run(func(st, ed int) {
			for ni := st; ni < ed; ni++ {
				nrn := &nt.Neurons[ni]
				ly := nt.Layers[nrn.LayIdx].(AxonLayer)
				fun(ly, ni-ly.NeurStartIdx(), nrn)
			}
		})
		nt.FunTimerStop(funame)
	}
}

// SendSpikeFun applies function of given name to all neurons
// using SendSpike threading (go routines) if thread is true and NThreads > 1.
// if wait is true, then it waits until all procs have completed.
func (nt *NetworkBase) SendSpikeFun(fun func(ly AxonLayer, ni int, nrn *Neuron), funame string, thread, wait bool) {
	if thread { // don't bother if not significant to thread
		nt.FunTimerStart(funame)
	}
	if !thread || nt.NThreads <= 1 {
		for ni := range nt.Neurons {
			nrn := &nt.Neurons[ni]
			ly := nt.Layers[nrn.LayIdx].(AxonLayer)
			fun(ly, ni-ly.NeurStartIdx(), nrn)
		}
		if thread {
			nt.FunTimerStop(funame)
		}
		return
	}

	// todo: ignoring wait for now..
	nt.Threads.SendSpike.Run(func(st, ed int) {
		for ni := st; ni < ed; ni++ {
			nrn := &nt.Neurons[ni]
			ly := nt.Layers[nrn.LayIdx].(AxonLayer)
			fun(ly, ni-ly.NeurStartIdx(), nrn)
		}
	})
	if thread {
		nt.FunTimerStop(funame)
	}
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
