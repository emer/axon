// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"

	"cogentcore.org/core/base/timer"
)

// SetNThreads sets number of threads to use for CPU parallel processing.
// pass 0 to use a default heuristic number based on current GOMAXPROCS
// processors and the number of neurons in the network (call after building)
func (nt *Network) SetNThreads(nthr int) {
	maxProcs := runtime.GOMAXPROCS(0) // query GOMAXPROCS
	if nthr <= 0 {
		nneur := nt.Neurons.Len()
		nthr = int(math.Ceil(float64(nneur) / (float64(10000) / float64(nt.MaxData))))
		if nthr < 1 { // shouldn't happen but justin..
			nthr = 1
		}
	}
	nt.NThreads = min(maxProcs, nthr)
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
		pcts[i] = float64(nt.FunTimes[fn].Total) / float64(time.Second)
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
