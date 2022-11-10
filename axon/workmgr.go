// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"sync"

	"github.com/goki/ki/atomctr"
)

// GreedyChunks selects a greedy chunk running mode -- else just spawn routines willy-nilly
var GreedyChunks = true

type StartEnd struct {
	Start int
	End   int
}

type WorkMgr struct {
	Cur    atomctr.Ctr
	Chunks []StartEnd
	Wait   sync.WaitGroup
}

func (wm *WorkMgr) Alloc(tot, nThr, chunksPer int) {
	if nThr < 1 {
		nThr = 1
	}
	if tot <= 0 {
		tot = 1
	}
	nc := nThr * chunksPer
	if nc > tot {
		nc = tot
	}
	nPer := tot / nc
	for {
		if nc*nPer < tot {
			nPer++
		} else {
			break
		}
	}
	wm.Chunks = make([]StartEnd, nc, nc)
	si := 0
	for i := range wm.Chunks {
		wm.Chunks[i].Start = si
		ed := si + nPer
		if ed > tot {
			ed = tot
		}
		wm.Chunks[i].End = ed
		si = ed
	}
	// fmt.Printf("tot: %d  nThr: %d  per: %d  cks: %v\n", tot, nThr, chunksPer, wm.Chunks)
}

func (wm *WorkMgr) Run(nthr int, fun func(st, ed int)) {
	nc := len(wm.Chunks)
	if nthr >= nc {
		for i := 0; i < nc; i++ {
			ch := wm.Chunks[i]
			fun(ch.Start, ch.End)
		}
		return
	}
	if GreedyChunks {
		wm.Cur.Set(-1)
		for i := 0; i < nthr; i++ {
			wm.Wait.Add(1)
			go func() {
				for {
					c := wm.Cur.Inc()
					if int(c) >= nc {
						wm.Wait.Done()
						return
					}
					ch := wm.Chunks[c]
					fun(ch.Start, ch.End)
				}
			}()
		}
		wm.Wait.Wait()
		return
	}
	// todo: this is NOT working at all! no idea why not.
	for i := 0; i < nc; i++ {
		c := i
		ch := wm.Chunks[c]
		wm.Wait.Add(1)
		go func() {
			fun(ch.Start, ch.End)
			wm.Wait.Done()
		}()
	}
	wm.Wait.Wait()
}
