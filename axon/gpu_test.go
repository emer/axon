// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"testing"
)

func gpuSynDataNs(t *testing.T, synN, maxData, maxTh int) (nCmd, nPer, nLast int) {
	maxThSyn := maxTh / maxData

	nCmd = synN / maxThSyn
	if synN%maxThSyn > 0 {
		nCmd++
	}
	nPer = synN / nCmd
	nLast = synN - (nCmd * nPer)
	// sanity checks:
	if nPer*maxData > maxTh {
		t.Errorf("axon.GPU.SynDataNs allocated too many nPer threads!")
	}
	if nLast*maxData > maxTh {
		t.Errorf("axon.GPU.SynDataNs allocated too many nLast threads. maxData: %d  nCmd: %d  synN: %X  nPer: %X  nLast: %X MaxComputeWorkGroupCount1D: %X", maxData, nCmd, synN, nPer, nLast, maxTh)
	}
	// fmt.Printf("axon.GPU.SynDataNs allocated: maxData: %d  nCmd: %d  synN: %X  nPer: %X  nLast: %X MaxComputeWorkGroupCount1D: %X\n", maxData, nCmd, synN, nPer, nLast, maxTh)
	return
}

func TestGPUSynDataNs(t *testing.T) {
	// 16  nCmd: 94  synN: 5D464  nPer: FE0  nLast: 1004 MaxComputeWorkGroupCount1D: 65535
	gpuSynDataNs(t, 0x5D464, 16, 65535)
}
