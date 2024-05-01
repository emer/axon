// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/prjn"
)

// BLANovelPrjn connects all other pools to the first, Novelty, pool in a BLA layer.
// This allows the known US representations to specifically inhibit the novelty pool.
type BLANovelPrjn struct {
}

func NewBLANovelPrjn() *BLANovelPrjn {
	return &BLANovelPrjn{}
}

func (ot *BLANovelPrjn) Name() string {
	return "BLANovelPrjn"
}

func (ot *BLANovelPrjn) Connect(send, recv *tensor.Shape, same bool) (sendn, recvn *tensor.Int32, cons *tensor.Bits) {
	sendn, recvn, cons = prjn.NewTensors(send, recv)
	sNtot := send.Len()
	// rNtot := recv.Len()
	sNp := send.DimSize(0) * send.DimSize(1)
	sNu := send.DimSize(2) * send.DimSize(3)
	rNu := recv.DimSize(2) * recv.DimSize(3)
	rnv := recvn.Values
	snv := sendn.Values
	npl := sNp
	rpi := 0
	for spi := 1; spi < npl; spi++ {
		for rui := 0; rui < rNu; rui++ {
			ri := rpi*rNu + rui
			for sui := 0; sui < sNu; sui++ {
				si := spi*sNu + sui
				off := ri*sNtot + si
				cons.Values.Set(off, true)
				rnv[ri] = int32(sNu * (npl - 1))
				snv[si] = int32(rNu)
			}
		}
	}
	return
}
