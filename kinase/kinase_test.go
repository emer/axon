// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"
	"testing"
)

func TestCaBinWts(t *testing.T) {
	spikeBinCycles := 25
	thetaCycles := 200
	plusCycles := 50
	nbins := thetaCycles / spikeBinCycles
	nplus := plusCycles / spikeBinCycles
	cp := make([]float32, nbins)
	cd := make([]float32, nbins)
	CaBinWts(nplus, spikeBinCycles, cp, cd)
	fmt.Println("CaP:", cp)
	fmt.Println("CaD:", cd)
	var cpsum, cdsum float32
	for i := range nbins {
		cpsum += cp[i]
		cdsum += cd[i]
	}
	fmt.Println("CaP Sum:", cpsum, "CaD Sum:", cdsum)
}