// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

import "github.com/emer/emergent/v2/paths"

// Paths holds all the special projections.
type Paths struct {
	// Standard feedforward topographic projection, recv = 1/2 send size
	PT4x4Skp2 *paths.PoolTile

	// Reciprocal
	PT4x4Skp2Recip *paths.PoolTile

	// sparser skip 2 -- no overlap
	PT2x2Skp2 *paths.PoolTile

	// Reciprocal
	PT2x2Skp2Recip *paths.PoolTile

	// Standard same-to-same size topographic projection
	PT3x3Skp1 *paths.PoolTile

	// Sigmoidal topographic projection used in LIP saccade remapping layers
	PTSigTopo *paths.PoolTile

	// Gaussian topographic projection used in LIP saccade remapping layers
	PTGaussTopo *paths.PoolTile
}

func (pj *Paths) Defaults() {
	pj.PT4x4Skp2 = paths.NewPoolTile()
	pj.PT4x4Skp2.Size.Set(4, 4)
	pj.PT4x4Skp2.Skip.Set(2, 2)
	pj.PT4x4Skp2.Start.Set(-1, -1)
	pj.PT4x4Skp2.TopoRange.Min = 0.8
	// but using a symmetric scale range .8 - 1.2 seems like it might be good -- otherwise
	// weights are systematicaly smaller.
	// note: gauss defaults on
	// pj.PT4x4Skp2.GaussFull.DefNoWrap()
	// pj.PT4x4Skp2.GaussInPool.DefNoWrap()

	pj.PT4x4Skp2Recip = paths.NewPoolTile()
	pj.PT4x4Skp2Recip.Size.Set(4, 4)
	pj.PT4x4Skp2Recip.Skip.Set(2, 2)
	pj.PT4x4Skp2Recip.Start.Set(-1, -1)
	pj.PT4x4Skp2Recip.TopoRange.Min = 0.8 // note: none of these make a very big diff
	pj.PT4x4Skp2Recip.Recip = true

	pj.PT2x2Skp2 = paths.NewPoolTile()
	pj.PT2x2Skp2.Size.Set(2, 2)
	pj.PT2x2Skp2.Skip.Set(2, 2)
	pj.PT2x2Skp2.Start.Set(0, 0)
	pj.PT2x2Skp2.TopoRange.Min = 0.8

	pj.PT2x2Skp2Recip = paths.NewPoolTile()
	pj.PT2x2Skp2Recip.Size.Set(2, 2)
	pj.PT2x2Skp2Recip.Skip.Set(2, 2)
	pj.PT2x2Skp2Recip.Start.Set(0, 0)
	pj.PT2x2Skp2Recip.TopoRange.Min = 0.8
	pj.PT2x2Skp2Recip.Recip = true

	pj.PT3x3Skp1 = paths.NewPoolTile()
	pj.PT3x3Skp1.Size.Set(3, 3)
	pj.PT3x3Skp1.Skip.Set(1, 1)
	pj.PT3x3Skp1.Start.Set(-1, -1)
	pj.PT3x3Skp1.TopoRange.Min = 0.8 // note: none of these make a very big diff

	pj.PTSigTopo = paths.NewPoolTile()
	pj.PTSigTopo.GaussOff()
	pj.PTSigTopo.Size.Set(1, 1)
	pj.PTSigTopo.Skip.Set(0, 0)
	pj.PTSigTopo.Start.Set(0, 0)
	pj.PTSigTopo.TopoRange.Min = 0.6
	pj.PTSigTopo.SigFull.On = true
	pj.PTSigTopo.SigFull.Gain = 0.05
	pj.PTSigTopo.SigFull.CtrMove = 0.5

	pj.PTGaussTopo = paths.NewPoolTile()
	pj.PTGaussTopo.Size.Set(1, 1)
	pj.PTGaussTopo.Skip.Set(0, 0)
	pj.PTGaussTopo.Start.Set(0, 0)
	pj.PTGaussTopo.TopoRange.Min = 0.6
	pj.PTGaussTopo.GaussInPool.On = false // Full only
	pj.PTGaussTopo.GaussFull.Sigma = 0.6
	pj.PTGaussTopo.GaussFull.Wrap = true
	pj.PTGaussTopo.GaussFull.CtrMove = 1
}
