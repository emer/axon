// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lvis

import "github.com/emer/emergent/v2/paths"

// Paths holds all the special projections.
type Paths struct {

	// Standard feedforward topographic projection, recv = 1/2 send size
	PT4x4Skp2 *paths.PoolTile

	// Reciprocal
	PT4x4Skp2Recip *paths.PoolTile

	// Standard feedforward topographic projection, recv = 1/2 send size
	PT4x4Skp2Sub2 *paths.PoolTileSub

	// Reciprocal
	PT4x4Skp2Sub2Recip *paths.PoolTileSub

	// Standard feedforward topographic projection, recv = 1/2 send size
	PT4x4Skp2Sub2Send *paths.PoolTileSub

	// Standard feedforward topographic projection, recv = 1/2 send size
	PT4x4Skp2Sub2SendRecip *paths.PoolTileSub

	// same-size paths
	PT2x2Skp1 *paths.PoolTile

	// same-size paths reciprocal
	PT2x2Skp1Recip *paths.PoolTile

	// same-size paths
	PT2x2Skp1Sub2 *paths.PoolTileSub

	// same-size paths reciprocal
	PT2x2Skp1Sub2Recip *paths.PoolTileSub

	// same-size paths
	PT2x2Skp1Sub2Send *paths.PoolTileSub

	// same-size paths reciprocal
	PT2x2Skp1Sub2SendRecip *paths.PoolTileSub

	// lateral inhib projection
	PT2x2Skp2 *paths.PoolTileSub

	// for V4 <-> TEO
	PT4x4Skp0 *paths.PoolTile

	// for V4 <-> TEO
	PT4x4Skp0Recip *paths.PoolTile

	// for V4 <-> TEO
	PT4x4Skp0Sub2 *paths.PoolTileSub

	// for V4 <-> TEO
	PT4x4Skp0Sub2Recip *paths.PoolTileSub

	// for TE <-> TEO
	PT1x1Skp0 *paths.PoolTile

	// for TE <-> TEO
	PT1x1Skp0Recip *paths.PoolTile

	// lateral inhibitory connectivity for subpools
	PT6x6Skp2Lat *paths.PoolTileSub
}

func (pj *Paths) Defaults() {
	pj.PT4x4Skp2 = paths.NewPoolTile()
	pj.PT4x4Skp2.Size.Set(4, 4)
	pj.PT4x4Skp2.Skip.Set(2, 2)
	pj.PT4x4Skp2.Start.Set(-1, -1)
	pj.PT4x4Skp2.TopoRange.Min = 0.8
	pj.PT4x4Skp2Recip = paths.NewPoolTileRecip(pj.PT4x4Skp2)

	pj.PT4x4Skp2Sub2 = paths.NewPoolTileSub()
	pj.PT4x4Skp2Sub2.Size.Set(4, 4)
	pj.PT4x4Skp2Sub2.Skip.Set(2, 2)
	pj.PT4x4Skp2Sub2.Start.Set(-1, -1)
	pj.PT4x4Skp2Sub2.Subs.Set(2, 2)
	pj.PT4x4Skp2Sub2.TopoRange.Min = 0.8
	pj.PT4x4Skp2Sub2Recip = paths.NewPoolTileSubRecip(pj.PT4x4Skp2Sub2)

	pj.PT4x4Skp2Sub2Send = paths.NewPoolTileSub()
	*pj.PT4x4Skp2Sub2Send = *pj.PT4x4Skp2Sub2
	pj.PT4x4Skp2Sub2Send.SendSubs = true
	pj.PT4x4Skp2Sub2SendRecip = paths.NewPoolTileSubRecip(pj.PT4x4Skp2Sub2Send)

	pj.PT2x2Skp1 = paths.NewPoolTile()
	pj.PT2x2Skp1.Size.Set(2, 2)
	pj.PT2x2Skp1.Skip.Set(1, 1)
	pj.PT2x2Skp1.Start.Set(0, 0)
	pj.PT2x2Skp1.TopoRange.Min = 0.8
	pj.PT2x2Skp1Recip = paths.NewPoolTileRecip(pj.PT2x2Skp1)

	pj.PT2x2Skp1Sub2 = paths.NewPoolTileSub()
	pj.PT2x2Skp1Sub2.Size.Set(2, 2)
	pj.PT2x2Skp1Sub2.Skip.Set(1, 1)
	pj.PT2x2Skp1Sub2.Start.Set(0, 0)
	pj.PT2x2Skp1Sub2.Subs.Set(2, 2)
	pj.PT2x2Skp1Sub2.TopoRange.Min = 0.8

	pj.PT2x2Skp1Sub2Recip = paths.NewPoolTileSubRecip(pj.PT2x2Skp1Sub2)

	pj.PT2x2Skp1Sub2Send = paths.NewPoolTileSub()
	pj.PT2x2Skp1Sub2Send.Size.Set(2, 2)
	pj.PT2x2Skp1Sub2Send.Skip.Set(1, 1)
	pj.PT2x2Skp1Sub2Send.Start.Set(0, 0)
	pj.PT2x2Skp1Sub2Send.Subs.Set(2, 2)
	pj.PT2x2Skp1Sub2Send.SendSubs = true
	pj.PT2x2Skp1Sub2Send.TopoRange.Min = 0.8

	pj.PT2x2Skp1Sub2SendRecip = paths.NewPoolTileSub()
	*pj.PT2x2Skp1Sub2SendRecip = *pj.PT2x2Skp1Sub2Send
	pj.PT2x2Skp1Sub2SendRecip.Recip = true

	pj.PT2x2Skp2 = paths.NewPoolTileSub()
	pj.PT2x2Skp2.Size.Set(2, 2)
	pj.PT2x2Skp2.Skip.Set(2, 2)
	pj.PT2x2Skp2.Start.Set(0, 0)
	pj.PT2x2Skp2.Subs.Set(2, 2)

	pj.PT4x4Skp0 = paths.NewPoolTile()
	pj.PT4x4Skp0.Size.Set(4, 4)
	pj.PT4x4Skp0.Skip.Set(0, 0)
	pj.PT4x4Skp0.Start.Set(0, 0)
	pj.PT4x4Skp0.GaussFull.Sigma = 1.5
	pj.PT4x4Skp0.GaussInPool.Sigma = 1.5
	pj.PT4x4Skp0.TopoRange.Min = 0.8
	pj.PT4x4Skp0Recip = paths.NewPoolTileRecip(pj.PT4x4Skp0)

	pj.PT4x4Skp0Sub2 = paths.NewPoolTileSub()
	pj.PT4x4Skp0Sub2.Size.Set(4, 4)
	pj.PT4x4Skp0Sub2.Skip.Set(0, 0)
	pj.PT4x4Skp0Sub2.Start.Set(0, 0)
	pj.PT4x4Skp0Sub2.Subs.Set(2, 2)
	pj.PT4x4Skp0Sub2.SendSubs = true
	pj.PT4x4Skp0Sub2.GaussFull.Sigma = 1.5
	pj.PT4x4Skp0Sub2.GaussInPool.Sigma = 1.5
	pj.PT4x4Skp0Sub2.TopoRange.Min = 0.8
	pj.PT4x4Skp0Sub2Recip = paths.NewPoolTileSubRecip(pj.PT4x4Skp0Sub2)

	pj.PT1x1Skp0 = paths.NewPoolTile()
	pj.PT1x1Skp0.Size.Set(1, 1)
	pj.PT1x1Skp0.Skip.Set(0, 0)
	pj.PT1x1Skp0.Start.Set(0, 0)
	pj.PT1x1Skp0.GaussFull.Sigma = 1.5
	pj.PT1x1Skp0.GaussInPool.Sigma = 1.5
	pj.PT1x1Skp0.TopoRange.Min = 0.8
	pj.PT1x1Skp0Recip = paths.NewPoolTileRecip(pj.PT1x1Skp0)

	pj.PT6x6Skp2Lat = paths.NewPoolTileSub()
	pj.PT6x6Skp2Lat.Size.Set(6, 6)
	pj.PT6x6Skp2Lat.Skip.Set(2, 2)
	pj.PT6x6Skp2Lat.Start.Set(-2, -2)
	pj.PT6x6Skp2Lat.Subs.Set(2, 2)
	pj.PT6x6Skp2Lat.TopoRange.Min = 0.8
}
