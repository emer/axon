// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
	"testing"

	"cogentcore.org/lab/base/randx"
)

func TestMake(t *testing.T) {
	ev := &ConSatEnv{}
	ev.Defaults()
	ev.RunRandSeed = 137
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(0)+1))
	// Debug = true

	ev.InitLines()
}
