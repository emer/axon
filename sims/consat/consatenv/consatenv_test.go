// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import "testing"

func TestBruteForce(t *testing.T) {
	ev := &ConSatEnv{}
	ev.Defaults()
	ev.NClauses = 2
	ev.Config(1, 0, 137)
	ev.Init(0)
	// ev.Step()
}
