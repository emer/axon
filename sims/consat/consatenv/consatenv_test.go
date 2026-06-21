// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import "testing"

func TestMake(t *testing.T) {
	ev := &ConSatEnv{}
	ev.Defaults()
	ev.NAry = 5
	ev.NVars = 5
	ev.NConstraints = 4 // 1 extra for none, improves overall "pure" score
	ev.RelationsPer = 3
	ev.Config(1, 0, 137) // 137 is good
	ev.Init(0)
	ev.MakeConstraints()
	// ev.Step()
}
