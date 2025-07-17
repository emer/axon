// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/axon/v2/sims/ra25"
	"github.com/emer/emergent/v2/egui"
)

func main() { egui.Run[ra25.Sim, ra25.Config]() }
