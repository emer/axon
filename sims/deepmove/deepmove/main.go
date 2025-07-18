// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/axon/v2/sims/deepmove"
	"github.com/emer/emergent/v2/egui"
)

func main() { egui.Run[deepmove.Sim, deepmove.Config]() }
