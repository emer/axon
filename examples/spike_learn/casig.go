// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CaSigState is entire intracellular Ca-driven signaling state
type CaSigState struct {
	CaMKII CaMKIIState `desc:"CaMKII state"`
	CaN    CaNState    `desc:"CaN = calcineurin state"`
	PKA    PKAState    `desc:"PKA = protein kinase A"`
	PP1    PP1State    `desc:"PP1 = protein phosphatase 1"`
	PP2A   float32     `desc:"PP2A = protein phosphatase 2A, only in Cyt"`
}
