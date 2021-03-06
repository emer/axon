// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/axon/chans"
	"github.com/emer/axon/deep"
	"github.com/emer/axon/pcore"
)

var (
	// NeuronVarsAll is the agate collection of all neuron-level vars (deep, chans, pcore)
	NeuronVarsAll []string
)

func init() {
	dln := len(deep.NeuronVarsAll)
	gln := len(chans.NeuronVars)
	pln := len(pcore.NeuronVars)
	NeuronVarsAll = make([]string, dln+gln+pln)
	copy(NeuronVarsAll, deep.NeuronVarsAll)
	copy(NeuronVarsAll[dln:], chans.NeuronVars)
	copy(NeuronVarsAll[dln+gln:], pcore.NeuronVars)
}
