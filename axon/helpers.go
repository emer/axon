// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/env"
	"github.com/emer/empi/mpi"
	"github.com/goki/gi/gi"
)

////////////////////////////////////////////////////
// Misc

// ToggleLayersOff can be used to disable layers in a Network, for example if you are doing an ablation study.
func ToggleLayersOff(net *Network, layerNames []string, off bool) {
	for _, lnm := range layerNames {
		lyi := net.LayByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

// EnvApplyInputs applies input patterns from given env.Env environment
// to Input and Target layer types, assuming that env provides State
// with the same names as the layers.
// If these assumptions don't fit, use a separate method.
func EnvApplyInputs(net *Network, ev env.Env) {
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByType(InputLayer, TargetLayer)
	for _, lnm := range lays {
		ly := net.LayByName(lnm)
		pats := ev.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

/////////////////////////////////////////////
// Weights files

// WeightsFileName returns default current weights file name,
// using train run and epoch counters from looper
// and the RunName string identifying tag, parameters and starting run,
func WeightsFileName(net *Network, ctrString, runName string) string {
	return net.Name() + "_" + runName + "_" + ctrString + ".wts.gz"
}

// SaveWeights saves network weights to filename with WeightsFileName information
// to identify the weights.
// only for 0 rank MPI if running mpi
// Returns the name of the file saved to, or empty if not saved.
func SaveWeights(net *Network, ctrString, runName string) string {
	if mpi.WorldRank() > 0 {
		return ""
	}
	fnm := WeightsFileName(net, ctrString, runName)
	fmt.Printf("Saving Weights to: %s\n", fnm)
	net.SaveWtsJSON(gi.FileName(fnm))
	return fnm
}

// SaveWeightsIfArgSet saves network weights if the "wts" arg has been set to true.
// uses WeightsFileName information to identify the weights.
// only for 0 rank MPI if running mpi
// Returns the name of the file saved to, or empty if not saved.
func SaveWeightsIfArgSet(net *Network, args *ecmd.Args, ctrString, runName string) string {
	if args.Bool("wts") {
		return SaveWeights(net, ctrString, runName)
	}
	return ""
}
