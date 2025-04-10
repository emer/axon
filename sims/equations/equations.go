// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// equations provides an interactive exploration of the various equations
// underlying the axon models, largely from the chans collection of channels.
package main

//go:generate core generate -add-types

import (
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/chans/chanplots"
	"github.com/emer/emergent/v2/egui"
)

func main() {
	root, _ := tensorfs.NewDir("Root")
	pl := &Plots{}

	pl.GUI.MakeBody(nil, pl, root, "Equations", "Axon Equations", "Equations used in Axon")
	pl.GUI.FinalizeGUI(false)
	pl.Config(root)
	pl.GUI.UpdateFiles()
	pl.GUI.Body.RunMainWindow()
}

type Plots struct {
	GUI egui.GUI `display:"-"`

	// AK is an A-type K channel, which is voltage gated with maximal
	// activation around -37 mV.  It has two state variables, M (v-gated opening)
	// and H (v-gated closing), which integrate with fast and slow time constants,
	// respectively.  H relatively quickly hits an asymptotic level of inactivation
	// for sustained activity patterns.
	// It is particularly important for counteracting the excitatory effects of
	// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
	// See AKsParams for a much simpler version that works fine when full AP-like spikes are
	// not simulated, as in our standard axon models.
	AK chanplots.AKPlot `new-window:"+" display:"no-inline"`

	// GABA-B is an inhibitory channel activated by the usual GABA inhibitory neurotransmitter,
	// which is coupled to the GIRK G-protein coupled inwardly rectifying potassium (K) channel.
	// It is ubiquitous in the brain, and critical for stability of spiking patterns over time in axon.
	// The inward rectification is caused by a Mg+ ion block *from the inside* of the neuron,
	// which means that these channels are most open when the neuron is hyperpolarized (inactive),
	// and thus it serves to keep inactive neurons inactive. Based on Thomson & Destexhe (1999).
	GABAB chanplots.GababPlot `new-window:"+" display:"no-inline"`

	// Kir is the kIR potassium inwardly rectifying current,
	// based on the equations from Lindroos et al (2018).
	// The conductance is highest at low membrane potentials.
	Kir chanplots.KirPlot `new-window:"+" display:"no-inline"`

	// Mahp implements an M-type medium afterhyperpolarizing (mAHP) channel,
	// where m also stands for muscarinic due to the ACh inactivation of this channel.
	// It has a slow activation and deactivation time constant, and opens at a lowish
	// membrane potential.
	// There is one gating variable n updated over time with a tau that is also voltage dependent.
	// The infinite-time value of n is voltage dependent according to a logistic function
	// of the membrane potential, centered at Voff with slope Vslope.
	Mahp chanplots.MahpPlot `new-window:"+" display:"no-inline"`

	// NMDA implements NMDA dynamics, based on Jahr & Stevens (1990) equations
	// which are widely used in models, from Brunel & Wang (2001) to Sanders et al. (2013).
	// The overall conductance is a function of a voltage-dependent postsynaptic factor based
	// on Mg ion blockage, and presynaptic Glu-based opening, which in a simple model just
	// increments
	NMDA chanplots.NMDAPlot `new-window:"+" display:"no-inline"`

	// Sahp implements a slow afterhyperpolarizing (sAHP) channel,
	// It has a slowly accumulating calcium value, aggregated at the
	// theta cycle level, that then drives the logistic gating function,
	// so that it only activates after a significant accumulation.
	// After which point it decays.
	// For the theta-cycle updating, the normal m-type tau is all within
	// the scope of a single theta cycle, so we just omit the time integration
	// of the n gating value, but tau is computed in any case.
	Sahp chanplots.SahpPlot `new-window:"+" display:"no-inline"`

	// SKCa describes the small-conductance calcium-activated potassium channel,
	// activated by intracellular stores in a way that drives pauses in firing,
	// and can require inactivity to recharge the Ca available for release.
	// These intracellular stores can release quickly, have a slow decay once released,
	// and the stores can take a while to rebuild, leading to rapidly triggered,
	// long-lasting pauses that don't recur until stores have rebuilt, which is the
	// observed pattern of firing of STNp pausing neurons.
	// CaIn = intracellular stores available for release; CaR = released amount from stores
	// CaM = K channel conductance gating factor driven by CaR binding,
	// computed using the Hill equations described in Fujita et al (2012), Gunay et al (2008)
	// (also Muddapu & Chakravarthy, 2021): X^h / (X^h + C50^h) where h ~= 4 (hard coded)
	SKCa chanplots.SKCaPlot `new-window:"+" display:"no-inline"`

	// VGCC plots the standard L-type voltage gated Ca channel.
	// All functions based on Urakubo et al (2008).
	VGCC chanplots.VGCCPlot `new-window:"+" display:"no-inline"`

	// SynCa plots synaptic calcium according to the kinase calcium dynamics.
	SynCa chanplots.SynCaPlot `new-window:"+" display:"no-inline"`
}

func (pl *Plots) Config(root *tensorfs.Node) {
	pl.AK.Config(root, pl.GUI.Tabs)
	pl.GABAB.Config(root, pl.GUI.Tabs)
	pl.Kir.Config(root, pl.GUI.Tabs)
	pl.Mahp.Config(root, pl.GUI.Tabs)
	pl.NMDA.Config(root, pl.GUI.Tabs)
	pl.Sahp.Config(root, pl.GUI.Tabs)
	pl.SKCa.Config(root, pl.GUI.Tabs)
	pl.VGCC.Config(root, pl.GUI.Tabs)
	pl.SynCa.Config(root, pl.GUI.Tabs)
}
