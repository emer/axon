// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	_ "cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/emer"
)

//gosl:start

// NeuronFlags are bit-flags encoding relevant binary state for neurons
type NeuronFlags int32 //enums:enum

// The neuron flags
const (
	// NeuronOff flag indicates that this neuron has been turned off (i.e., lesioned).
	NeuronOff NeuronFlags = 1

	// NeuronHasExt means the neuron has external input in its Ext field.
	NeuronHasExt NeuronFlags = 2

	// NeuronHasTarg means the neuron has external target input in its Target field.
	NeuronHasTarg NeuronFlags = 4

	// NeuronHasCmpr means the neuron has external comparison input in its Target field.
	// Used for computing comparison statistics but does not drive neural activity ever.
	NeuronHasCmpr NeuronFlags = 8
)

// NeuronVars are the neuron variables representing current active state,
// specific to each input data state.
// See NeuronAvgVars for vars shared across data.
type NeuronVars int32 //enums:enum

const (

	//////// Spiking, Activation

	// Spike is whether neuron has spiked or not on this cycle (0 or 1).
	Spike NeuronVars = iota

	// Spiked is 1 if neuron has spiked within the last 10 cycles (msecs),
	// corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise.
	// Useful for visualization and computing activity levels in terms of
	// average spiked levels.
	Spiked

	// Act is rate-coded activation value reflecting instantaneous estimated rate
	// of spiking, based on 1 / ISIAvg. It is integrated over time for ActInt
	// which is then used for performance statistics and layer average activations, etc.
	// Should not be used for learning or other computations: just for stats / display.
	Act

	// ActInt is integrated running-average activation value computed from Act
	// with time constant Act.Dt.IntTau, to produce a longer-term integrated value
	// reflecting the overall activation state across the ThetaCycle time scale,
	// as the overall response of network to current input state. This is copied
	// to ActM and ActP at the ends of the minus and plus phases, respectively,
	// and used in computing some performance-level statistics (based on ActM).
	// Should not be used for learning or other computations.
	ActInt

	//////// Major conductances, Vm

	// Ge is total excitatory conductance, including all forms of excitation
	// (e.g., NMDA). Does *not* include the Gbar.E factor.
	Ge

	// Gi is total inhibitory synaptic conductance, i.e., the net inhibitory input
	// to the neuron. Does *not* include the Gbar.I factor.
	Gi

	// Gk is total potassium conductance, typically reflecting sodium-gated potassium
	// currents involved in adaptation effects. Does *not* include the Gbar.K factor.
	Gk

	// Inet is net current produced by all channels, which drives update of Vm.
	Inet

	// Vm is the membrane potential at the cell body, which integrates Inet current
	// over time, and drives spiking at the axon initial segment of the neuron.
	Vm

	// VmDend is the dendritic membrane potential, which has a slower time constant
	// than Vm and is not subject to the VmR reset after spiking.
	VmDend

	// ISI is the current inter-spike-interval, which counts up since last spike.
	// Starts at -1 when initialized.
	ISI

	// ISIAvg is the average inter-spike-interval, i.e., the average time interval
	// between spikes, integrated with ISITau rate constant (relatively fast) to
	// capture something close to an instantaneous spiking rate.  Starts at -1 when
	// initialized, and goes to -2 after first spike, and is only valid after the
	// second spike post-initialization.
	ISIAvg

	// Ext is the external input: drives activation of unit from outside influences
	// (e.g., sensory input).
	Ext

	// Target is the target value: drives learning to produce this activation value.
	Target

	//////// Spike-driven calcium for stats

	// CaM is the spike-driven calcium trace at the neuron level, which then drives
	// longer time-integrated variables: [CaP] and [CaD]. These variables are used
	// for statistics and display to capture spiking activity at different timescales.
	// They fluctuate more than [Act] and [ActInt], but are closer to the biological
	// variables driving learning. CaM is the exponential integration of SpikeG * Spike
	// using the MTau time constant (typically 5), and simulates a calmodulin (CaM)
	// like signal, at an abstract level.
	CaM

	// CaP is the continuous cascaded integration of [CaM] using the PTau time constant
	// (typically 40), representing a neuron-level, purely spiking version of the plus,
	// LTP direction of weight change in the Kinase learning rule, dependent on CaMKII.
	// This is not used for learning (see [LearnCaP]), but instead for statistics
	// as a representation of recent activity.
	CaP

	// CaD is the continuous cascaded integration [CaP] using the DTau time constant
	// (typically 40), representing a neuron-level, purely spiking version of the minus,
	// LTD direction of weight change in the Kinase learning rule, dependent on DAPK1.
	// This is not used for learning (see [LearnCaD]), but instead for statistics
	// as a representation of trial-level activity.
	CaD

	// CaDPrev is the final [CaD] activation state at the end of previous theta cycle.
	// This is used for specialized learning mechanisms that operate on delayed
	// sending activations.
	CaDPrev

	//////// Calcium for learning

	// CaSyn is the neuron-level integration of spike-driven calcium, used to approximate
	// synaptic calcium influx as a product of sender and receiver neuron CaSyn values,
	// which are integrated separately because it is computationally much more efficient.
	// CaSyn enters into a Sender * Receiver product at each synapse to give the effective
	// credit assignment factor for learning.
	// This value is driven directly by spikes, with an exponential integration time
	// constant of 30 msec (default), which captures the coincidence window for pre*post
	// firing on NMDA receptor opening. The neuron [CaBins] values record the temporal
	// trajectory of CaSyn over the course of the theta cycle window, and then the
	// pre*post product is integrated over these bins at the synaptic level.
	CaSyn

	// LearnCa is the receiving neuron calcium signal, which is integrated up to
	// [LearnCaP] and [LearnCaD], the difference of which is the temporal error
	// component of the kinase cortical learning rule.
	// LearnCa combines NMDA via [NmdaCa] and spiking-driven VGCC [VgccCaInt] calcium
	// sources. The NMDA signal reflects both sending and receiving activity, while the
	// VGCC signal is purely receiver spiking, and a balance of both works best.
	LearnCa

	// LearnCaM is the integrated [LearnCa] at the MTau timescale (typically 5),
	// simulating a calmodulin (CaM) like signal, which then drives [LearnCaP],
	// and [LearnCaD] for the delta signal for error-driven learning.
	LearnCaM

	// LearnCaP is the cascaded integration of [LearnCaM] using the PTau time constant
	// (typically 40), representing the plus, LTP direction of weight change,
	// capturing the function of CaMKII in the Kinase learning rule.
	LearnCaP

	// LearnCaD is the cascaded integration of [LearnCaP] using the DTau time constant
	// (typically 40), representing the minus, LTD direction of weight change,
	// capturing the function of DAPK1 in the Kinase learning rule.
	LearnCaD

	// CaDiff is difference between [LearnCaP] - [LearnCaD]. This is the error
	// signal that drives error-driven learning.
	CaDiff

	//////// Learning Timing

	// GaM is first-level integration of all input conductances g_a,
	// which then drives longer time-integrated variables: [GaP] and [GaD].
	// These variables are used for timing of learning based on bursts of activity
	// change over time: at the minus and plus phases.
	GaM

	// GaP is the continuous cascaded integration of [GaM] using the PTau time constant
	// (typically 40), representing a neuron-level, all-conductance-based version
	// of the plus, LTP direction of weight change in the Kinase learning rule.
	GaP

	// GaD is the continuous cascaded integration of [GaP] using the DTau time constant
	// (typically 40), representing a neuron-level, all-conductance-based version
	// of the minus, LTD direction of weight change in the Kinase learning rule.
	GaD

	// TimeDiff is the running time-average of |P - D| (absolute value),
	// used for determining the timing of learning in terms of onsets of peaks.
	// See [MinusPeak] and [PlusPeak]. The GaP - GaD value is much smoother and
	// more reliable than LearnCaP - LearnCaD (i.e., CaDiff).
	TimeDiff

	// TimeDiffPeak is the value of the current peak (local maximum) of [TimeDiff].
	TimeDiffPeak

	// TimeDiffPeakCyc is the absolute cycle where [TimeDiffPeak] occurred.
	TimeDiffPeakCyc

	// MinusPeak is the value of the first, minus-phase peak of [TimeDiffAvg],
	// which occurs when new input drives the fast integral to diverge from slow.
	MinusPeak

	// MinusPeakCyc is the absolute cycle where [MinusPeak] occurred.
	MinusPeakCyc

	// PlusPeak is the value of the second, plus-phase peak of [TimeDiffAvg],
	// which occurs when an outcome causes fast integral to diverge from slow.
	PlusPeak

	// PlusPeakCyc is the absolute cycle where [PlusPeak] occurred.
	PlusPeakCyc

	// LearnDiff is the actual difference signal that drives learning, which is
	// computed from [CaDiff] for neocortical neurons, but specifically at the
	// point of learning (LearnNow), based on [PlusPeakCyc].
	// It is cleared at the start of a new learning window.
	LearnDiff

	// LearnNow is activated at the moment when the receiving neuron is learning,
	// based on timing computed from [MinusPeak] and [PlusPeak].
	// See [LearnTimingParams].
	LearnNow

	// RLRate is recv-unit based learning rate multiplier, reflecting the sigmoid
	// derivative computed from [CaD] of recv unit, and the normalized difference
	// (CaP - CaD) / MAX(CaP - CaD).
	RLRate

	// ETrace is the eligibility trace for this neuron.
	ETrace

	// ETrace is the learning factor for the eligibility trace for this neuron.
	ETraceLearn

	//////// NMDA channels

	// GnmdaSyn is the integrated NMDA synaptic current on the receiving neuron.
	// It adds GeRaw and decays with a time constant.
	GnmdaSyn

	// Gnmda is the net postsynaptic (receiving) NMDA conductance,
	// after Mg V-gating and Gbar. This is added directly to Ge as it has the same
	// reversal potential.
	Gnmda

	// GnmdaLrn is learning version of integrated NMDA recv synaptic current.
	// It adds [GeRaw] and decays with a time constant. This drives [NmdaCa] that
	// then drives [LearnCa] for learning.
	GnmdaLrn

	// GnmdaMaint is net postsynaptic maintenance NMDA conductance, computed from
	// [GMaintSyn] and [GMaintRaw], after Mg V-gating and Gbar. This is added directly
	// to Ge as it has the same reversal potential.
	GnmdaMaint

	// NmdaCa is NMDA calcium computed from GnmdaLrn, drives learning via CaM.
	NmdaCa

	////////  VGCC voltage gated calcium channels

	// Gvgcc is conductance (via Ca) for VGCC voltage gated calcium channels.
	Gvgcc

	// VgccM is activation gate of VGCC channels.
	VgccM

	// VgccH inactivation gate of VGCC channels.
	VgccH

	// VgccCa is the instantaneous VGCC calcium flux: can be driven by spiking
	// or directly from Gvgcc.
	VgccCa

	// VgccCaInt is the time-integrated VGCC calcium flux. This is actually
	// what drives learning.
	VgccCaInt

	// Burst is the layer 5 IB intrinsic bursting neural activation value,
	// computed by thresholding the [CaP] value in Super superficial layers.
	Burst

	// BurstPrv is previous Burst bursting activation from prior time step.
	// Used for context-based learning.
	BurstPrv

	// CtxtGe is context (temporally delayed) excitatory conductance,
	// driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGe

	// CtxtGeRaw is raw update of context (temporally delayed) excitatory
	// conductance, driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGeRaw

	// CtxtGeOrig is original CtxtGe value prior to any decay factor.
	// Updates at end of plus phase.
	CtxtGeOrig

	//////// GABA-B channels

	// GgabaB is net GABA-B conductance, after Vm gating and Gk + Gbase.
	// Applies to Gk, not Gi, for GIRK, with .1 reversal potential.
	GgabaB

	// GababM is the GABA-B / GIRK activation, which is a time-integrated value
	// with rise and decay time constants.
	GababM

	// GababX is GABA-B / GIRK internal drive variable. This gets the raw
	// activation and decays.
	GababX

	//////// SST somatostatin inhibition factors

	// Gak is the conductance of A-type K potassium channels.
	Gak

	// SSGiDend is the amount of SST+ somatostatin positive slow spiking
	// inhibition applied to dendritic Vm (VmDend).
	SSGiDend

	// GknaMed is the conductance of sodium-gated potassium channel (KNa)
	// medium dynamics (Slick), which produces accommodation / adaptation.
	GknaMed

	// GknaSlow is the conductance of sodium-gated potassium channel (KNa)
	// slow dynamics (Slack), which produces accommodation / adaptation.
	GknaSlow

	// Gkir is the conductance of the potassium (K) inwardly rectifying channel,
	// which is strongest at low membrane potentials.  Can be modulated by DA.
	Gkir

	// KirM is the Kir potassium (K) inwardly rectifying gating value.
	KirM

	////////  SKCa small conductance calcium-gated potassium channels

	// Gsk is Calcium-gated potassium channel conductance as a function
	// of Gbar * SKCaM.
	Gsk

	// SKCaIn is intracellular calcium store level, available to be released
	// with spiking as SKCaR, which can bind to SKCa receptors and drive K
	// current. replenishment is a function of spiking activity being below
	// a threshold.
	SKCaIn

	// SKCaR is the released amount of intracellular calcium, from SKCaIn,
	// as a function of spiking events. This can bind to SKCa channels and
	// drive K currents.
	SKCaR

	// SKCaM is the Calcium-gated potassium channel gating factor, driven by
	// SKCaR via a Hill equation as in chans.SKPCaParams.
	SKCaM

	///////// AHP channels: Mahp, Sahp, Gkna

	// Gmahp is medium time scale AHP conductance.
	Gmahp

	// MahpN is accumulating voltage-gated gating value for the medium time
	// scale AHP.
	MahpN

	// Gsahp is slow time scale AHP conductance.
	Gsahp

	// SahpCa is slowly accumulating calcium value that drives the slow AHP.
	SahpCa

	// SahpN is the sAHP gating value.
	SahpN

	//////// Stats, aggregate values

	// ActM is ActInt activation state at end of third quarter, representing
	// the posterior-cortical minus phase activation. This is used for statistics
	// and monitoring network performance.
	// Should not be used for learning or other computations.
	ActM

	// ActP is ActInt activation state at end of fourth quarter, representing
	// the posterior-cortical plus_phase activation. This is used for statistics
	// and monitoring network performance.
	// Should not be used for learning or other computations.
	ActP

	// Beta1 is the activation state at the first beta cycle within current
	// state processing window (i.e., at 50 msec), as saved by Beta1() function.
	// Used for example in hippocampus for CA3, CA1 learning.
	Beta1

	// Beta2 is the activation state at the second beta cycle within current
	// state processing window (i.e., at 100 msec), as saved by Beta2() function.
	// Used for example in hippocampus for CA3, CA1 learning.
	Beta2

	// CaPMax is the maximum [CaP] across one theta cycle time window
	// (max of CaPMaxCa). It is used for specialized algorithms that have more
	// phasic behavior within a single trial, e.g., BG Matrix layer gating.
	// Also useful for visualization of peak activity of neurons.
	CaPMax

	// CaPMaxCa is the Ca integrated like [CaP] but only starting at
	// the MaxCycStart cycle, to prevent inclusion of carryover spiking from
	// prior theta cycle trial. The PTau time constant otherwise results in
	// significant carryover. This is the input to CaPMax.
	CaPMaxCa

	//////// Noise

	// GeNoise is integrated noise excitatory conductance, added into Ge.
	GeNoise

	// GeNoiseP is accumulating poisson probability factor for driving excitatory
	// noise spiking. Multiply times uniform random deviate at each time step,
	// until it gets below the target threshold based on poisson lambda as function
	//  of noise firing rate.
	GeNoiseP

	// GiNoise is integrated noise inhibotyr conductance, added into Gi.
	GiNoise

	// GiNoiseP is accumulating poisson probability factor for driving inhibitory
	// noise spiking. Multiply times uniform random deviate at each time step,
	// until it gets below the target threshold based on poisson lambda as a function
	// of noise firing rate.
	GiNoiseP

	//////// Ge, Gi integration

	// GeExt is extra excitatory conductance added to Ge, from Ext input, GeCtxt etc.
	GeExt

	// GeRaw is the raw excitatory conductance (net input) received from
	// senders = current raw spiking drive.
	GeRaw

	// GeSyn is the time-integrated total excitatory (AMPA) synaptic conductance,
	// with an instantaneous rise time from each spike (in GeRaw) and
	// exponential decay with Dt.GeTau, aggregated over pathways.
	// Does *not* include Gbar.E.
	GeSyn

	// GiRaw is the raw inhibitory conductance (net input) received from senders
	// = current raw spiking drive.
	GiRaw

	// GiSyn is time-integrated total inhibitory synaptic conductance, with an
	// instantaneous rise time from each spike (in GiRaw) and exponential decay
	// with Dt.GiTau, aggregated over pathways -- does *not* include Gbar.I.
	// This is added with computed FFFB inhibition to get the full inhibition in Gi.
	GiSyn

	// GeInt is integrated running-average activation value computed from Ge
	// with time constant Act.Dt.IntTau, to produce a longer-term integrated value
	// reflecting the overall Ge level across the ThetaCycle time scale (Ge itself
	// fluctuates considerably). This is useful for stats to set strength of
	// connections etc to get neurons into right range of overall excitatory drive.
	GeInt

	// GeIntNorm is normalized GeInt value (divided by the layer maximum).
	// This is used for learning in layers that require learning on
	//  subthreshold activity.
	GeIntNorm

	// GiInt is integrated running-average activation value computed from GiSyn
	// with time constant Act.Dt.IntTau, to produce a longer-term integrated
	// value reflecting the overall synaptic Gi level across the ThetaCycle
	// time scale (Gi itself fluctuates considerably). Useful for stats to set
	// strength of connections etc to get neurons into right range of overall
	// inhibitory drive.
	GiInt

	// GModRaw is raw modulatory conductance, received from GType
	// = ModulatoryG pathways.
	GModRaw

	// GModSyn is syn integrated modulatory conductance, received from GType
	// = ModulatoryG pathways.
	GModSyn

	// SMaintP is accumulating poisson probability factor for driving
	// self-maintenance by simulating a population of mutually interconnected neurons.
	// Multiply times uniform random deviate at each time step, until it gets below
	// the target threshold based on poisson lambda based on accumulating self maint
	// factor.
	SMaintP

	// GMaintRaw is raw maintenance conductance, received from GType
	// = MaintG pathways.
	GMaintRaw

	// GMaintSyn is syn integrated maintenance conductance, integrated
	// using MaintNMDA params.
	GMaintSyn

	// NeurFlags are bit flags for binary state variables, which are converted
	// to / from uint32. These need to be in Vars because they can be
	// differential per data (for ext inputs) and are writable (indexes are read only).
	NeurFlags

	// CaBins is a vector of values starting here, with aggregated [CaSyn] values
	// in time bins of [Context.CaBinCycles] across the theta cycle,
	// for computing synaptic calcium efficiently. Each bin = Sum(CaSyn) / CaBinCycles.
	// Total number of bins = [Context.ThetaCycles] / CaBinCycles.
	// Synaptic calcium is integrated from sender * receiver CaBins values,
	// with weights for CaP vs CaD that reflect their faster vs. slower time constants,
	// respectively. CaD is used for the credit assignment factor, while CaP - CaD is
	// used directly for error-driven learning at Target layers.
	CaBins
)

// NeuronAvgVars are mostly neuron variables involved in longer-term average activity
// which is aggregated over time and not specific to each input data state,
// along with any other state that is not input data specific.
type NeuronAvgVars int32 //enums:enum

const (
	// ActAvg is average activation (of minus phase activation state)
	// over long time intervals (time constant = Dt.LongAvgTau).
	// Useful for finding hog units and seeing overall distribution of activation.
	ActAvg NeuronAvgVars = iota

	// AvgPct is ActAvg as a proportion of overall layer activation.
	// This is used for synaptic scaling to match TrgAvg activation,
	// updated at SlowInterval intervals.
	AvgPct

	// TrgAvg is neuron's target average activation as a proportion
	// of overall layer activation, assigned during weight initialization,
	// driving synaptic scaling relative to AvgPct.
	TrgAvg

	// DTrgAvg is change in neuron's target average activation as a result
	// of unit-wise error gradient. Acts like a bias weight.
	// MPI needs to share these across processors.
	DTrgAvg

	// AvgDif is AvgPct - TrgAvg, i.e., the error in overall activity level
	// relative to set point for this neuron, which drives synaptic scaling.
	// Updated at SlowInterval intervals.
	AvgDif

	// GeBase is baseline level of Ge, added to GeRaw, for intrinsic excitability.
	GeBase

	// GiBase is baseline level of Gi, added to GiRaw, for intrinsic excitability.
	GiBase
)

// NeuronIndexVars are neuron-level indexes used to access layers and pools
// from the individual neuron level.
type NeuronIndexVars int32 //enums:enum

const (
	// NrnNeurIndex is the index of this neuron within its owning layer.
	NrnNeurIndex NeuronIndexVars = iota

	// NrnLayIndex is the index of the layer that this neuron belongs to,
	// needed for neuron-level parallel code.
	NrnLayIndex

	// NrnSubPool is the index of the sub-level inhibitory pool for this neuron
	// (only for 4D shapes, the pool (unit-group / hypercolumn) structure level).
	// Indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	NrnSubPool
)

//gosl:end

var VarCategories = []emer.VarCategory{
	{"Act", "basic activation variables, including conductances, current, Vm, spiking"},
	{"Learn", "calcium-based learning variables and other related learning factors"},
	{"Excite", "excitatory channels including NMDA, Vgcc and other excitatory inputs"},
	{"Inhib", "inhibitory channels including GABA inhibition, after hyperpolarization (AHP) and other K channels"},
	{"Stats", "statistics and aggregate values"},
	{"Gmisc", "more detailed conductance (G) variables for integration and other computational values"},
	{"Avg", "longer-term average variables and homeostatic regulation"},
	{"Spikes", "Binned spike counts used for learning"},
	{"Wts", "weights and other synaptic-level variables"},
}

// NeuronVarProps has display properties for neuron variables.
var NeuronVarProps = map[string]string{

	//////// Spiking, Activation, Major conductances, Vm

	"Spike":  `cat:"Act"`,
	"Spiked": `cat:"Act"`,
	"Act":    `cat:"Act"`,
	"ActInt": `cat:"Act"`,
	"Ge":     `cat:"Act" range:"2"`,
	"Gi":     `cat:"Act" auto-scale:"+"`,
	"Gk":     `cat:"Act" auto-scale:"+"`,
	"Inet":   `cat:"Act" auto-scale:"+"`,
	"Vm":     `cat:"Act" min:"-100" max:"0"`,
	"VmDend": `cat:"Act" min:"-100" max:"0"`,
	"ISI":    `cat:"Act" auto-scale:"+"`,
	"ISIAvg": `cat:"Act" auto-scale:"+"`,
	"Ext":    `cat:"Act"`,
	"Target": `cat:"Act"`,

	//////// Calcium for learning

	"CaM":     `cat:"Learn"`,
	"CaP":     `cat:"Learn"`,
	"CaD":     `cat:"Learn"`,
	"CaDPrev": `cat:"Learn"`,

	"CaSyn":    `cat:"Learn"`,
	"LearnCa":  `cat:"Learn"`,
	"LearnCaM": `cat:"Learn"`,
	"LearnCaP": `cat:"Learn"`,
	"LearnCaD": `cat:"Learn"`,
	"CaDiff":   `cat:"Learn"`,

	"GaM": `cat:"Learn"`,
	"GaP": `cat:"Learn"`,
	"GaD": `cat:"Learn"`,

	"TimeDiff": `cat:"Learn"`,

	"TimeDiffPeak":    `cat:"Learn"`,
	"TimeDiffPeakCyc": `cat:"Learn" auto-scale:"+"`,
	"MinusPeak":       `cat:"Learn"`,
	"MinusPeakCyc":    `cat:"Learn" auto-scale:"+"`,
	"PlusPeak":        `cat:"Learn"`,
	"PlusPeakCyc":     `cat:"Learn" auto-scale:"+"`,

	"LearnDiff":   `cat:"Learn"`,
	"LearnNow":    `cat:"Learn"`,
	"RLRate":      `cat:"Learn" auto-scale:"+"`,
	"ETrace":      `cat:"Learn"`,
	"ETraceLearn": `cat:"Learn" auto-scale:"+"`,

	//////// NMDA channels

	"GnmdaSyn":   `cat:"Excite" auto-scale:"+"`,
	"Gnmda":      `cat:"Excite" auto-scale:"+"`,
	"GnmdaLrn":   `cat:"Excite" auto-scale:"+"`,
	"GnmdaMaint": `cat:"Excite" auto-scale:"+"`,
	"NmdaCa":     `cat:"Excite" auto-scale:"+"`,

	////////  VGCC voltage gated calcium channels

	"Gvgcc":     `cat:"Excite" auto-scale:"+"`,
	"VgccM":     `cat:"Excite"`,
	"VgccH":     `cat:"Excite"`,
	"VgccCa":    `cat:"Excite" auto-scale:"+"`,
	"VgccCaInt": `cat:"Excite" auto-scale:"+"`,

	////////  Misc Excitatory Vars

	"Burst":      `cat:"Excite"`,
	"BurstPrv":   `cat:"Excite"`,
	"CtxtGe":     `cat:"Excite"`,
	"CtxtGeRaw":  `cat:"Excite"`,
	"CtxtGeOrig": `cat:"Excite"`,

	//////// GABA channels

	"GgabaB": `cat:"Inhib" auto-scale:"+"`,
	"GababM": `cat:"Inhib" auto-scale:"+"`,
	"GababX": `cat:"Inhib" auto-scale:"+"`,

	//////// SST somatostatin inhibition factors

	"Gak":      `cat:"Inhib" auto-scale:"+"`,
	"SSGiDend": `cat:"Inhib" auto-scale:"+"`,

	"GknaMed":  `cat:"Inhib" auto-scale:"+"`,
	"GknaSlow": `cat:"Inhib" auto-scale:"+"`,
	"Gkir":     `cat:"Inhib"`,
	"KirM":     `cat:"Inhib"`,

	////////  SKCa small conductance calcium-gated potassium channels

	"Gsk":    `cat:"Inhib"`,
	"SKCaIn": `cat:"Inhib"`,
	"SKCaR":  `cat:"Inhib"`,
	"SKCaM":  `cat:"Inhib"`,

	//////// AHP channels: Mahp, Sahp

	"Gmahp":  `cat:"Inhib" auto-scale:"+"`,
	"MahpN":  `cat:"Inhib" auto-scale:"+"`,
	"Gsahp":  `cat:"Inhib" auto-scale:"+"`,
	"SahpCa": `cat:"Inhib"`,
	"SahpN":  `cat:"Inhib"`,

	//////// Stats, aggregate values

	"ActM":     `cat:"Stats"`,
	"ActP":     `cat:"Stats"`,
	"Beta1":    `cat:"Stats"`,
	"Beta2":    `cat:"Stats"`,
	"CaPMax":   `cat:"Stats"`,
	"CaPMaxCa": `cat:"Stats"`,

	//////// Noise

	"GeNoise":  `cat:"Gmisc"`,
	"GeNoiseP": `cat:"Gmisc"`,
	"GiNoise":  `cat:"Gmisc"`,
	"GiNoiseP": `cat:"Gmisc"`,

	//////// Ge, Gi integration

	"GeExt":     `cat:"Gmisc"`,
	"GeRaw":     `cat:"Gmisc"`,
	"GeSyn":     `cat:"Gmisc" range:"2"`,
	"GiRaw":     `cat:"Gmisc"`,
	"GiSyn":     `cat:"Gmisc"`,
	"GeInt":     `cat:"Gmisc" range:"2"`,
	"GeIntNorm": `cat:"Gmisc" range:"1"`,
	"GiInt":     `cat:"Gmisc" range:"2"`,
	"GModRaw":   `cat:"Gmisc"`,
	"GModSyn":   `cat:"Gmisc"`,
	"SMaintP":   `cat:"Gmisc"`,
	"GMaintRaw": `cat:"Gmisc"`,
	"GMaintSyn": `cat:"Gmisc"`,

	"NeurFlags": `display:"-"`,

	"CaBins": `cat:"Spikes"`,

	//////// Long-term average activation, set point for synaptic scaling

	"ActAvg":  `cat:"Avg"`,
	"AvgPct":  `cat:"Avg" range:"2"`,
	"TrgAvg":  `cat:"Avg" range:"2"`,
	"DTrgAvg": `cat:"Avg" auto-scale:"+"`,
	"AvgDif":  `cat:"Avg"`,
	"GeBase":  `cat:"Avg"`,
	"GiBase":  `cat:"Avg"`,

	//////// Layer-level variables

	"DA":       `cat:"Learn" doc:"dopamine neuromodulation (layer-level variable)"`,
	"ACh":      `cat:"Learn" doc:"cholinergic neuromodulation (layer-level variable)"`,
	"NE":       `cat:"Learn" doc:"norepinepherine (noradrenaline) neuromodulation  (layer-level variable)"`,
	"Ser":      `cat:"Learn" doc:"serotonin neuromodulation (layer-level variable)"`,
	"Gated":    `cat:"Learn" doc:"signals whether the layer gated (pool-level variable)"`,
	"ModAct":   `cat:"Learn" doc:"pool-level modulatory activity signal (for BG Matrix and Patch layers)"`,
	"PoolDAD1": `cat:"Learn" doc:"pool-level dopamine D1 signal (for BG Matrix layers only)"`,
	"PoolDAD2": `cat:"Learn" doc:"pool-level dopamine D2 signal (for BG Matrix layers only)"`,
}

var (
	NeuronVarNames []string
	NeuronVarsMap  map[string]int
)

// NeuronLayerVars are pool or layer-level variables displayed as neuron layers.
var (
	NeuronLayerVars  = []string{"DA", "ACh", "NE", "Ser", "Gated", "ModAct", "PoolDAD1", "PoolDAD2"}
	NNeuronLayerVars = len(NeuronLayerVars)
	NNeuronCaBins    = 20 // generic max for display
)

func init() {
	NeuronVarsMap = make(map[string]int, int(NeuronVarsN)+int(NeuronAvgVarsN)+NNeuronLayerVars)
	for i := Spike; i < CaBins; i++ {
		vnm := i.String()
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = int(i)
		tag := NeuronVarProps[vnm]
		NeuronVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
	}
	for i := range NNeuronCaBins {
		vnm := fmt.Sprintf("CaBin%02d", i)
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = int(CaBins) + i
		tag := NeuronVarProps[CaBins.String()]
		NeuronVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(CaBins.Desc(), "\n", " ") + `"`
	}
	nVars := int(CaBins) + NNeuronCaBins
	for i := ActAvg; i < NeuronAvgVarsN; i++ {
		vnm := i.String()
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = nVars + int(i)
		tag := NeuronVarProps[vnm]
		NeuronVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
	}
	for i, vnm := range NeuronLayerVars {
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = i + int(nVars) + int(NeuronAvgVarsN)
	}
}

// NeuronVarIndexByName returns the index of the variable in the Neuron, or error
func NeuronVarIndexByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %s not valid", varNm)
	}
	return i, nil
}
