// Code generated by "core generate -add-types"; DO NOT EDIT.

package chanplots

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.AKPlot", IDName: "ak-plot", Methods: []types.Method{{Name: "GVRun", Doc: "GVRun plots the conductance G (and other variables) as a function of V.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equations over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "AK", Doc: "AK function"}, {Name: "AKs", Doc: "AKs simplified function"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeSpike", Doc: "do spiking instead of voltage ramp"}, {Name: "SpikeFreq", Doc: "spiking frequency"}, {Name: "TimeVstart", Doc: "time-run starting membrane potential"}, {Name: "TimeVend", Doc: "time-run ending membrane potential"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.GababPlot", IDName: "gabab-plot", Methods: []types.Method{{Name: "GVRun", Doc: "GVRun plots the conductance G (and other variables) as a function of V.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "GSRun", Doc: "GSRun plots conductance over spiking.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equations over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "GABAstd", Doc: "standard chans version of GABAB"}, {Name: "GABAbv", Doc: "multiplier on GABAb as function of voltage"}, {Name: "GABAbo", Doc: "offset of GABAb function"}, {Name: "GABAberev", Doc: "GABAb reversal / driving potential"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "Smax", Doc: "max number of spikes"}, {Name: "RiseTau", Doc: "rise time constant"}, {Name: "DecayTau", Doc: "decay time constant -- must NOT be same as RiseTau"}, {Name: "GsXInit", Doc: "initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start"}, {Name: "MaxTime", Doc: "time when peak conductance occurs, in TimeInc units"}, {Name: "TauFact", Doc: "time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))"}, {Name: "TimeSteps", Doc: "total number of time steps to take"}, {Name: "TimeInc", Doc: "time increment per step"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.KirPlot", IDName: "kir-plot", Methods: []types.Method{{Name: "GVRun", Doc: "VmRun plots the equation as a function of V", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "Kir", Doc: "kIR function"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeSpike", Doc: "do spiking instead of voltage ramp"}, {Name: "SpikeFreq", Doc: "spiking frequency"}, {Name: "TimeVstart", Doc: "time-run starting membrane potential"}, {Name: "TimeVend", Doc: "time-run ending membrane potential"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.MahpPlot", IDName: "mahp-plot", Methods: []types.Method{{Name: "GVRun", Doc: "GVRun plots the conductance G (and other variables) as a function of V.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "Mahp", Doc: "mAHP function"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeSpike", Doc: "do spiking instead of voltage ramp"}, {Name: "SpikeFreq", Doc: "spiking frequency"}, {Name: "TimeVstart", Doc: "time-run starting membrane potential"}, {Name: "TimeVend", Doc: "time-run ending membrane potential"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.NMDAPlot", IDName: "nmda-plot", Methods: []types.Method{{Name: "GVRun", Doc: "GVRun plots the conductance G (and other variables) as a function of V.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "NMDAStd", Doc: "standard NMDA implementation in chans"}, {Name: "NMDAv", Doc: "multiplier on NMDA as function of voltage"}, {Name: "MgC", Doc: "magnesium ion concentration -- somewhere between 1 and 1.5"}, {Name: "NMDAd", Doc: "denominator of NMDA function"}, {Name: "NMDAerev", Doc: "NMDA reversal / driving potential"}, {Name: "BugVoff", Doc: "for old buggy NMDA: voff value to use"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "Tau", Doc: "decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeV", Doc: "voltage for TimeRun"}, {Name: "TimeGin", Doc: "NMDA Gsyn current input at every time step"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.SahpPlot", IDName: "sahp-plot", Methods: []types.Method{{Name: "GCaRun", Doc: "GCaRun plots the conductance G (and other variables) as a function of Ca.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "Sahp", Doc: "sAHP function"}, {Name: "CaStart", Doc: "starting calcium"}, {Name: "CaEnd", Doc: "ending calcium"}, {Name: "CaStep", Doc: "calcium increment"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeCaStart", Doc: "time-run starting calcium"}, {Name: "TimeCaD", Doc: "time-run CaD value at end of each theta cycle"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.SKCaPlot", IDName: "sk-ca-plot", Methods: []types.Method{{Name: "GCaRun", Doc: "GCaRun plots the conductance G (and other variables) as a function of Ca.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "SKCa", Doc: "SKCa params"}, {Name: "CaParams", Doc: "time constants for integrating Ca from spiking across M, P and D cascading levels"}, {Name: "NoSpikeThr", Doc: "threshold of SK M gating factor above which the neuron cannot spike"}, {Name: "CaStep", Doc: "Ca conc increment for M gating func plot"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeSpike", Doc: "do spiking instead of Ca conc ramp"}, {Name: "SpikeFreq", Doc: "spiking frequency"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.SynCaPlot", IDName: "syn-ca-plot", Methods: []types.Method{{Name: "TimeRun", Doc: "TimeRun runs the equation.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "CaDt", Doc: "Ca time constants"}, {Name: "Minit"}, {Name: "Pinit"}, {Name: "Dinit"}, {Name: "MdtAdj", Doc: "adjustment to dt to account for discrete time updating"}, {Name: "PdtAdj", Doc: "adjustment to dt to account for discrete time updating"}, {Name: "DdtAdj", Doc: "adjustment to dt to account for discrete time updating"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "Dir"}, {Name: "Tabs"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/axon/v2/chans/chanplots.VGCCPlot", IDName: "vgcc-plot", Methods: []types.Method{{Name: "GVRun", Doc: "GVRun plots the conductance G (and other variables) as a function of V.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}, {Name: "TimeRun", Doc: "TimeRun runs the equation over time.", Directives: []types.Directive{{Tool: "types", Directive: "add"}}}}, Fields: []types.Field{{Name: "VGCC", Doc: "VGCC function"}, {Name: "Vstart", Doc: "starting voltage"}, {Name: "Vend", Doc: "ending voltage"}, {Name: "Vstep", Doc: "voltage increment"}, {Name: "TimeSteps", Doc: "number of time steps"}, {Name: "TimeSpike", Doc: "do spiking instead of voltage ramp"}, {Name: "SpikeFreq", Doc: "spiking frequency"}, {Name: "TimeVstart", Doc: "time-run starting membrane potential"}, {Name: "TimeVend", Doc: "time-run ending membrane potential"}, {Name: "Dir"}, {Name: "Tabs"}}})