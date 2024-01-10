// Code generated by "goki generate ./..."; DO NOT EDIT.

package cond

import (
	"goki.dev/gti"
	"goki.dev/ordmap"
)

var _ = gti.AddType(&gti.Type{
	Name:       "github.com/emer/axon/examples/pvlv/cond.Condition",
	ShortName:  "cond.Condition",
	IDName:     "condition",
	Doc:        "Condition defines parameters for running a specific type of conditioning expt",
	Directives: gti.Directives{},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Name", &gti.Field{Name: "Name", Type: "string", LocalType: "string", Doc: "identifier for this type of configuration", Directives: gti.Directives{}, Tag: ""}},
		{"Desc", &gti.Field{Name: "Desc", Type: "string", LocalType: "string", Doc: "description of this configuration", Directives: gti.Directives{}, Tag: ""}},
		{"Block", &gti.Field{Name: "Block", Type: "string", LocalType: "string", Doc: "mix of trial types per block to run -- must be listed in AllBlocks", Directives: gti.Directives{}, Tag: ""}},
		{"FixedProb", &gti.Field{Name: "FixedProb", Type: "bool", LocalType: "bool", Doc: "use a permuted list to ensure an exact number of trials have US -- else random draw each time", Directives: gti.Directives{}, Tag: ""}},
		{"NBlocks", &gti.Field{Name: "NBlocks", Type: "int", LocalType: "int", Doc: "number of full blocks of different trial types to run (like Epochs)", Directives: gti.Directives{}, Tag: ""}},
		{"NTrials", &gti.Field{Name: "NTrials", Type: "int", LocalType: "int", Doc: "number of behavioral trials per block -- blocks, with the different types of Trials specified in Block allocated across these Trials.  More different trial types and greater stochasticity (lower probability) of US presentation requires more trials.", Directives: gti.Directives{}, Tag: ""}},
		{"Permute", &gti.Field{Name: "Permute", Type: "bool", LocalType: "bool", Doc: "permute list of generated trials in random order after generation -- otherwise presented in order specified in the Block type", Directives: gti.Directives{}, Tag: ""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:       "github.com/emer/axon/examples/pvlv/cond.CondEnv",
	ShortName:  "cond.CondEnv",
	IDName:     "cond-env",
	Doc:        "CondEnv provides a flexible implementation of standard Pavlovian\nconditioning experiments involving CS -> US sequences (trials).\nHas a large database of standard conditioning paradigms\nparameterized in a controlled manner.\n\nTime hierarchy:\n* Run = sequence of 1 or more Conditions\n* Condition = specific mix of trial types, generated at start of Condition\n* Block = one full pass through all trial types generated for condition (like Epoch)\n* Trial = one behavioral trial consisting of CS -> US presentation over time steps (Ticks)\n* Tick = discrete time steps within behavioral Trial, typically one Network update (Alpha / Theta cycle)",
	Directives: gti.Directives{},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Nm", &gti.Field{Name: "Nm", Type: "string", LocalType: "string", Doc: "name of this environment", Directives: gti.Directives{}, Tag: ""}},
		{"Dsc", &gti.Field{Name: "Dsc", Type: "string", LocalType: "string", Doc: "description of this environment", Directives: gti.Directives{}, Tag: ""}},
		{"NYReps", &gti.Field{Name: "NYReps", Type: "int", LocalType: "int", Doc: "number of Y repetitions for localist reps", Directives: gti.Directives{}, Tag: ""}},
		{"RunName", &gti.Field{Name: "RunName", Type: "string", LocalType: "string", Doc: "current run name", Directives: gti.Directives{}, Tag: ""}},
		{"RunDesc", &gti.Field{Name: "RunDesc", Type: "string", LocalType: "string", Doc: "description of current run", Directives: gti.Directives{}, Tag: ""}},
		{"CondName", &gti.Field{Name: "CondName", Type: "string", LocalType: "string", Doc: "name of current condition", Directives: gti.Directives{}, Tag: ""}},
		{"CondDesc", &gti.Field{Name: "CondDesc", Type: "string", LocalType: "string", Doc: "description of current condition", Directives: gti.Directives{}, Tag: ""}},
		{"Run", &gti.Field{Name: "Run", Type: "github.com/emer/emergent/v2/env.Ctr", LocalType: "env.Ctr", Doc: "counter over runs", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"Condition", &gti.Field{Name: "Condition", Type: "github.com/emer/emergent/v2/env.Ctr", LocalType: "env.Ctr", Doc: "counter over Condition within a run -- Max depends on number of conditions specified in given Run", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"Block", &gti.Field{Name: "Block", Type: "github.com/emer/emergent/v2/env.Ctr", LocalType: "env.Ctr", Doc: "counter over full blocks of all trial types within a Condition -- like an Epoch", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"Trial", &gti.Field{Name: "Trial", Type: "github.com/emer/emergent/v2/env.Ctr", LocalType: "env.Ctr", Doc: "counter of behavioral trials within a Block", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"Tick", &gti.Field{Name: "Tick", Type: "github.com/emer/emergent/v2/env.Ctr", LocalType: "env.Ctr", Doc: "counter of discrete steps within a behavioral trial -- typically maps onto Alpha / Theta cycle in network", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"TrialName", &gti.Field{Name: "TrialName", Type: "string", LocalType: "string", Doc: "name of current trial step", Directives: gti.Directives{}, Tag: "inactive:\"+\""}},
		{"TrialType", &gti.Field{Name: "TrialType", Type: "string", LocalType: "string", Doc: "type of current trial step", Directives: gti.Directives{}, Tag: "inactive:\"+\""}},
		{"USTimeInStr", &gti.Field{Name: "USTimeInStr", Type: "string", LocalType: "string", Doc: "decoded value of USTimeIn", Directives: gti.Directives{}, Tag: "inactive:\"+\""}},
		{"Trials", &gti.Field{Name: "Trials", Type: "[]*github.com/emer/axon/examples/pvlv/cond.Trial", LocalType: "[]*Trial", Doc: "current generated set of trials per Block", Directives: gti.Directives{}, Tag: ""}},
		{"CurRun", &gti.Field{Name: "CurRun", Type: "github.com/emer/axon/examples/pvlv/cond.Run", LocalType: "Run", Doc: "copy of current run parameters", Directives: gti.Directives{}, Tag: ""}},
		{"CurTrial", &gti.Field{Name: "CurTrial", Type: "github.com/emer/axon/examples/pvlv/cond.Trial", LocalType: "Trial", Doc: "copy of info for current trial", Directives: gti.Directives{}, Tag: ""}},
		{"CurStates", &gti.Field{Name: "CurStates", Type: "map[string]*goki.dev/etable/v2/etensor.Float32", LocalType: "map[string]*etensor.Float32", Doc: "current rendered state tensors -- extensible map", Directives: gti.Directives{}, Tag: ""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:       "github.com/emer/axon/examples/pvlv/cond.Run",
	ShortName:  "cond.Run",
	IDName:     "run",
	Doc:        "Run is a sequence of Conditions to run in order",
	Directives: gti.Directives{},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Name", &gti.Field{Name: "Name", Type: "string", LocalType: "string", Doc: "Name of the run", Directives: gti.Directives{}, Tag: ""}},
		{"Desc", &gti.Field{Name: "Desc", Type: "string", LocalType: "string", Doc: "Description", Directives: gti.Directives{}, Tag: ""}},
		{"Weights", &gti.Field{Name: "Weights", Type: "string", LocalType: "string", Doc: "name of condition for weights file to load prior to starting -- allows faster testing but weights may be out of date", Directives: gti.Directives{}, Tag: ""}},
		{"Cond1", &gti.Field{Name: "Cond1", Type: "string", LocalType: "string", Doc: "name of condition 1", Directives: gti.Directives{}, Tag: ""}},
		{"Cond2", &gti.Field{Name: "Cond2", Type: "string", LocalType: "string", Doc: "name of condition 2", Directives: gti.Directives{}, Tag: ""}},
		{"Cond3", &gti.Field{Name: "Cond3", Type: "string", LocalType: "string", Doc: "name of condition 3", Directives: gti.Directives{}, Tag: ""}},
		{"Cond4", &gti.Field{Name: "Cond4", Type: "string", LocalType: "string", Doc: "name of condition 4", Directives: gti.Directives{}, Tag: ""}},
		{"Cond5", &gti.Field{Name: "Cond5", Type: "string", LocalType: "string", Doc: "name of condition 5", Directives: gti.Directives{}, Tag: ""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:      "github.com/emer/axon/examples/pvlv/cond.Valence",
	ShortName: "cond.Valence",
	IDName:    "valence",
	Doc:       "Valence",
	Directives: gti.Directives{
		&gti.Directive{Tool: "enums", Directive: "enum", Args: []string{}},
	},

	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:       "github.com/emer/axon/examples/pvlv/cond.Trial",
	ShortName:  "cond.Trial",
	IDName:     "trial",
	Doc:        "Trial represents one behavioral trial, unfolding over\nNTicks individual time steps, with one or more CS's (conditioned stimuli)\nand a US (unconditioned stimulus -- outcome).",
	Directives: gti.Directives{},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Name", &gti.Field{Name: "Name", Type: "string", LocalType: "string", Doc: "conventional suffixes: _R = reward, _NR = non-reward; _test = test trial (no learning)", Directives: gti.Directives{}, Tag: ""}},
		{"Test", &gti.Field{Name: "Test", Type: "bool", LocalType: "bool", Doc: "true if testing only -- no learning", Directives: gti.Directives{}, Tag: ""}},
		{"Pct", &gti.Field{Name: "Pct", Type: "float32", LocalType: "float32", Doc: "Percent of all trials for this type", Directives: gti.Directives{}, Tag: ""}},
		{"Valence", &gti.Field{Name: "Valence", Type: "github.com/emer/axon/examples/pvlv/cond.Valence", LocalType: "Valence", Doc: "Positive or negative reward valence", Directives: gti.Directives{}, Tag: ""}},
		{"USProb", &gti.Field{Name: "USProb", Type: "float32", LocalType: "float32", Doc: "Probability of US", Directives: gti.Directives{}, Tag: ""}},
		{"MixedUS", &gti.Field{Name: "MixedUS", Type: "bool", LocalType: "bool", Doc: "Mixed US set?", Directives: gti.Directives{}, Tag: ""}},
		{"USMag", &gti.Field{Name: "USMag", Type: "float32", LocalType: "float32", Doc: "US magnitude", Directives: gti.Directives{}, Tag: ""}},
		{"NTicks", &gti.Field{Name: "NTicks", Type: "int", LocalType: "int", Doc: "Number of ticks for a trial", Directives: gti.Directives{}, Tag: ""}},
		{"CS", &gti.Field{Name: "CS", Type: "string", LocalType: "string", Doc: "Conditioned stimulus", Directives: gti.Directives{}, Tag: ""}},
		{"CSStart", &gti.Field{Name: "CSStart", Type: "int", LocalType: "int", Doc: "Tick of CS start", Directives: gti.Directives{}, Tag: ""}},
		{"CSEnd", &gti.Field{Name: "CSEnd", Type: "int", LocalType: "int", Doc: "Tick of CS end", Directives: gti.Directives{}, Tag: ""}},
		{"CS2Start", &gti.Field{Name: "CS2Start", Type: "int", LocalType: "int", Doc: "Tick of CS2 start: -1 for none", Directives: gti.Directives{}, Tag: ""}},
		{"CS2End", &gti.Field{Name: "CS2End", Type: "int", LocalType: "int", Doc: "Tick of CS2 end: -1 for none", Directives: gti.Directives{}, Tag: ""}},
		{"US", &gti.Field{Name: "US", Type: "int", LocalType: "int", Doc: "Unconditioned stimulus", Directives: gti.Directives{}, Tag: ""}},
		{"USStart", &gti.Field{Name: "USStart", Type: "int", LocalType: "int", Doc: "Tick for start of US presentation", Directives: gti.Directives{}, Tag: ""}},
		{"USEnd", &gti.Field{Name: "USEnd", Type: "int", LocalType: "int", Doc: "Tick for end of US presentation", Directives: gti.Directives{}, Tag: ""}},
		{"Context", &gti.Field{Name: "Context", Type: "string", LocalType: "string", Doc: "Context -- typically same as CS -- if blank CS will be copied -- different in certain extinguishing contexts", Directives: gti.Directives{}, Tag: ""}},
		{"USOn", &gti.Field{Name: "USOn", Type: "bool", LocalType: "bool", Doc: "for rendered trials, true if US active", Directives: gti.Directives{}, Tag: ""}},
		{"CSOn", &gti.Field{Name: "CSOn", Type: "bool", LocalType: "bool", Doc: "for rendered trials, true if CS active", Directives: gti.Directives{}, Tag: ""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:       "github.com/emer/axon/examples/pvlv/cond.Block",
	ShortName:  "cond.Block",
	IDName:     "block",
	Doc:        "Block represents a set of trial types",
	Directives: gti.Directives{},

	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})