// Code generated by "goki generate ./..."; DO NOT EDIT.

package kinase

import (
	"goki.dev/gti"
	"goki.dev/ordmap"
)

var _ = gti.AddType(&gti.Type{
	Name:      "github.com/emer/axon/kinase.CaDtParams",
	ShortName: "kinase.CaDtParams",
	IDName:    "ca-dt-params",
	Doc:       "CaDtParams has rate constants for integrating Ca calcium\nat different time scales, including final CaP = CaMKII and CaD = DAPK1\ntimescales for LTP potentiation vs. LTD depression factors.",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gosl", Directive: "start", Args: []string{"kinase"}},
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"MTau", &gti.Field{Name: "MTau", Type: "float32", LocalType: "float32", Doc: "CaM (calmodulin) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau.", Directives: gti.Directives{}, Tag: "def:\"2,5\" min:\"1\""}},
		{"PTau", &gti.Field{Name: "PTau", Type: "float32", LocalType: "float32", Doc: "LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information.", Directives: gti.Directives{}, Tag: "def:\"39\" min:\"1\""}},
		{"DTau", &gti.Field{Name: "DTau", Type: "float32", LocalType: "float32", Doc: "LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome).  For integration equations, this cannot be identical to PTau.", Directives: gti.Directives{}, Tag: "def:\"41\" min:\"1\""}},
		{"ExpAdj", &gti.Field{Name: "ExpAdj", Type: "goki.dev/gosl/v2/slbool.Bool", LocalType: "slbool.Bool", Doc: "if true, adjust dt time constants when using exponential integration equations to compensate for difference between discrete and continuous integration", Directives: gti.Directives{}, Tag: ""}},
		{"MDt", &gti.Field{Name: "MDt", Type: "float32", LocalType: "float32", Doc: "rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"PDt", &gti.Field{Name: "PDt", Type: "float32", LocalType: "float32", Doc: "rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"DDt", &gti.Field{Name: "DDt", Type: "float32", LocalType: "float32", Doc: "rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"M4Dt", &gti.Field{Name: "M4Dt", Type: "float32", LocalType: "float32", Doc: "4 * rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"P4Dt", &gti.Field{Name: "P4Dt", Type: "float32", LocalType: "float32", Doc: "4 * rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"D4Dt", &gti.Field{Name: "D4Dt", Type: "float32", LocalType: "float32", Doc: "4 * rate = 1 / tau", Directives: gti.Directives{}, Tag: "view:\"-\" json:\"-\" xml:\"-\" inactive:\"+\""}},
		{"pad", &gti.Field{Name: "pad", Type: "int32", LocalType: "int32", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		{"pad1", &gti.Field{Name: "pad1", Type: "int32", LocalType: "int32", Doc: "", Directives: gti.Directives{}, Tag: ""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:      "github.com/emer/axon/kinase.CaParams",
	ShortName: "kinase.CaParams",
	IDName:    "ca-params",
	Doc:       "CaParams has rate constants for integrating spike-driven Ca calcium\nat different time scales, including final CaP = CaMKII and CaD = DAPK1\ntimescales for LTP potentiation vs. LTD depression factors.",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"SpikeG", &gti.Field{Name: "SpikeG", Type: "float32", LocalType: "float32", Doc: "spiking gain factor for SynSpk learning rule variants.  This alters the overall range of values, keeping them in roughly the unit scale, and affects effective learning rate.", Directives: gti.Directives{}, Tag: "def:\"12\""}},
		{"MaxISI", &gti.Field{Name: "MaxISI", Type: "int32", LocalType: "int32", Doc: "maximum ISI for integrating in Opt mode -- above that just set to 0", Directives: gti.Directives{}, Tag: "def:\"100\""}},
		{"pad", &gti.Field{Name: "pad", Type: "int32", LocalType: "int32", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		{"pad1", &gti.Field{Name: "pad1", Type: "int32", LocalType: "int32", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		{"Dt", &gti.Field{Name: "Dt", Type: "github.com/emer/axon/kinase.CaDtParams", LocalType: "CaDtParams", Doc: "time constants for integrating at M, P, and D cascading levels", Directives: gti.Directives{}, Tag: "view:\"inline\""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:      "github.com/emer/axon/kinase.Rules",
	ShortName: "kinase.Rules",
	IDName:    "rules",
	Doc:       "Rules are different options for Kinase-based learning rules\nThese are now implemented using separate Prjn types in kinasex",
	Directives: gti.Directives{
		&gti.Directive{Tool: "go", Directive: "generate", Args: []string{"goki", "generate", "-add-types"}},
		&gti.Directive{Tool: "enums", Directive: "enum", Args: []string{}},
	},

	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})