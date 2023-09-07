# CondEnv

CondEnv provides a flexible implementation of standard Pavlovian conditioning experiments involving CS -> US sequences (trials). Has a large database of standard conditioning paradigms parameterized in a controlled manner.

Time hierarchy:

* `Run` = sequence of 1 or more Conditions
* `Condition` = specific mix of trial types, generated at start of Condition
* `Block` = one full pass through all trial types generated for condition (like Epoch)
* `Trial` = one behavioral trial consisting of CS -> US presentation over time steps (Ticks)
* `Tick` = discrete time steps within behavioral Trial, typically one Network update (Alpha / Theta cycle)

**Be sure to do `go test` if you modify or add** runs, conds, or blocks -- it tests that everything linked in runs exists etc.

# Example

AllRuns (in `runs_all.go`) contains this case:

```Go
	"PosAcqExt_A100_A0": {
		Name:  "PosAcq",
		Desc:  "Standard positive valence acquisition: A = 100%, then extinction A0",
		Cond1: "PosAcq_A100",
		Cond2: "PosExt_A0",
	},
```

Which references these cases in AllConditions (`conds_all.go`):

```Go
	"PosAcq_A100": {
		Name:      "PosAcq_A100",
		Desc:      "Standard positive valence acquisition: A = 100%",
		Block:     "PosAcq_A100",
		FixedProb: true,
		NBlocks:   51,
		NTrials:   4,
		Permute:   true,
	},
	"PosExt_A0": {
		Name:      "PosExt_A0",
		Desc:      "Pavlovian extinction: A_NR_Pos, A = 0%",
		Block:     "PosExt_A0",
		FixedProb: false,
		NBlocks:   50,
		NTrials:   8,
		Permute:   true,
	},
```

Which then reference corresponding block types in AllBlocks (`blocks_all.go`):

```Go
	"PosAcq_A100": {
		{
			Name:     "A_R",
			Pct:      1,
			Valence:  Pos,
			USProb:   1,
			MixedUS:  false,
			USMag:    1,
			NTicks:   5,
			CS:       "A",
			CSStart:  1,
			CSEnd:    3,
			CS2Start: -1,
			CS2End:   -1,
			US:       0,
			USStart:  3,
			USEnd:    3,
			Context:  "A",
		},
	},
	"PosExt_A0": {
		{
			Name:     "A_NR",
			Pct:      1,
			Valence:  Pos,
			USProb:   0,
			MixedUS:  false,
			USMag:    1,
			NTicks:   5,
			CS:       "A",
			CSStart:  1,
			CSEnd:    3,
			CS2Start: -1,
			CS2End:   -1,
			US:       0,
			USStart:  3,
			USEnd:    3,
			Context:  "A",
		},
	},
```

