// Code generated by "core generate -add-types"; DO NOT EDIT.

package armaze

import (
	"cogentcore.org/core/enums"
)

var _ActionsValues = []Actions{0, 1, 2, 3, 4}

// ActionsN is the highest valid value for type Actions, plus one.
const ActionsN Actions = 5

var _ActionsValueMap = map[string]Actions{`Forward`: 0, `Left`: 1, `Right`: 2, `Consume`: 3, `None`: 4}

var _ActionsDescMap = map[Actions]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``}

var _ActionsMap = map[Actions]string{0: `Forward`, 1: `Left`, 2: `Right`, 3: `Consume`, 4: `None`}

// String returns the string representation of this Actions value.
func (i Actions) String() string { return enums.String(i, _ActionsMap) }

// SetString sets the Actions value from its string representation,
// and returns an error if the string is invalid.
func (i *Actions) SetString(s string) error {
	return enums.SetString(i, s, _ActionsValueMap, "Actions")
}

// Int64 returns the Actions value as an int64.
func (i Actions) Int64() int64 { return int64(i) }

// SetInt64 sets the Actions value from an int64.
func (i *Actions) SetInt64(in int64) { *i = Actions(in) }

// Desc returns the description of the Actions value.
func (i Actions) Desc() string { return enums.Desc(i, _ActionsDescMap) }

// ActionsValues returns all possible values for the type Actions.
func ActionsValues() []Actions { return _ActionsValues }

// Values returns all possible values for the type Actions.
func (i Actions) Values() []enums.Enum { return enums.Values(_ActionsValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Actions) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Actions) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "Actions") }

var _ParadigmsValues = []Paradigms{0, 1}

// ParadigmsN is the highest valid value for type Paradigms, plus one.
const ParadigmsN Paradigms = 2

var _ParadigmsValueMap = map[string]Paradigms{`GroupGoodBad`: 0, `GroupRisk`: 1}

var _ParadigmsDescMap = map[Paradigms]string{0: `GroupGoodBad allocates Arms into 2 groups, with first group unambiguously Good and the second Bad, using the Min, Max values of each Range parameter: Length, Effort, USMag, USProb. Good has Min cost, Max US, and opposite for Bad. This also aligns with the ordering of USs, such that negative USs are last.`, 1: `GroupRisk allocates Arms into 2 groups with conflicting Cost and Benefit tradeoffs, with the first group having Min cost and Min US, and the second group having Max cost and Max US.`}

var _ParadigmsMap = map[Paradigms]string{0: `GroupGoodBad`, 1: `GroupRisk`}

// String returns the string representation of this Paradigms value.
func (i Paradigms) String() string { return enums.String(i, _ParadigmsMap) }

// SetString sets the Paradigms value from its string representation,
// and returns an error if the string is invalid.
func (i *Paradigms) SetString(s string) error {
	return enums.SetString(i, s, _ParadigmsValueMap, "Paradigms")
}

// Int64 returns the Paradigms value as an int64.
func (i Paradigms) Int64() int64 { return int64(i) }

// SetInt64 sets the Paradigms value from an int64.
func (i *Paradigms) SetInt64(in int64) { *i = Paradigms(in) }

// Desc returns the description of the Paradigms value.
func (i Paradigms) Desc() string { return enums.Desc(i, _ParadigmsDescMap) }

// ParadigmsValues returns all possible values for the type Paradigms.
func ParadigmsValues() []Paradigms { return _ParadigmsValues }

// Values returns all possible values for the type Paradigms.
func (i Paradigms) Values() []enums.Enum { return enums.Values(_ParadigmsValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Paradigms) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Paradigms) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "Paradigms")
}

var _TraceStatesValues = []TraceStates{0, 1, 2, 3, 4, 5, 6, 7}

// TraceStatesN is the highest valid value for type TraceStates, plus one.
const TraceStatesN TraceStates = 8

var _TraceStatesValueMap = map[string]TraceStates{`TrSearching`: 0, `TrDeciding`: 1, `TrJustEngaged`: 2, `TrApproaching`: 3, `TrConsuming`: 4, `TrRewarded`: 5, `TrGiveUp`: 6, `TrBumping`: 7}

var _TraceStatesDescMap = map[TraceStates]string{0: `Searching is not yet goal engaged, looking for a goal`, 1: `Deciding is having some partial gating but not in time for action`, 2: `JustEngaged means just decided to engage in a goal`, 3: `Approaching is goal engaged, approaching the goal`, 4: `Consuming is consuming the US, first step (prior to getting reward, step1)`, 5: `Rewarded is just received reward from a US`, 6: `GiveUp is when goal is abandoned`, 7: `Bumping is bumping into a wall`}

var _TraceStatesMap = map[TraceStates]string{0: `TrSearching`, 1: `TrDeciding`, 2: `TrJustEngaged`, 3: `TrApproaching`, 4: `TrConsuming`, 5: `TrRewarded`, 6: `TrGiveUp`, 7: `TrBumping`}

// String returns the string representation of this TraceStates value.
func (i TraceStates) String() string { return enums.String(i, _TraceStatesMap) }

// SetString sets the TraceStates value from its string representation,
// and returns an error if the string is invalid.
func (i *TraceStates) SetString(s string) error {
	return enums.SetString(i, s, _TraceStatesValueMap, "TraceStates")
}

// Int64 returns the TraceStates value as an int64.
func (i TraceStates) Int64() int64 { return int64(i) }

// SetInt64 sets the TraceStates value from an int64.
func (i *TraceStates) SetInt64(in int64) { *i = TraceStates(in) }

// Desc returns the description of the TraceStates value.
func (i TraceStates) Desc() string { return enums.Desc(i, _TraceStatesDescMap) }

// TraceStatesValues returns all possible values for the type TraceStates.
func TraceStatesValues() []TraceStates { return _TraceStatesValues }

// Values returns all possible values for the type TraceStates.
func (i TraceStates) Values() []enums.Enum { return enums.Values(_TraceStatesValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i TraceStates) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *TraceStates) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "TraceStates")
}