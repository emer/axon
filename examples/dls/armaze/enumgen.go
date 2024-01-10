// Code generated by "goki generate -add-types"; DO NOT EDIT.

package armaze

import (
	"errors"
	"log"
	"strconv"
	"strings"

	"goki.dev/enums"
)

var _ActionsValues = []Actions{0, 1, 2, 3, 4}

// ActionsN is the highest valid value
// for type Actions, plus one.
const ActionsN Actions = 5

// An "invalid array index" compiler error signifies that the constant values have changed.
// Re-run the enumgen command to generate them again.
func _ActionsNoOp() {
	var x [1]struct{}
	_ = x[Forward-(0)]
	_ = x[Left-(1)]
	_ = x[Right-(2)]
	_ = x[Consume-(3)]
	_ = x[None-(4)]
}

var _ActionsNameToValueMap = map[string]Actions{
	`Forward`: 0,
	`forward`: 0,
	`Left`:    1,
	`left`:    1,
	`Right`:   2,
	`right`:   2,
	`Consume`: 3,
	`consume`: 3,
	`None`:    4,
	`none`:    4,
}

var _ActionsDescMap = map[Actions]string{
	0: ``,
	1: ``,
	2: ``,
	3: ``,
	4: ``,
}

var _ActionsMap = map[Actions]string{
	0: `Forward`,
	1: `Left`,
	2: `Right`,
	3: `Consume`,
	4: `None`,
}

// String returns the string representation
// of this Actions value.
func (i Actions) String() string {
	if str, ok := _ActionsMap[i]; ok {
		return str
	}
	return strconv.FormatInt(int64(i), 10)
}

// SetString sets the Actions value from its
// string representation, and returns an
// error if the string is invalid.
func (i *Actions) SetString(s string) error {
	if val, ok := _ActionsNameToValueMap[s]; ok {
		*i = val
		return nil
	}
	if val, ok := _ActionsNameToValueMap[strings.ToLower(s)]; ok {
		*i = val
		return nil
	}
	return errors.New(s + " is not a valid value for type Actions")
}

// Int64 returns the Actions value as an int64.
func (i Actions) Int64() int64 {
	return int64(i)
}

// SetInt64 sets the Actions value from an int64.
func (i *Actions) SetInt64(in int64) {
	*i = Actions(in)
}

// Desc returns the description of the Actions value.
func (i Actions) Desc() string {
	if str, ok := _ActionsDescMap[i]; ok {
		return str
	}
	return i.String()
}

// ActionsValues returns all possible values
// for the type Actions.
func ActionsValues() []Actions {
	return _ActionsValues
}

// Values returns all possible values
// for the type Actions.
func (i Actions) Values() []enums.Enum {
	res := make([]enums.Enum, len(_ActionsValues))
	for i, d := range _ActionsValues {
		res[i] = d
	}
	return res
}

// IsValid returns whether the value is a
// valid option for type Actions.
func (i Actions) IsValid() bool {
	_, ok := _ActionsMap[i]
	return ok
}

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Actions) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Actions) UnmarshalText(text []byte) error {
	if err := i.SetString(string(text)); err != nil {
		log.Println(err)
	}
	return nil
}

var _ParadigmsValues = []Paradigms{0}

// ParadigmsN is the highest valid value
// for type Paradigms, plus one.
const ParadigmsN Paradigms = 1

// An "invalid array index" compiler error signifies that the constant values have changed.
// Re-run the enumgen command to generate them again.
func _ParadigmsNoOp() {
	var x [1]struct{}
	_ = x[Approach-(0)]
}

var _ParadigmsNameToValueMap = map[string]Paradigms{
	`Approach`: 0,
	`approach`: 0,
}

var _ParadigmsDescMap = map[Paradigms]string{
	0: `Approach is a basic case where one Drive (chosen at random each trial) is fully active and others are at InactiveDrives levels -- goal is to approach the CS associated with the Drive-satisfying US, and avoid negative any negative USs. USs are always placed in same Arms (NArms must be &gt;= NUSs -- any additional Arms are filled at random with additional US copies)`,
}

var _ParadigmsMap = map[Paradigms]string{
	0: `Approach`,
}

// String returns the string representation
// of this Paradigms value.
func (i Paradigms) String() string {
	if str, ok := _ParadigmsMap[i]; ok {
		return str
	}
	return strconv.FormatInt(int64(i), 10)
}

// SetString sets the Paradigms value from its
// string representation, and returns an
// error if the string is invalid.
func (i *Paradigms) SetString(s string) error {
	if val, ok := _ParadigmsNameToValueMap[s]; ok {
		*i = val
		return nil
	}
	if val, ok := _ParadigmsNameToValueMap[strings.ToLower(s)]; ok {
		*i = val
		return nil
	}
	return errors.New(s + " is not a valid value for type Paradigms")
}

// Int64 returns the Paradigms value as an int64.
func (i Paradigms) Int64() int64 {
	return int64(i)
}

// SetInt64 sets the Paradigms value from an int64.
func (i *Paradigms) SetInt64(in int64) {
	*i = Paradigms(in)
}

// Desc returns the description of the Paradigms value.
func (i Paradigms) Desc() string {
	if str, ok := _ParadigmsDescMap[i]; ok {
		return str
	}
	return i.String()
}

// ParadigmsValues returns all possible values
// for the type Paradigms.
func ParadigmsValues() []Paradigms {
	return _ParadigmsValues
}

// Values returns all possible values
// for the type Paradigms.
func (i Paradigms) Values() []enums.Enum {
	res := make([]enums.Enum, len(_ParadigmsValues))
	for i, d := range _ParadigmsValues {
		res[i] = d
	}
	return res
}

// IsValid returns whether the value is a
// valid option for type Paradigms.
func (i Paradigms) IsValid() bool {
	_, ok := _ParadigmsMap[i]
	return ok
}

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Paradigms) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Paradigms) UnmarshalText(text []byte) error {
	if err := i.SetString(string(text)); err != nil {
		log.Println(err)
	}
	return nil
}

var _TraceStatesValues = []TraceStates{0, 1, 2, 3, 4, 5, 6, 7}

// TraceStatesN is the highest valid value
// for type TraceStates, plus one.
const TraceStatesN TraceStates = 8

// An "invalid array index" compiler error signifies that the constant values have changed.
// Re-run the enumgen command to generate them again.
func _TraceStatesNoOp() {
	var x [1]struct{}
	_ = x[TrSearching-(0)]
	_ = x[TrDeciding-(1)]
	_ = x[TrJustEngaged-(2)]
	_ = x[TrApproaching-(3)]
	_ = x[TrConsuming-(4)]
	_ = x[TrRewarded-(5)]
	_ = x[TrGiveUp-(6)]
	_ = x[TrBumping-(7)]
}

var _TraceStatesNameToValueMap = map[string]TraceStates{
	`TrSearching`:   0,
	`trsearching`:   0,
	`TrDeciding`:    1,
	`trdeciding`:    1,
	`TrJustEngaged`: 2,
	`trjustengaged`: 2,
	`TrApproaching`: 3,
	`trapproaching`: 3,
	`TrConsuming`:   4,
	`trconsuming`:   4,
	`TrRewarded`:    5,
	`trrewarded`:    5,
	`TrGiveUp`:      6,
	`trgiveup`:      6,
	`TrBumping`:     7,
	`trbumping`:     7,
}

var _TraceStatesDescMap = map[TraceStates]string{
	0: `Searching is not yet goal engaged, looking for a goal`,
	1: `Deciding is having some partial gating but not in time for action`,
	2: `JustEngaged means just decided to engage in a goal`,
	3: `Approaching is goal engaged, approaching the goal`,
	4: `Consuming is consuming the US, first step (prior to getting reward, step1)`,
	5: `Rewarded is just received reward from a US`,
	6: `GiveUp is when goal is abandoned`,
	7: `Bumping is bumping into a wall`,
}

var _TraceStatesMap = map[TraceStates]string{
	0: `TrSearching`,
	1: `TrDeciding`,
	2: `TrJustEngaged`,
	3: `TrApproaching`,
	4: `TrConsuming`,
	5: `TrRewarded`,
	6: `TrGiveUp`,
	7: `TrBumping`,
}

// String returns the string representation
// of this TraceStates value.
func (i TraceStates) String() string {
	if str, ok := _TraceStatesMap[i]; ok {
		return str
	}
	return strconv.FormatInt(int64(i), 10)
}

// SetString sets the TraceStates value from its
// string representation, and returns an
// error if the string is invalid.
func (i *TraceStates) SetString(s string) error {
	if val, ok := _TraceStatesNameToValueMap[s]; ok {
		*i = val
		return nil
	}
	if val, ok := _TraceStatesNameToValueMap[strings.ToLower(s)]; ok {
		*i = val
		return nil
	}
	return errors.New(s + " is not a valid value for type TraceStates")
}

// Int64 returns the TraceStates value as an int64.
func (i TraceStates) Int64() int64 {
	return int64(i)
}

// SetInt64 sets the TraceStates value from an int64.
func (i *TraceStates) SetInt64(in int64) {
	*i = TraceStates(in)
}

// Desc returns the description of the TraceStates value.
func (i TraceStates) Desc() string {
	if str, ok := _TraceStatesDescMap[i]; ok {
		return str
	}
	return i.String()
}

// TraceStatesValues returns all possible values
// for the type TraceStates.
func TraceStatesValues() []TraceStates {
	return _TraceStatesValues
}

// Values returns all possible values
// for the type TraceStates.
func (i TraceStates) Values() []enums.Enum {
	res := make([]enums.Enum, len(_TraceStatesValues))
	for i, d := range _TraceStatesValues {
		res[i] = d
	}
	return res
}

// IsValid returns whether the value is a
// valid option for type TraceStates.
func (i TraceStates) IsValid() bool {
	_, ok := _TraceStatesMap[i]
	return ok
}

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i TraceStates) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *TraceStates) UnmarshalText(text []byte) error {
	if err := i.SetString(string(text)); err != nil {
		log.Println(err)
	}
	return nil
}