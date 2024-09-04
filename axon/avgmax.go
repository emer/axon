// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"log"
	"runtime/debug"
	"sync"
)

// AvgMaxFloatFromIntErr is called when there is an overflow error in AvgMaxI32 FloatFromInt
var (
	AvgMaxFloatFromIntErr   func()
	AvgMaxFloatFromIntErrMu sync.Mutex
)

func SetAvgMaxFloatFromIntErr(fun func()) {
	AvgMaxFloatFromIntErrMu.Lock()
	AvgMaxFloatFromIntErr = fun
	AvgMaxFloatFromIntErrMu.Unlock()
}

//gosl:start avgmaxi

// AvgMaxI32 holds average and max statistics for float32,
// and values used for computing them incrementally,
// using a fixed precision int32 based float representation
// that can be used with GPU-based atomic add and max functions.
// This ONLY works for positive values with averages around 1, and
// the N must be set IN ADVANCE to the correct number of items.
// Once Calc() is called, the incremental values are reset
// via Init() so it is always ready for updating without a separate
// Init() pass.
type AvgMaxI32 struct {

	// Average, from Calc when last computed as Sum / N
	Avg float32 `edit:"-"`

	// Maximum value, copied from CurMax in Calc
	Max float32 `edit:"-"`

	// sum for computing average -- incremented in UpdateVal, reset in Calc
	Sum int32 `edit:"-"`

	// current maximum value, updated via UpdateVal, reset in Calc
	CurMax int32 `edit:"-"`

	// number of items in the sum -- this must be set in advance to a known value and it is used in computing the float <-> int conversion factor to maximize precision.
	N int32 `edit:"-"`

	pad, pad1, pad2 int32
}

// Init initializes incremental values used during updating.
func (am *AvgMaxI32) Init() {
	am.Sum = 0
	am.CurMax = 0
}

// Zero resets everything completely -- back to "as new" state.
func (am *AvgMaxI32) Zero() {
	am.Init()
	am.Avg = 0
	am.Max = 0
}

// FloatToIntFactor returns the factor used for converting float32
// to int32 for Max updating, assuming that
// the overall value is in the general order of 0-1 (127 is the max).
func (am *AvgMaxI32) FloatToIntFactor() float32 {
	return float32(1 << 20) // leaves 7 bits = 128 to cover extreme values
}

// FloatFromIntFactor returns the factor used for converting int32
// back to float32 -- this is 1 / FloatToIntFactor for faster multiplication
// instead of dividing.
func (am *AvgMaxI32) FloatFromIntFactor() float32 {
	return 1.0 / float32(1<<20)
}

// FloatToInt converts the given floating point value
// to a large int for max updating.
func (am *AvgMaxI32) FloatToInt(val float32) int32 {
	return int32(val * am.FloatToIntFactor())
}

// FloatToIntSum converts the given floating point value
// to a large int for sum accumulating -- divides by N.
func (am *AvgMaxI32) FloatToIntSum(val float32) int32 {
	return int32(val * (am.FloatToIntFactor() / float32(am.N)))
}

// FloatFromInt converts the given int32 value produced
// via FloatToInt back into a float32 (divides by factor)
func (am *AvgMaxI32) FloatFromInt(ival, refIndex int32) float32 {
	//gosl:end avgmaxi
	// note: this is not GPU-portable..
	if ival < 0 {
		log.Printf("axon.AvgMaxI32: FloatFromInt is negative, there was an overflow error, in refIndex: %d\n", refIndex)
		fmt.Println(string(debug.Stack()))
		if AvgMaxFloatFromIntErr != nil {
			AvgMaxFloatFromIntErr()
		}
		return 1
	}
	//gosl:start avgmaxi
	return float32(ival) * am.FloatFromIntFactor()
}

// UpdateVal updates stats from given value
func (am *AvgMaxI32) UpdateValue(val float32) {
	am.Sum += am.FloatToIntSum(val)
	ival := am.FloatToInt(val)
	if ival > am.CurMax {
		am.CurMax = ival
	}
}

// Calc computes the average given the current Sum
// and copies over CurMax to Max
// refIndex is a reference index of thing being computed, which will be printed
// in case there is an overflow, for debugging (can't be a string because
// this code runs on GPU).
func (am *AvgMaxI32) Calc(refIndex int32) {
	am.Max = am.FloatFromInt(am.CurMax, refIndex)
	am.Avg = am.FloatFromInt(am.Sum, refIndex) // N is already baked in
	am.Init()                                  // immediately ready to go
}

//gosl:end avgmaxi

//gosl:wgsl avgmaxi
/*
// // AtomicUpdateAvgMaxI32 provides an atomic update using atomic ints
// // implemented by InterlockedAdd HLSL intrinsic.
// // This is a #define because it doesn't work on arg values --
// // must be directly operating on a RWStorageBuffer entity.
// // TODO:gosl do atomics!
// // #define AtomicUpdateAvgMaxI32(am, val) InterlockedAdd(am.Sum, am.FloatToIntSum(val)); InterlockedMax(am.CurMax, am.FloatToInt(val))
*/
//gosl:end avgmaxi

func (am *AvgMaxI32) String() string {
	return fmt.Sprintf("{Avg: %g, Max: %g, N: %d}", am.Avg, am.Max, am.N)
}
