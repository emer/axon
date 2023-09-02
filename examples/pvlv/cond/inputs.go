// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

import "github.com/emer/etable/etensor"

var (
	NUSs = 4

	NStims = 12

	MaxTime = 6

	USShape = []int{1, NUSs}

	StimShape = []int{3, 4}

	ContextShape = []int{6, 5}

	// USTimeShape is overall shape of USTime
	USTimeShape = []int{StimShape[0], StimShape[1], 1, MaxTime}

	// USTimeOff is activated when the US goes off
	USTimeOff = []int{StimShape[0] - 1, StimShape[1] - 1, 0, 5}

	// Stims maps stimuli to indexes for input layer
	Stims = map[string]int{
		"A": 0,
		"B": 1,
		"C": 2,
		"D": 3,
		"E": 4,
		"F": 5,
		"U": 6,
		"V": 7,
		"W": 8,
		"X": 9,
		"Y": 10,
		"Z": 11,
	}

	// Contexts maps contexts to indexes for input layer
	Contexts = map[string]int{
		"A":   0,
		"B":   1,
		"C":   2,
		"D":   3,
		"E":   4,
		"F":   5,
		"U":   6,
		"V":   7,
		"W":   8,
		"X":   9,
		"Y":   10,
		"Z":   11,
		"AB":  12,
		"AC":  13,
		"AX":  14,
		"AY":  15,
		"AZ":  16,
		"BX":  17,
		"BY":  18,
		"BZ":  19,
		"CX":  20,
		"CY":  21,
		"CZ":  22,
		"DU":  23,
		"ED":  24,
		"EU":  25,
		"EV":  26,
		"FV":  27,
		"A_B": 28,
	}
)

// StimIdx returns index for given stimulus
func StimIdx(stm string) int {
	return Stims[stm]
}

// StimYX returns stimulus YX indexes for stimulus number
func StimYX(stidx int) []int {
	y := stidx / StimShape[1]
	x := stidx % StimShape[1]
	return []int{y, x}
}

// SetStim sets stimulus for given input, returns index
func SetStim(tsr *etensor.Float32, nyrep int, stm string) int {
	stidx := StimIdx(stm)
	xy := StimYX(stidx)
	xy = append(xy, 0)
	xy = append(xy, 0)
	for y := 0; y < nyrep; y++ {
		xy[2] = y
		tsr.Set(xy, 1)
	}
	return stidx
}

// ContextIdx returns index for given context
func ContextIdx(ctx string) int {
	return Contexts[ctx]
}

// ContextYX returns context YX indexes for context number
func ContextYX(ctidx int) []int {
	y := ctidx / ContextShape[1]
	x := ctidx % ContextShape[1]
	return []int{y, x}
}

// SetContext sets context for given input
func SetContext(tsr *etensor.Float32, nyrep int, ctx string) int {
	ctidx := ContextIdx(ctx)
	xy := ContextYX(ctidx)
	xy = append(xy, 0)
	xy = append(xy, 0)
	for y := 0; y < nyrep; y++ {
		xy[2] = y
		tsr.Set(xy, 1)
	}
	return ctidx
}

// USTimeIdx returns index for US time based on stimulus, tick, start and end
// returns nil if not active.
func USTimeIdx(stidx, tick, start, end int) []int {
	tm := tick - start
	if tm < 1 {
		return nil
	}
	if tick > end {
		return nil
	}
	st := StimYX(stidx)
	st = append(st, 0)
	st = append(st, tm-1)
	return st
}

// SetUSTime sets USTime based on given values.
// returns false if not set.
func SetUSTime(tsr *etensor.Float32, nyrep, stidx, tick, start, end int) bool {
	idx := USTimeIdx(stidx, tick, start, end)
	if idx == nil {
		return false
	}
	for y := 0; y < nyrep; y++ {
		idx[2] = y
		tsr.Set(idx, 1)
	}
	return true
}

// SetTime sets Time input
func SetTime(tsr *etensor.Float32, nyrep int, tick int) {
	if tick < 0 {
		tick = 0
	}
	idx := []int{0, tick, 0, 0}
	for y := 0; y < nyrep; y++ {
		idx[2] = y
		tsr.Set(idx, 1)
	}
}

// SetUS sets US input
func SetUS(tsr *etensor.Float32, nyrep int, pv int, mag float32) {
	idx := []int{0, pv, 0, 0}
	for y := 0; y < nyrep; y++ {
		idx[2] = y
		tsr.Set(idx, mag)
	}
}
