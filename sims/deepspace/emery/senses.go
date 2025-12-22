// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import "fmt"

// Senses are sensory inputs that unfold over time.
// Can also use to store abstracted sensory state.
type Senses int64 //enums:bitflag

const (
	// VSLinearAccel is vestibular linear acceleration.
	VSLinearAccel Senses = iota

	// VSLinearVel is vestibular linear velocity.
	VSLinearVel

	// VSRotHAccel is vestibular rotational acceleration (horiz plane).
	VSRotHAccel

	// VSRotHVel is vestibular rotational velocity (horiz plane).
	VSRotHVel

	// VSRotHDir is abstracted current horiz rotational direction (vestibular timing).
	VSRotHDir

	// VMRotVel is full-field visual-motion rotation (horiz plane).
	VMRotVel
)

// Sense represents a sensory state.
type Sense struct {
	// Time is the timestamp (Cycles, ms) for this state.
	Time int

	// Senses has the bits of active senses.
	Senses Senses

	// Values are the values for each sense.
	Values []float32
}

// Init initializes state for given timestamp (ms).
func (a *Sense) Init(ms int) {
	a.Time = ms
	a.Senses = 0
	a.Time = 0
	a.Values = make([]float32, SensesN)
}

func (a *Sense) String() string {
	s := fmt.Sprintf("t:%d", a.Time)
	for ai := range a.Senses {
		if a.Senses.HasFlag(ai) {
			s += fmt.Sprintf(" %s_%g", ai.String(), a.Values[ai])
		}
	}
	return s
}

// SenseBuffer is a ring buffer for senses.
type SenseBuffer struct {
	// States are the action states.
	States []Sense

	// WriteIndex is the current write index where a new item will be
	// written (within range of States). Add post-increments.
	WriteIndex int
}

func (ab *SenseBuffer) Init(n int) {
	ab.States = make([]Sense, n)
	ab.WriteIndex = 0
	for i := range n {
		a := &ab.States[i]
		a.Init(0)
	}
}

// Add adds a new action at given time stamp (ms).
// Returns pointer to an existing state per ring buffer logic.
func (ab *SenseBuffer) Add(ms int) *Sense {
	a := &ab.States[ab.WriteIndex]
	a.Init(ms)
	ab.WriteIndex++
	if ab.WriteIndex >= len(ab.States) {
		ab.WriteIndex = 0
	}
	return a
}

// Prior returns action state that is given number of items
// prior to the last-added item. 0 = last-added item.
func (ab *SenseBuffer) Prior(nPrior int) *Sense {
	n := len(ab.States)
	ix := (ab.WriteIndex - 1) - nPrior
	for ix < 0 {
		ix += n
	}
	return &ab.States[ix]
}
