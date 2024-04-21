// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

import (
	"math/rand"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/erand"
)

// Valence
type Valence int32 //enums:enum

const (
	Pos Valence = iota
	Neg
)

// TickTypes
type TickTypes int32 //enums:enum

const (
	// Pre is before the CS
	Pre TickTypes = iota

	// CS is CS onset
	CS

	// Maint is after CS before US
	Maint

	// US is the US
	US

	// Post is after the US
	Post
)

// Sequence represents a sequence of ticks for one behavioral trial, unfolding over
// NTicks individual time steps, with one or more CS's (conditioned stimuli)
// and a US (unconditioned stimulus -- outcome).
type Sequence struct {

	// conventional suffixes: _R = reward, _NR = non-reward; _test = test trial (no learning)
	Name string

	// true if testing only -- no learning
	Test bool

	// Percent of all trials for this type
	Pct float32

	// Positive or negative reward valence
	Valence Valence

	// Probability of US
	USProb float32

	// Mixed US set?
	MixedUS bool

	// US magnitude
	USMag float32

	// Number of ticks for a sequence
	NTicks int

	// Conditioned stimulus
	CS string

	// Tick of CS start
	CSStart int

	// Tick of CS end
	CSEnd int

	// Tick of CS2 start: -1 for none
	CS2Start int

	// Tick of CS2 end: -1 for none
	CS2End int

	// Unconditioned stimulus
	US int

	// Tick for start of US presentation
	USStart int

	// Tick for end of US presentation
	USEnd int

	// Context -- typically same as CS -- if blank CS will be copied -- different in certain extinguishing contexts
	Context string

	// for rendered sequence, true if US active
	USOn bool

	// for rendered sequence, true if CS active
	CSOn bool

	// for rendered sequence, the tick type
	Type TickTypes
}

// Block represents a set of sequence types
type Block []*Sequence

func (cd *Block) Length() int {
	return len(*cd)
}

func (cd *Block) Append(seq *Sequence) {
	*cd = append(*cd, seq)
}

// SequenceReps generates repetitions of specific sequence types
// for given condition name, based on Pct of total blocks,
// and sets the USOn flag for proportion of sequences
// based on USProb probability.
// If Condition.Permute is true, order of all sequences are permuted.
// Gets the block name from the condition name.
func SequenceReps(condNm string) []*Sequence {
	var seqs []*Sequence
	cond := AllConditions[condNm]
	cond.Name = condNm
	block := AllBlocks[cond.Block]
	for _, seq := range block {
		if seq.Context == "" {
			seq.Context = seq.CS
		}
		nRpt := int(math32.Round(seq.Pct * float32(cond.NSequences)))
		if nRpt < 1 {
			if seq.Pct > 0.0 {
				nRpt = 1
			} else {
				continue // shouldn't happen
			}
		}
		useIsOnList := false
		var usIsOn []bool
		if cond.FixedProb {
			if seq.USProb != 0.0 && seq.USProb != 1.0 {
				useIsOnList = true
				pn := int(math32.Round(float32(nRpt) * seq.USProb))
				usIsOn = make([]bool, nRpt) // defaults to false
				for i := 0; i < pn; i++ {
					usIsOn[i] = true
				}
				rand.Shuffle(len(usIsOn), func(i, j int) {
					usIsOn[i], usIsOn[j] = usIsOn[j], usIsOn[i]
				})
			}
		}
		for ri := 0; ri < nRpt; ri++ {
			trlNm := seq.Name + "_" + seq.Valence.String()
			usOn := false
			if !useIsOnList {
				usOn = erand.BoolP32(seq.USProb, -1)
			} else {
				usOn = usIsOn[ri]
			}
			curSeq := &Sequence{}
			*curSeq = *seq
			curSeq.Name = trlNm
			curSeq.USOn = usOn
			seqs = append(seqs, curSeq)
		}
	}
	if cond.Permute {
		rand.Shuffle(len(seqs), func(i, j int) {
			seqs[i], seqs[j] = seqs[j], seqs[i]
		})
	}
	return seqs
}
