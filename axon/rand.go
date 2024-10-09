package axon

import (
	"cogentcore.org/core/goal/gosl/slrand"
)

//gosl:start

type RandFunIndex uint32

// We use this enum to store a unique index for each function that
// requires random number generation. If you add a new function, you need to add
// a new enum entry here.
// RandFunIndexN is the total number of random functions. It autoincrements due to iota.
const (
	RandFunActPGe RandFunIndex = iota
	RandFunActPGi
	RandFunActSMaintP
	RandFunIndexN
)

// GetRandomNumber returns a random number that depends on the index,
// counter and function index.
// We increment the counter after each cycle, so that we get new random numbers.
// This whole scheme exists to ensure equal results under different multithreading settings.
func GetRandomNumber(index uint32, counter uint64, funIndex RandFunIndex) float32 {
	return slrand.Float32(counter, funIndex, index)
}

//gosl:end
