package axon

import (
	"github.com/goki/gosl/slrand"
)

type RandFunIdx uint32

// We use this enum to store a unique index for each function that
// requires random number generation. If you add a new function, you need to add
// a new enum entry here.
// RandFunIdxN is the total number of random functions. It autoincrements due to iota.
const (
	RandFunActPGe RandFunIdx = iota
	RandFunActPGi
	RandFunIdxN
)

// GetRandomNumber returns a random number that depends on the index, counter and function index.
// We increment the counter after each cycle, so that we get new random numbers.
// This whole scheme exists to ensure equal results under different multithreading settings.
func GetRandomNumber(index uint32, counter slrand.Counter, funIdx RandFunIdx) float32 {
	randCtr := counter.Uint2()
	slrand.CounterAdd(&randCtr, uint32(funIdx))
	return slrand.Float(&randCtr, index)
}
