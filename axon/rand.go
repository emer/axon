package axon

import (
	"goki.dev/gosl/v2/slrand"
)

//gosl: hlsl axonrand
// #include "slrand.hlsl"
//gosl: end axonrand

//gosl: start axonrand

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
	// todo: gpu needs to have the shortcut to work directly on uint2
	var randCtr slrand.Counter
	randCtr = counter
	randCtr.Add(uint32(funIdx))
	ctr := randCtr.Uint2()
	return slrand.Float(&ctr, index)
}

//gosl: end axonrand
