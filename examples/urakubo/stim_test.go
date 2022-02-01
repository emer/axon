package main

import (
	"fmt"
	"testing"
)

func Test_RGeStimForHz(t *testing.T) {
	for hz := 5; hz <= 150; hz += 5 {
		ge := RGeStimForHz(float32(hz))
		fmt.Printf("hz: %d   ge: %g\n", hz, ge)
	}
}
