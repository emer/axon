package main

import (
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// Ra25Env stores random patterns that can be memorized. There is no internal structure between the input and output.
type Ra25Env struct {
	agent.WorldInterface

	PatternSize int `desc:"The size of input and output patterns, which are square. Defaults to 5."`
	NumPatterns int `desc:"Number of patterns. Defaults to 30."`

	patterns []struct {
		Input  etensor.Tensor
		Output etensor.Tensor
	}
	patternIdx int `desc:"The index of the current pattern in Patterns."`
}

// InitWorld creates and stores patterns.
func (world *Ra25Env) InitWorld(details map[string]string) (actionSpace map[string]agent.SpaceSpec, observationSpace map[string]agent.SpaceSpec) {
	if world.PatternSize == 0 {
		world.PatternSize = 5
	}
	if world.NumPatterns == 0 {
		world.NumPatterns = 30
	}

	// Create patterns with the PermutedBinaryRows function.
	for i := 0; i < world.NumPatterns; i++ {
		in := etensor.New(etensor.FLOAT32, []int{world.PatternSize, world.PatternSize}, nil, []string{"Y", "X"})
		patgen.PermutedBinary(in, 6, 1, 0)
		out := etensor.New(etensor.FLOAT32, []int{world.PatternSize, world.PatternSize}, nil, []string{"Y", "X"})
		patgen.PermutedBinary(out, 6, 1, 0)
		world.patterns = append(world.patterns, struct {
			Input  etensor.Tensor
			Output etensor.Tensor
		}{in, out})
	}
	fivebyfive := agent.SpaceSpec{
		ContinuousShape: []int{5, 5},
		Min:             0,
		Max:             1,
	}
	return map[string]agent.SpaceSpec{"Output": fivebyfive}, map[string]agent.SpaceSpec{"Input": fivebyfive, "Output": fivebyfive}
}

// StepWorld steps the index of the current pattern.
func (world *Ra25Env) StepWorld(actions map[string]agent.Action, agentDone bool) (worldDone bool, debug string) {
	// TODO Evaluate the performance of actions.
	world.patternIdx += 1
	if world.patternIdx >= world.NumPatterns {
		world.patternIdx = 0
		return true, ""
	}
	return false, ""
}

// Observe returns an observation from the cache.
func (world *Ra25Env) Observe(name string) etensor.Tensor {
	if name == "Input" {
		return world.patterns[world.patternIdx].Input
	}
	if name == "Output" {
		return world.patterns[world.patternIdx].Output
	}
	return nil
}
