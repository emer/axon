package main

import (
	"github.com/emer/etable/etensor"
)

// TODO Move this to a library package like emer.

// WorldInterface is like env.Env.
type WorldInterface interface {
	// Init Initializes or reinitialize the world
	Init(details string)

	// StepN Updates n timesteps (e.g. milliseconds)
	StepN(n int)

	// Step 1
	Step()

	// GetObservationSpace describes the shape and names of what the model can expect as inputs. This will be constant across the run.
	GetObservationSpace() map[string][]int

	// GetActionSpace describes the shape and names of what the model can send as outputs. This will be constant across the run.
	GetActionSpace() map[string][]int

	// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
	Observe(name string) etensor.Tensor

	// ObserveWithShape Returns a tensor for the named modality. E.g. “x” or “vision” or “reward” but returns a specific shape, like having four eyes versus 2 eyes
	ObserveWithShape(name string, shape []int) etensor.Tensor

	// ObserveWithShape Returns a tensor for the named modality. E.g. “x” or “vision” or “reward” but returns a specific shape, like having four eyes versus 2 eyes
	ObserveWithShapeStride(name string, shape, stride []int) etensor.Tensor

	// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
	Action(action, details string)
	//ActionContinuous(action string, details []float32)

	// DecodeAndTakeAction takes an action specified by a tensor. // TODO Rename and also take in a string
	DecodeAndTakeAction(action string, vt *etensor.Float32) string

	// Done Returns true if episode has ended, e.g. when exiting maze
	Done() bool

	// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
	Info() string

	// Display displays environment to the user
	Display()
}
