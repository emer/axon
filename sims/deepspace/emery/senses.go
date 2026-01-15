// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/tensor"
)

// Senses are sensory inputs that unfold over time.
// Can also use to store abstracted sensory state.
type Senses int32 //enums:enum

const (
	// VSRotHVel is vestibular rotational head velocity (horiz plane).
	VSRotHVel Senses = iota

	// VMRotHVel is full-field visual-motion rotation (horiz plane).
	VMRotHVel

	// note: values below VSRotHDir are not rendered, only for reference

	// VSRotHDir is the ground-truth actual head direction (horiz plane).
	VSRotHDir

	// VSRotHAccel is vestibular rotational head acceleration (horiz plane).
	VSRotHAccel

	// VSLinearVel is vestibular linear velocity. This is not actually something
	// that can be sensed directly by the vestibular system: only linear accel.
	VSLinearVel

	// VSLinearAccel is vestibular linear acceleration.
	VSLinearAccel
)

// IsVestibular returns true if given sense is vestibular, else visual
func (s Senses) IsVestibular() bool {
	if s == VMRotHVel {
		return false
	}
	return true
}

// SenseMaxValues are expected max sensory value, for normalizing.
var SenseMaxValues = [SensesN]float32{.2, .2, 180, 10, 1, 10}

// ConfigSensoryDelays sets the sensory delays for each sense.
func (ev *EmeryEnv) ConfigSensoryDelays() {
	for s := range SensesN {
		if s.IsVestibular() {
			ev.SensoryDelays[s] = ev.Params.Delays.Vestibular
		} else {
			ev.SensoryDelays[s] = ev.Params.Delays.Visual
		}
	}
}

// RecordSenses records senses, every step.
func (ev *EmeryEnv) RecordSenses() {
	ev.Physics.Builder.RunSensors()
	if ev.Cycle.Cur%ev.Params.VisMotionInterval == 0 {
		// note: due to https://github.com/gfx-rs/wgpu/issues/8119
		// this is very slow in reading back images from the GPU
		// so until that is fixed, we need a reasonably large interval
		// which is generally fine once the time-averaging is taken into account.
		ev.VisMotion()
	}
	dir := ev.SenseData.Dir("Cycle")
	for di := range ev.NData {
		es := ev.EmeryState(di)
		for sense := range SensesN {
			snm := sense.String()
			val := es.SenseValues[sense]
			if sense.IsVestibular() {
				for t := range ev.Params.VisMotionInterval {
					val += ev.ReadData(dir, di, snm, t)
				}
				val /= float32(1 + float32(ev.Params.VisMotionInterval))
			}
			ev.WriteData(dir, di, snm, val)
		}
	}
}

// VisMotion updates the visual motion value based on last action.
func (ev *EmeryEnv) VisMotion() {
	eyesk := ev.Emery.EyeR.Skin
	imgs := ev.Physics.Scene.RenderFrom(eyesk, &ev.Camera)
	ev.Motion.RunImages(&ev.MotionImage, imgs...)
	full := ev.Motion.FullField
	for di := range ev.NData {
		es := ev.EmeryState(di)
		es.EyeRImage = imgs[di]
		eyelv := full.Value(di, 0, 1) - full.Value(di, 0, 0)
		ev.SetSenseValue(di, VMRotHVel, eyelv)
	}
}

// AverageSenses computes time-lagged sensory averages over SensoryWindow.
// These are the values that are actually rendered for input to the model.
func (ev *EmeryEnv) AverageSenses() {
	dir := ev.SenseData.Dir("Cycle")
	avgDir := ev.SenseData.Dir("Avg")
	avgBufSz := ev.BufferSize / 10
	for s := range SensesN {
		del := ev.SensoryDelays[s]
		for di := range ev.NData {
			es := ev.EmeryState(di)
			diName := ev.diName(di)
			ts := dir.Dir(diName).Float32(s.String(), ev.BufferSize)
			avg := float64(0)
			for t := range ev.SensoryWindow {
				pidx := ev.PriorIndex(t + del)
				avg += ts.Float1D(pidx)
			}
			avg /= float64(ev.SensoryWindow)
			es.SenseAverages[s] = float32(avg)
			nrm := float32(avg) * ev.SenseNorms[s]
			if math32.Abs(nrm) > 1 {
				nrm = math32.Sign(nrm)
			}
			es.SenseNormed[s] = nrm
			avgDir.Dir(diName).Float32(s.String(), avgBufSz).SetFloat1D(float64(nrm), ev.AvgWriteIndex)
		}
	}
	ev.AvgWriteIncr()
	// es := ev.EmeryState(0)
	// fmt.Println("avgs: ", es.SenseAverages)
	// fmt.Println("norms:", es.SenseNormed)
}

// RenderSenses renders sensory states for current sensory values.
func (ev *EmeryEnv) RenderSenses() {
	ev.AverageSenses()
	for s := range VSRotHDir { // only render below VSRotHDir ground truth
		for di := range ev.NData {
			es := ev.EmeryState(di)
			val := es.SenseNormed[s]
			ev.RenderValue(di, s.String(), val)
		}
	}
}

// VisVestibCorrelCycle returns the correlation between the visual (VMRotHVel)
// and vestibular (VSRotHVel) signals at the cycle level,
// for tuning the visual motion params (VSRotHVel is ground truth).
func (ev *EmeryEnv) VisVestibCorrelCycle(di int) float64 {
	dd := ev.SenseData.Dir("Cycle").Dir(ev.diName(di))
	vm := dd.Float32(VMRotHVel.String(), ev.BufferSize)
	madj := tensor.NewFloat32FromValues(vm.Values[ev.Params.VisMotionInterval:]...)
	vs := dd.Float32(VSRotHVel.String(), ev.BufferSize)
	sadj := tensor.NewFloat32FromValues(vs.Values[:ev.BufferSize-ev.Params.VisMotionInterval]...)
	cor := metric.Correlation(madj, sadj).Float1D(0)
	return cor
}

// VisVestibCorrelAvg returns the correlation between the visual (VMRotHVel)
// and vestibular (VSRotHVel) signals at the averaged and normalized level.
// for tuning the visual motion params (VSRotHVel is ground truth).
func (ev *EmeryEnv) VisVestibCorrelAvg(di int) float64 {
	avgBufSz := ev.BufferSize / 10
	dd := ev.SenseData.Dir("Avg").Dir(ev.diName(di))
	vm := dd.Float32(VMRotHVel.String(), avgBufSz)
	vs := dd.Float32(VSRotHVel.String(), avgBufSz)
	cor := metric.Correlation(vm, vs).Float1D(0)
	return cor
}

// SenseValue returns the given sensory value, either current or
// delayed.
func (ev *EmeryEnv) SenseValue(di int, sense Senses, delayed bool) float32 {
	nPrior := 0
	if delayed {
		nPrior = ev.SensoryDelays[sense]
	}
	val := ev.ReadData(ev.SenseData.Dir("Cycle"), di, sense.String(), nPrior)
	return val
}
