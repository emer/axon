// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"

	"cogentcore.org/core/cli"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
)

// Sim is the overall sim.
type Sim struct {

	// TD implements TD Q learning.
	TD TD

	// Env is the motor sequence env.
	Env MotorSeqEnv

	// Number of runs.
	Runs int `default:"10"`

	// Number of trials per epoch; one trial is a sequence through the env.
	Trials int `default:"128"`

	// Number of epochs per Run, max.
	Epochs int `default:"1000"`

	// PrintInterval is the interval in epochs to print data.
	PrintInterval int `default:"10"`

	// StopCrit is the stopping criterion in terms of average reward per epoch.
	StopCrit float32 `default:"0.98"`

	// LogEpochs records data by epoch
	LogEpochs bool

	// Debug prints detailed debug info
	Debug bool
}

func (sim *Sim) Defaults() {
	sim.TD.Defaults()
	sim.Env.Defaults()
	sim.Env.NUnitsPer = 1
}

func main() {
	sim := &Sim{}
	sim.Defaults()
	cli.SetFromDefaults(sim)
	opts := cli.DefaultOptions("SeqTD", "Motor Sequence TD Q Learning")
	opts.DefaultFiles = append(opts.DefaultFiles, "config.toml")
	cli.Run(opts, sim, RunSim)
}

func RunSim(sim *Sim) error {
	debug := sim.Debug
	logEpoch := sim.LogEpochs
	td := &sim.TD
	env := &sim.Env
	td.Config(env.SeqLen, env.NActions)
	env.NUnitsPer = 1
	env.Config(0)

	epcs := tensor.NewFloat64(sim.Runs)
	for run := range sim.Runs {
		td.Init()
		env.Init(run)
		finalRew := float32(0)
		finalEpoch := 0
		for epoch := range sim.Epochs {
			rewSum := float32(0)
			for trial := range sim.Trials {
				for step := range env.SeqLen {
					state := env.Trial.Cur
					action := td.Action(state)
					env.Step()
					next := env.Trial.Cur
					env.Action(strconv.Itoa(action), nil)
					if step == env.SeqLen-1 {
						td.UpdateFinal(state, action, env.Rew)
					} else {
						td.UpdateQ(state, action, next, 0) // no feedback during trials
					}
					if debug {
						fmt.Printf("%02d\t%05d\t%d\t%d\t%d\t%v\t%d\t%7.4f\n", run, trial, step, state, action, env.Correct, env.NCorrect, env.Rew)
					}
				}
				env.Step()
				rewSum += env.Rew
			}
			rewSum /= float32(sim.Trials)
			stop := rewSum >= sim.StopCrit
			if logEpoch && (stop || (epoch+1)%sim.PrintInterval == 0) {
				fmt.Printf("%02d\t%05d\tRew: %7.4f\tLRate: %7.4f\tEpsilon: %7.4f\n", run, epoch, rewSum, td.LRate.Current, td.Epsilon.Current)
			}
			td.EpochUpdate(epoch + 1)
			finalRew = rewSum
			finalEpoch = epoch
			if stop {
				break
			}
		}
		epcs.Set(float64(finalEpoch), run)
		fmt.Printf("%02d\tNEpochs: %d\tRew: %7.4f\n", run, finalEpoch, finalRew)
		if debug {
			fmt.Println("Final Q Weights:\n", td.Q.String())
		}
	}
	dir, _ := tensorfs.NewDir("Desc")
	stats.Describe(dir, epcs)
	dt := tensorfs.DirTable(dir, nil)
	fmt.Println(dt)

	return nil
}
