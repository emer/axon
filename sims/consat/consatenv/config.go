// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction environments.
package consatenv

import (
	"bytes"
	"fmt"
	"math"
	"slices"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/popcode"
)

var (
	defaultCats string = `[[[2],[0],[2,0],[0],[2],[0],[0,2,3],[1]],[[3],[2],[1,2,3,0],[0],[0],[3],[1,0],[2]],[[1],[3],[2],[3],[0,2,1,3],[2],[1,2,3,0],[2]],[[1,0,2,3],[0],[1],[2],[3],[3,2,0,1],[2],[3]],[[2,0,3,1],[2],[3],[1],[3],[2],[3],[3,2,1,0]],[[2],[3,0,1,2],[0],[1],[1],[3],[2,1],[0]]]`

	defaultPlans string = `[[[7,3],[3,2],[3,2],[2,1],[5,0],[4,1]],[[6,2],[7,3],[6,1],[7,0],[5,1],[3,0]],[[0,0],[6,0],[0,3],[2,3],[4,3],[1,2]],[[2,0],[0,2],[0,0],[4,3],[3,0],[5,2]],[[0,2],[5,0],[0,2],[2,2],[4,3],[0,1]],[[2,1],[0,3],[4,3],[5,0],[6,3],[3,1]],[[1,2],[2,3],[4,2],[4,3],[7,3],[3,2]],[[7,2],[4,1],[4,1],[6,3],[5,2],[6,2]],[[5,3],[6,2],[2,1],[1,0],[1,0],[5,0]],[[7,3],[0,1],[2,2],[7,2],[2,0],[6,2]],[[6,2],[7,0],[1,1],[0,3],[4,1],[7,2]],[[3,1],[1,2],[6,2],[0,0],[4,0],[5,1]],[[2,0],[3,1],[2,0],[1,2],[6,0],[7,3]],[[5,1],[7,1],[0,1],[6,2],[0,0],[0,1]],[[5,2],[3,0],[1,1],[7,3],[0,3],[4,0]],[[1,3],[1,3],[1,1],[3,1],[4,2],[1,2]],[[0,3],[5,2],[7,2],[5,3],[1,1],[7,2]],[[4,1],[2,0],[0,0],[0,3],[2,3],[3,1]],[[0,2],[7,3],[4,3],[4,3],[7,2],[7,1]],[[6,3],[0,0],[5,1],[0,0],[3,1],[5,0]],[[2,2],[5,0],[1,3],[0,1],[0,0],[2,3]],[[7,0],[2,3],[1,3],[7,3],[6,3],[5,3]],[[1,2],[6,3],[0,0],[5,2],[5,1],[6,0]],[[3,2],[2,1],[6,0],[2,1],[4,0],[4,0]]]`
)

// Config has configuration parameters for the environment.
type Config struct {

	// NChannels is the number of sensory input channels, where each
	// channel has NFeatures features with NValues values,
	// organized into NCats categories.
	NChannels int

	// Number of different sensory input features.
	// Each feature has 0-1 population-coded value. -1 means absent.
	// The 0-1 range is divided into NValues disjoint ranges for
	// each distinct Value of each Feature.
	// The combinatorial space is NValues^NFeatures.
	// Any given input will have an actual numerical input value
	// that is within the value range, with a proportional buffer
	// on either side for greater distinctiveness.
	NFeatures int

	// NValues is the number of discrete intervals of each feature
	// dedicated to each value of an input feature.
	NValues int

	// Proportion of total value range (range = 1/NValues) that is
	// not used for rendering input values on each side of range
	// (e.g., 0.1 = 10% on either side).
	ValueBuffer float32

	// Proportion of total value range (range = 1/NValues) to use
	// for the popcode sigma. 1 = full range, etc.
	ValueSigma float32

	// Number of low-level categories, where each category
	// is associated with a pattern of values across features.
	NCats int

	// NAmbigFeatures is the number of Features that are ambiguous
	// (i.e., have more than 1 value). Ambiguity is 2 to NValues..
	NAmbigFeatures int

	// Number of high-level plans. Each plan has a target feature and value
	// in each Channel.
	NPlans int

	// Range of number of features to input per channel, on any given input.
	InputFeatureRange minmax.Int

	// PopCodeUnits is the number of units to use for population code for features.
	PopCodeUnits int

	// population code for variables.
	PopCode popcode.OneD

	// Cats holds the list of values values for each feature,
	// to count as positive evidence for a given category.
	// only one value is ever generated at a given time.
	// [NCats][NFeatures][Value(s)]
	Cats [][][]int

	// Plans holds the list of values for each plan, one per channel.
	// The per-channel data is [Feature,Value]
	// [NPlans][NChannels][2]
	Plans [][][]int
}

func (cfg *Config) Defaults() {
	cfg.NChannels = 6
	cfg.NFeatures = 8 // 6 min
	cfg.NValues = 4   // lower = more avg overlap
	cfg.ValueBuffer = 0.1
	cfg.ValueSigma = 0.3
	cfg.NCats = 6 // 4 >> 6
	cfg.NAmbigFeatures = 2
	cfg.InputFeatureRange.Set(3, 3)
	cfg.NPlans = 24
	cfg.PopCodeUnits = 12
	cfg.PopCode.Defaults()
}

func (cfg *Config) ValRange() float32 {
	return 1.0 / float32(cfg.NValues)
}

func (cfg *Config) PopCodeSigma() float32 {
	return cfg.ValueSigma * cfg.ValRange()
}

// MakeCats randomly generates values for the sensory input
// categories. See [ConSatEnv.MakeBestCats] to find best.
func (ev *ConSatEnv) MakeCats() {
	cfg := &ev.Configs
	if len(cfg.Cats) != cfg.NCats {
		cfg.Cats = make([][][]int, cfg.NCats)
		for i := range cfg.NCats {
			cfg.Cats[i] = make([][]int, cfg.NFeatures)
		}
	}

	rndFs := ev.Rand.Perm(cfg.NFeatures)
	rndVs := ev.Rand.Perm(cfg.NValues)

	for i := range cfg.NCats {
		randx.PermuteInts(rndFs, ev.Rand)
		for j := range cfg.NAmbigFeatures {
			nv := 2 + ev.Rand.Intn(cfg.NValues-1)
			randx.PermuteInts(rndVs, ev.Rand)
			for k := range nv {
				if k == 0 {
					cfg.Cats[i][rndFs[j]] = []int{rndVs[k]}
				} else {
					cfg.Cats[i][rndFs[j]] = append(cfg.Cats[i][rndFs[j]], rndVs[k])
				}
			}
		}
		for j := cfg.NAmbigFeatures; j < cfg.NFeatures; j++ {
			v := ev.Rand.Intn(cfg.NValues)
			cfg.Cats[i][rndFs[j]] = []int{v}
		}
	}
}

// CatsDist returns the distance matrix for categories.
// and the average and max distances.
func (ev *ConSatEnv) CatsDist(idm *tensor.Float32) (dm *tensor.Float32, avg, mx float32) {
	cfg := &ev.Configs
	nc := cfg.NCats
	nf := cfg.NFeatures
	cf := cfg.Cats
	dm = idm
	if dm == nil {
		dm = tensor.NewFloat32(nc, nc)
	}
	for i := range nc {
		iavg := float32(0)
		for j := range nc {
			if j < i {
				ov := dm.Value(j, i)
				iavg += ov
				dm.Set(ov, i, j)
				continue
			}
			ov := float32(0)
			for f := range nf {
				inv := len(cf[i][f])
				for _, iv := range cf[i][f] {
					jnv := len(cf[j][f])
					for _, jv := range cf[j][f] {
						if iv == jv {
							ov += 1.0 / float32(inv*jnv)
						}
					}
				}
			}
			dm.Set(ov, i, j)
			if i == j {
				continue
			}
			iavg += ov
			mx = max(mx, ov)
		}
		iavg /= float32(nc - 1)
		avg = max(avg, iavg)
	}
	return
}

func (cfg *Config) cloneCats() [][][]int {
	cf := slices.Clone(cfg.Cats)
	for i := range cf {
		cf[i] = slices.Clone(cfg.Cats[i])
		for j := range cf[i] {
			cf[i][j] = slices.Clone(cfg.Cats[i][j])
		}
	}
	return cf
}

func (ev *ConSatEnv) CatsFilename() string {
	cfg := &ev.Configs
	return fmt.Sprintf("cats_nc%d_nf%d_nv%d.json", cfg.NCats, cfg.NFeatures, cfg.NValues)
}

// load shared global config
func (ev *ConSatEnv) InitOpen() {
	globalLock.Lock()
	defer globalLock.Unlock()
	if cats != nil {
		ev.Configs.Cats = cats
		return
	}
	b := bytes.NewBuffer([]byte(defaultCats))
	errors.Log(jsonx.Read(&cats, b))
	b = bytes.NewBuffer([]byte(defaultPlans))
	errors.Log(jsonx.Read(&plans, b))
	// items = ev.Rand.Perm(statesN)
}

// PrintCats prints the categories
func (cfg *Config) PrintCats(cf [][][]int) string {
	s := "Cat\tFeat\tValues\n"
	for i := range cfg.NCats {
		s += "---------------------------\n"
		for j := range cfg.NFeatures {
			s += fmt.Sprintf("%d\t%d\t%v\n", i, j, cf[i][j])
		}
	}
	return s
}

// MakeBestCats iterates for given number of tries to find the
// best category features in terms of max then average overlap.
func (ev *ConSatEnv) MakeBestCats(niter int) {
	cfg := &ev.Configs
	var dm, bdm *tensor.Float32
	var bcats [][][]int
	bmax := float32(math.MaxFloat32)
	bavg := float32(math.MaxFloat32)
	for itr := range niter {
		ev.MakeCats()
		dm, avg, mx := ev.CatsDist(dm)
		if mx <= bmax {
			if mx < bmax {
				bavg = float32(math.MaxFloat32) // start over
			}
			bmax = mx
			if avg < bavg {
				bavg = avg
				bcats = cfg.cloneCats()
				// bdm = tensor.CloneValues(dm)
				bdm = tensor.Clone(dm).(*tensor.Float32)
				fmt.Println("\t", itr, "\tmax:", bmax, "\tavg:", bavg)
			}
		}
	}
	bdms := tensor.Sprintf("%.3g", bdm, 0)
	fmt.Println("Best: max:", bmax, "\tavg:", bavg, "\tdm\n", bdms, cfg.PrintCats(bcats))
	errors.Log(jsonx.Save(bcats, ev.CatsFilename()))
}

//////// Plans

// MakePlans randomly generates plan features.
func (ev *ConSatEnv) MakePlans() {
	cfg := &ev.Configs
	if len(cfg.Plans) != cfg.NPlans {
		cfg.Plans = make([][][]int, cfg.NPlans)
		for i := range cfg.NPlans {
			cfg.Plans[i] = make([][]int, cfg.NChannels)
			for j := range cfg.NChannels {
				cfg.Plans[i][j] = make([]int, 2)
			}
		}
	}

	for i := range cfg.NPlans {
		for j := range cfg.NChannels {
			f := ev.Rand.Intn(cfg.NFeatures)
			v := ev.Rand.Intn(cfg.NValues)
			cfg.Plans[i][j][0] = f
			cfg.Plans[i][j][1] = v
		}
	}
}

// PlansDist returns the distance matrix for categories.
// and the average and max distances.
func (ev *ConSatEnv) PlansDist(idm *tensor.Float32) (dm *tensor.Float32, avg, mx float32) {
	cfg := &ev.Configs
	np := cfg.NPlans
	nc := cfg.NChannels
	pf := cfg.Plans
	dm = idm
	if dm == nil {
		dm = tensor.NewFloat32(np, np)
	}
	for i := range np {
		iavg := float32(0)
		for j := range np {
			if j < i {
				ov := dm.Value(j, i)
				iavg += ov
				dm.Set(ov, i, j)
				continue
			}
			ov := float32(0)
			for c := range nc {
				fi := pf[i][c][0]
				vi := pf[i][c][1]
				fj := pf[j][c][0]
				vj := pf[j][c][1]
				if fi == fj && vi == vj {
					ov += 1.0
				}
			}
			dm.Set(ov, i, j)
			if i == j {
				continue
			}
			iavg += ov
			mx = max(mx, ov)
		}
		iavg /= float32(np - 1)
		avg = max(avg, iavg)
	}
	return
}

// PrintPlans prints the plans
func (cfg *Config) PrintPlans(cf [][][]int) string {
	s := "Plan\tChan\tFeat:Value\n"
	for i := range cfg.NPlans {
		s += "---------------------------\n"
		for j := range cfg.NChannels {
			s += fmt.Sprintf("%d\t%d\t%d:%d\n", i, j, cf[i][j][0], cf[i][j][1])
		}
	}
	return s
}

func (cfg *Config) clonePlans() [][][]int {
	cf := slices.Clone(cfg.Plans)
	for i := range cf {
		cf[i] = slices.Clone(cfg.Plans[i])
		for j := range cf[i] {
			cf[i][j] = slices.Clone(cfg.Plans[i][j])
		}
	}
	return cf
}

func (ev *ConSatEnv) PlansFilename() string {
	cfg := &ev.Configs
	return fmt.Sprintf("plans_np%d_nc%d_nf%d_nv%d.json", cfg.NPlans, cfg.NChannels, cfg.NFeatures, cfg.NValues)
}

// MakeBestPlans iterates for given number of tries to find the
// best Plan features in terms of max then average overlap.
func (ev *ConSatEnv) MakeBestPlans(niter int) {
	cfg := &ev.Configs
	var dm, bdm *tensor.Float32
	var bplans [][][]int
	bmax := float32(math.MaxFloat32)
	bavg := float32(math.MaxFloat32)
	for itr := range niter {
		ev.MakePlans()
		dm, avg, mx := ev.PlansDist(dm)
		if mx <= bmax {
			bmax = mx
			if mx < bmax {
				bavg = float32(math.MaxFloat32) // start over
			}
			if avg < bavg {
				bavg = avg
				bplans = cfg.clonePlans()
				// bdm = tensor.CloneValues(dm)
				bdm = tensor.Clone(dm).(*tensor.Float32)
				fmt.Println("\t", itr, "\tmax:", bmax, "\tavg:", bavg)
			}
		}
	}
	bdms := tensor.Sprintf("%.3g", bdm, 0)
	fmt.Println("Best: max:", bmax, "\tavg:", bavg, "\tdm\n", bdms, cfg.PrintPlans(bplans))
	errors.Log(jsonx.Save(bplans, ev.PlansFilename()))
}
