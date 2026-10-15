// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
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

	// UnitsPer is the number of units to use for discrete features (plans)
	UnitsPer int

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
	cfg.NChannels = 4
	cfg.NFeatures = 8 // 6 min
	cfg.NValues = 4   // lower = more avg overlap
	cfg.ValueBuffer = 0.0
	cfg.ValueSigma = 0.5
	cfg.NCats = 6 // 6 >> 4
	cfg.NAmbigFeatures = 2
	cfg.InputFeatureRange.Set(3, 3)
	cfg.NPlans = 24
	cfg.PopCodeUnits = 12
	cfg.PopCode.Defaults()
	cfg.UnitsPer = 4
}

func (cfg *Config) ValRange() float32 {
	return 1.0 / float32(cfg.NValues)
}

func (cfg *Config) PopCodeSigma() float32 {
	return cfg.ValueSigma * cfg.ValRange()
}

// ValMinMax returns the min / max value for encoding given int value.
func (ev *ConSatEnv) ValMinMax(v int) (vmin, vmax float32) {
	cfg := &ev.Configs
	vr := cfg.ValRange()
	vmin = (float32(v) + cfg.ValueBuffer) * vr
	vmax = vmin + vr*(1-cfg.ValueBuffer)
	return
}

// RandVal returns a random rendering value for given discrete input value.
func (ev *ConSatEnv) RandVal(v int) float32 {
	vmin, vmax := ev.ValMinMax(v)
	fv := vmin + ev.Rand.Float32()*(vmax-vmin)
	return fv
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

	for c := range cfg.NCats {
		randx.PermuteInts(rndFs, ev.Rand)
		for f := range cfg.NAmbigFeatures {
			nv := 2 + ev.Rand.Intn(cfg.NValues-1)
			randx.PermuteInts(rndVs, ev.Rand)
			for k := range nv {
				if k == 0 {
					cfg.Cats[c][rndFs[f]] = []int{rndVs[k]}
				} else {
					cfg.Cats[c][rndFs[f]] = append(cfg.Cats[c][rndFs[f]], rndVs[k])
				}
			}
		}
		for f := cfg.NAmbigFeatures; f < cfg.NFeatures; f++ {
			v := ev.Rand.Intn(cfg.NValues)
			cfg.Cats[c][rndFs[f]] = []int{v}
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
	for c := range nc {
		iavg := float32(0)
		for j := range nc {
			if j < c {
				ov := dm.Value(j, c)
				iavg += ov
				dm.Set(ov, c, j)
				continue
			}
			ov := float32(0)
			for f := range nf {
				inv := len(cf[c][f])
				for _, iv := range cf[c][f] {
					jnv := len(cf[j][f])
					for _, jv := range cf[j][f] {
						if iv == jv {
							ov += 1.0 / float32(inv*jnv)
						}
					}
				}
			}
			dm.Set(ov, c, j)
			if c == j {
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
	for c := range cf {
		cf[c] = slices.Clone(cfg.Cats[c])
		for j := range cf[c] {
			cf[c][j] = slices.Clone(cfg.Cats[c][j])
		}
	}
	return cf
}

func (ev *ConSatEnv) Filename(prefix string) string {
	cfg := &ev.Configs
	return fmt.Sprintf("%s_np%d_nc%d_nf%d_nv%d.json", prefix, cfg.NPlans, cfg.NChannels, cfg.NFeatures, cfg.NValues)
}

// load shared global config
func (ev *ConSatEnv) InitOpen() {
	globalLock.Lock()
	defer globalLock.Unlock()

	if cats == nil {
		errors.Log(jsonx.OpenFS(&cats, embedfs, ev.Filename("cats")))
		errors.Log(jsonx.OpenFS(&plans, embedfs, ev.Filename("plans")))
		errors.Log(jsonx.OpenFS(&items, embedfs, ev.Filename("items")))
		errors.Log(jsonx.OpenFS(&order, embedfs, ev.Filename("order")))
	}

	cfg := &ev.Configs
	cfg.Cats = cats
	cfg.Plans = plans
}

// PrintCats prints the categories
func (cfg *Config) PrintCats(cf [][][]int) string {
	s := "Cat\tFeat\tValues\n"
	for c := range cfg.NCats {
		s += "---------------------------\n"
		for j := range cfg.NFeatures {
			s += fmt.Sprintf("%d\t%d\t%v\n", c, j, cf[c][j])
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
	cfg.Cats = bcats
	bdms := tensor.Sprintf("%.3g", bdm, 0)
	fmt.Println("Best: max:", bmax, "\tavg:", bavg, "\tdm\n", bdms, cfg.PrintCats(bcats))
	errors.Log(jsonx.Save(bcats, ev.Filename("cats")))
}

//////// Plans

// MakePlans randomly generates plan features.
func (ev *ConSatEnv) MakePlans() {
	cfg := &ev.Configs
	if len(cfg.Plans) != cfg.NPlans {
		cfg.Plans = make([][][]int, cfg.NPlans)
		for p := range cfg.NPlans {
			cfg.Plans[p] = make([][]int, cfg.NChannels)
			for j := range cfg.NChannels {
				cfg.Plans[p][j] = make([]int, 2)
			}
		}
	}

	for p := range cfg.NPlans {
		for j := range cfg.NChannels {
			f := ev.Rand.Intn(cfg.NFeatures)
			v := ev.Rand.Intn(cfg.NValues)
			cfg.Plans[p][j][0] = f
			cfg.Plans[p][j][1] = v
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
	for p := range np {
		iavg := float32(0)
		for j := range np {
			if j < p {
				ov := dm.Value(j, p)
				iavg += ov
				dm.Set(ov, p, j)
				continue
			}
			ov := float32(0)
			for c := range nc {
				fi := pf[p][c][0]
				vi := pf[p][c][1]
				fj := pf[j][c][0]
				vj := pf[j][c][1]
				if fi == fj && vi == vj {
					ov += 1.0
				}
			}
			dm.Set(ov, p, j)
			if p == j {
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
	for p := range cfg.NPlans {
		s += "---------------------------\n"
		for j := range cfg.NChannels {
			s += fmt.Sprintf("%d\t%d\t%d:%d\n", p, j, cf[p][j][0], cf[p][j][1])
		}
	}
	return s
}

func (cfg *Config) clonePlans() [][][]int {
	cf := slices.Clone(cfg.Plans)
	for p := range cf {
		cf[p] = slices.Clone(cfg.Plans[p])
		for j := range cf[p] {
			cf[p][j] = slices.Clone(cfg.Plans[p][j])
		}
	}
	return cf
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
	cfg.Plans = bplans
	bdms := tensor.Sprintf("%.3g", bdm, 0)
	fmt.Println("Best: max:", bmax, "\tavg:", bavg, "\tdm\n", bdms, cfg.PrintPlans(bplans))
	errors.Log(jsonx.Save(bplans, ev.Filename("plans")))
}
