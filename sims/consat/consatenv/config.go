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
	defaultLoFeats string = `[[[1],[3],[0,2,1],[2],[2],[3],[0],[3,2]],[[1],[1,2],[3],[3],[2],[2,1],[3],[1]],[[2],[2],[1],[0,3,1,2],[1],[0],[1,2,3,0],[3]],[[1,3,0],[3],[1],[0],[0],[2,0,1,3],[2],[2]],[[0],[1,2],[2],[1],[3],[2],[2,3,1,0],[3]],[[3],[1],[0],[1,0,2,3],[0],[1,3,0],[1],[0]]]`
)

// Config has configuration parameters for the environment.
type Config struct {

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
	NLoCats int

	// NLoCatAmbig is the number of Features that are ambiguous
	// (i.e., have more than 1 value). Ambiguity is 2 to NValues..
	NLoCatAmbig int

	// LoNIn is min / max (inclusive) number of input features per trial.
	LoNIn minmax.Int

	// Number of high-level categories, as in NLoCats.
	NHiCats int

	// Number of high-level category features of relevance.
	// Each high-level category only has one value that it
	// looks for within a given relevant feature, so that it
	// can focus top-down attention on that feature.
	NHiCatFeatures int

	// PopCodeUnits is the number of units to use for population code for features.
	PopCodeUnits int

	// population code for variables.
	PopCode popcode.OneD

	// LoFeatures holds the list of values values for each feature,
	// to count as positive evidence for a given category.
	// only one value is ever generated at a given time.
	// [NLoCats][NFeatures][Value(s)]
	LoFeatures [][][]int

	// HiFeatures holds the list of values for each feature,
	// to count as positive evidence for a given category.
	// for each NFeature, it is the value number that counts
	// or -1 if this category does not care about this feature.
	// [NHiCats][NFeatures]
	HiCatFeaures [][]int
}

func (cfg *Config) Defaults() {
	cfg.NFeatures = 8 // 6 min
	cfg.NValues = 4   // lower = more avg overlap
	cfg.ValueBuffer = 0.1
	cfg.ValueSigma = 0.3
	cfg.NLoCats = 6 // 4 >> 6
	cfg.NLoCatAmbig = 2
	cfg.LoNIn.Set(3, 3)
	cfg.NHiCats = 6
	cfg.NHiCatFeatures = 3
	cfg.PopCodeUnits = 12
	cfg.PopCode.Defaults()
}

func (cfg *Config) ValRange() float32 {
	return 1.0 / float32(cfg.NValues)
}

func (cfg *Config) PopCodeSigma() float32 {
	return cfg.ValueSigma * cfg.ValRange()
}

// MakeLoRandomFeatures randomly generates values for the Lo features.
func (ev *ConSatEnv) MakeLoRandomFeatures() {
	// todo: min overlap? or just generate and pick with best?
	cfg := &ev.Configs
	if len(cfg.LoFeatures) != cfg.NLoCats {
		cfg.LoFeatures = make([][][]int, cfg.NLoCats)
		for i := range cfg.NLoCats {
			cfg.LoFeatures[i] = make([][]int, cfg.NFeatures)
		}
	}

	rndFs := ev.Rand.Perm(cfg.NFeatures)
	rndVs := ev.Rand.Perm(cfg.NValues)

	for i := range cfg.NLoCats {
		randx.PermuteInts(rndFs, ev.Rand)
		for j := range cfg.NLoCatAmbig {
			nv := 2 + ev.Rand.Intn(cfg.NValues-1)
			randx.PermuteInts(rndVs, ev.Rand)
			for k := range nv {
				if k == 0 {
					cfg.LoFeatures[i][rndFs[j]] = []int{rndVs[k]}
				} else {
					cfg.LoFeatures[i][rndFs[j]] = append(cfg.LoFeatures[i][rndFs[j]], rndVs[k])
				}
			}
		}
		for j := cfg.NLoCatAmbig; j < cfg.NFeatures; j++ {
			v := ev.Rand.Intn(cfg.NValues)
			cfg.LoFeatures[i][rndFs[j]] = []int{v}
		}
	}
}

// LoCatDist returns the distance matrix for Lo categories.
// and the average and max distances.
func (ev *ConSatEnv) LoCatDist(idm *tensor.Float32) (dm *tensor.Float32, avg, mx float32) {
	cfg := &ev.Configs
	nc := cfg.NLoCats
	nf := cfg.NFeatures
	lf := cfg.LoFeatures
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
				inv := len(lf[i][f])
				for _, iv := range lf[i][f] {
					jnv := len(lf[j][f])
					for _, jv := range lf[j][f] {
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

func (cfg *Config) cloneLoFeatures() [][][]int {
	lf := slices.Clone(cfg.LoFeatures)
	for i := range lf {
		lf[i] = slices.Clone(cfg.LoFeatures[i])
		for j := range lf[i] {
			lf[i][j] = slices.Clone(cfg.LoFeatures[i][j])
		}
	}
	return lf
}

func (ev *ConSatEnv) Filename() string {
	cfg := &ev.Configs
	return fmt.Sprintf("cfg_nc%d_nf%d_nv%d.json", cfg.NLoCats, cfg.NFeatures, cfg.NValues)
}

// load shared global config
func (ev *ConSatEnv) InitOpen() {
	globalLock.Lock()
	defer globalLock.Unlock()
	if lofeats != nil {
		ev.Configs.LoFeatures = lofeats
		return
	}
	b := bytes.NewBuffer([]byte(defaultLoFeats))
	errors.Log(jsonx.Read(&lofeats, b))
	// items = ev.Rand.Perm(statesN)
}

// load shared global config
func (cfg *Config) PrintLoFeatures(lf [][][]int) string {
	s := "Cat\tFeat\tValues\n"
	for i := range cfg.NLoCats {
		s += "---------------------------\n"
		for j := range cfg.NFeatures {
			s += fmt.Sprintf("%d\t%d\t%v\n", i, j, lf[i][j])
		}
	}
	return s
}

// MakeBestLoRandomFeatures iterates for given number of tries to find the
// best Lo features in terms of max then average overlap.
func (ev *ConSatEnv) MakeBestLoRandomFeatures(niter int) {
	cfg := &ev.Configs
	var dm, bdm *tensor.Float32
	var bfeats [][][]int
	bmax := float32(math.MaxFloat32)
	bavg := float32(math.MaxFloat32)
	for itr := range niter {
		ev.MakeLoRandomFeatures()
		dm, avg, mx := ev.LoCatDist(dm)
		if mx <= bmax {
			bmax = mx
			if avg < bavg {
				bavg = avg
				bfeats = cfg.cloneLoFeatures()
				// bdm = tensor.CloneValues(dm)
				bdm = tensor.Clone(dm).(*tensor.Float32)
				fmt.Println("\t", itr, "\tmax:", bmax, "\tavg:", bavg)
			}
		}
	}
	bdms := tensor.Sprintf("%.3g", bdm, 0)
	fmt.Println("Best: max:", bmax, "\tavg:", bavg, "\tdm\n", bdms, cfg.PrintLoFeatures(bfeats))
	errors.Log(jsonx.Save(bfeats, ev.Filename()))
}
