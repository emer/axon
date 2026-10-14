// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
	"fmt"
	"math"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/lab/base/randx"
)

// Item represents one input item.
type Item struct {
	// Cats is the stimulus category for each channel.
	// [NChannels]
	Cats []int

	// Values is the input value for each feature in each channel (-1 = none).
	// [NChannels][NFeatures]
	Values [][]int

	// Plan is the plan number, which best matches the current input values.
	Plan int

	// PlanMatch are the Plan values that match the current input.
	// [NChannels][2], per-channel: [Feature,Value] or -1,-1 for none.
	PlanMatch [][]int

	// PlanNoMatch are the Plan values that don't match the current input.
	// [NChannels][2], per-channel: [Feature,Value] or -1,-1 for none.
	PlanNoMatch [][]int
}

func (itm *Item) String() string {
	s := fmt.Sprintf("C: %v ", itm.Cats)
	nc := len(itm.Values)
	nf := len(itm.Values[0])
	for ch := range nc {
		s += fmt.Sprintf("\n\t%d:%d[", ch, itm.Cats[ch])
		for ft := range nf {
			v := itm.Values[ch][ft]
			vs := "."
			if v >= 0 {
				vs = fmt.Sprintf("%d", v)
			}
			m := "  "
			if itm.PlanMatch[ch][0] == ft {
				if itm.PlanMatch[ch][1] == v {
					m = "* "
				} else {
					m = fmt.Sprintf("-%d", itm.PlanMatch[ch][1])
				}
			} else if itm.PlanNoMatch[ch][0] == ft {
				m = fmt.Sprintf("*%d", itm.PlanNoMatch[ch][1])
			}
			s += fmt.Sprintf("%s%s, ", vs, m)
		}
		s += "]"
	}
	return s
}

func (cfg *Config) NewItem() Item {
	itm := Item{}
	itm.Cats = make([]int, cfg.NChannels)
	itm.Values = make([][]int, cfg.NChannels)
	itm.PlanMatch = make([][]int, cfg.NChannels)
	itm.PlanNoMatch = make([][]int, cfg.NChannels)
	for ch := range cfg.NChannels {
		itm.Values[ch] = make([]int, cfg.NFeatures)
		itm.PlanMatch[ch] = make([]int, 2)
		itm.PlanNoMatch[ch] = make([]int, 2)
		for j := range 2 {
			itm.PlanMatch[ch][j] = -1
			itm.PlanNoMatch[ch][j] = -1
		}
	}
	return itm
}

// MakeItems makes lists of training and testing items.
// testing items have values that are not present in input.
func (ev *ConSatEnv) MakeItems() (train, test []Item) {
	cfg := &ev.Configs
	nch := cfg.NChannels
	nc := cfg.NCats
	nf := cfg.NFeatures
	np := cfg.NPlans
	nccs := int(math.Pow(float64(nc), float64(nch)))
	fmt.Printf("cat combinations across channels: %d^%d = %d\n", nc, nch, nccs)

	rndFs := ev.Rand.Perm(nf)

	nIn := func() int {
		return cfg.InputFeatureRange.Min + ev.Rand.Intn(cfg.InputFeatureRange.Range()+1)
	}

	fVal := func(c, f int) int {
		vls := cfg.Cats[c][f]
		n := len(vls)
		if n == 0 {
			return -1
		}
		i := ev.Rand.Intn(n)
		return vls[i]
	}

	items = make([]Item, 0, nccs)

	pfreq := make([]float32, np)

	for cc := range nccs {
		rem := cc
		itm := cfg.NewItem()
		for ch := range nch {
			c := rem % nc
			rem = rem / nc
			randx.PermuteInts(rndFs, ev.Rand)
			nin := nIn()
			itm.Cats[ch] = c
			for f := range nin {
				ft := rndFs[f]
				v := fVal(c, ft)
				itm.Values[ch][ft] = v
			}
			for f := nin; f < nf; f++ {
				itm.Values[ch][rndFs[f]] = -1
			}
		}
		// find best-fitting plan
		bplan := 0
		bfit := float32(0)
		for p := range np {
			fit := float32(0)
			for ch := range nch {
				ft := cfg.Plans[p][ch][0]
				pv := cfg.Plans[p][ch][1]
				v := itm.Values[ch][ft]
				if v >= 0 {
					if pv == v {
						fit += 1.0
					}
				} else {
					vls := cfg.Cats[itm.Cats[ch]][ft]
					nv := len(vls)
					for _, v := range vls {
						if v == pv {
							fit += 1.0 / float32(nv)
						}
					}
				}
				if fit > bfit {
					bfit = fit
					bplan = p
				}
			}
		}
		itm.Plan = bplan
		p := bplan
		for ch := range nch {
			ft := cfg.Plans[p][ch][0]
			pv := cfg.Plans[p][ch][1]
			v := itm.Values[ch][ft]
			// fmt.Println("ch:", ch, "ft:", ft, "pv:", pv, "v:", v)
			if v >= 0 {
				if pv == v {
					itm.PlanMatch[ch][0] = ft
					itm.PlanMatch[ch][1] = v
				}
			} else {
				vls := cfg.Cats[itm.Cats[ch]][ft]
				for _, v := range vls {
					if v == pv {
						itm.PlanNoMatch[ch][0] = ft
						itm.PlanNoMatch[ch][1] = v
					}
				}
			}
		}
		if Debug {
			fmt.Println("best Plan:", bplan, "fit:", bfit) // , "match:", itm.PlanMatch, "no:", itm.PlanNoMatch)
			fmt.Println(itm.String())
			if cc > 100 {
				break
			}
		}
		items = append(items, itm)
		pfreq[bplan] += 1.0
	}

	order = ev.Rand.Perm(nccs)
	errors.Log(jsonx.Save(items, ev.Filename("items")))
	errors.Log(jsonx.Save(order, ev.Filename("order")))

	// if Debug {
	fmt.Println("pfreq:", pfreq)
	// }

	return nil, nil
}
