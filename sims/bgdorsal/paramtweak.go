// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgdorsal

import (
	"fmt"
	"os"

	"cogentcore.org/core/base/errors"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/params"
)

var (
	defSearch   = params.Increment
	defTweakPct = float32(0.2)
)

var PSearch = axon.PathSearches{
	{Sel: "#DGPiToM1VM", Set: func(pt *axon.PathParams, val float32) {
		pt.PathScale.Abs = val
	}, Vals: func() []float32 {
		return params.TweakPct(2, defTweakPct)
	}},
}

// ParamSearch applies param search values for given index (in [0..n) range),
// saving a `job.label` file with the param value.
func (ss *Sim) ParamSearch(paramIndex int) error {
	lbl, err := axon.ApplyPathSearch(ss.Net, PSearch, paramIndex)
	if err != nil {
		return errors.Log(err)
	}
	err = os.WriteFile("job.label", []byte(lbl), 0666)
	if err != nil {
		errors.Log(err)
	} else {
		fmt.Println("Running Search:", lbl)
	}
	return nil
}
