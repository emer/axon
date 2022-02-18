// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"

	"github.com/emer/etable/etensor"
)

// Funts are function tables for CaP and CaD integration levels
// allowing integration timecourse of CaP and CaD variables
// driven by CaM, only when there is a spike.
type Funts struct {
	CaP etensor.Float32 `view:"-" desc:"CaP timescale function table"`
	CaD etensor.Float32 `view:"-" desc:"CaD timescale function table"`
}

// FuntMap is a map of rendered Funts for different parameters.
// The parameters are printed as a string for the key.
// Many cases will use the same
var FuntMap map[string]*Funts

// ParamString returns the string for given parameters, space separated
func ParamString(params []float32) string {
	pstr := ""
	for _, p := range params {
		pstr += fmt.Sprintf("%g ", p)
	}
	return pstr
}

// FuntsForParams returns function tables for given parameters.
// Returns true if already existed, false if newly created.
func FuntsForParams(paramkey string) (*Funts, bool) {
	if FuntMap == nil {
		FuntMap = make(map[string]*Funts)
	}
	ft, ok := FuntMap[paramkey]
	if ok {
		return ft, ok
	}
	ft = &Funts{}
	FuntMap[paramkey] = ft
	return ft, false
}
