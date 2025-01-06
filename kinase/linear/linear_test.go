// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"testing"

	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
)

func TestLinear(t *testing.T) {
	var ls Linear
	ls.Defaults()
	ls.Cycles = 200
	ls.PlusCycles = 50
	ls.Update()
	ls.Init()
	ls.Run()
	ls.Data.SaveCSV("linear_data.tsv", tensor.Tab, table.Headers)
	ls.Regress()
}
