// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//go:generate gosl  -exclude=Update,UpdateParams,Defaults github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax chans/chans.go chans kinase time.go neuron.go act.go learn.go layer.go gpu.go
