// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build multinet

package axon

func GlobalNetwork(ctx *Context) *Network {
	return Networks[ctx.NetIdxs.NetIdx]
}
