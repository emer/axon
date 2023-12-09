// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package main

import "goki.dev/mat32/v2"

// StimAttnSize is a list of stimuli manipulating the size of stimuli vs. attention
// it is the primary test of Reynolds & Heeger 2009 attentional dynamics.
// Presents two gabor filters at .25 vs. .75 positions, one of which is in attn focus.
// focus is LIP input at end, with -1 feature.
// small LIP = 0.09, large LIP = 0.30
// small V1 = 0.08, large V1 = 0.012
var StimAttnSizeAll = Stims{
	// small input, large attention
	StimSet{"InS_AtL_C0", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.0}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.0}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C1", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.1}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.1}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C2", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.2}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.2}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C4", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.4}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.4}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C5", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.5}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.5}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C7", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.7}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.7}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C8", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.8}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.8}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	StimSet{"InS_AtL_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.25, 0}}},
	// large input, small attention
	StimSet{"InL_AtS_C0", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.0}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.0}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C1", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.1}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.1}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C2", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.2}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.2}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C4", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.4}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.4}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C5", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.5}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.5}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C7", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.7}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.7}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C8", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.8}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.8}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
	StimSet{"InL_AtS_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.11, 0}}},
}

// StimAttnSizeDebug is a list of stimuli manipulating the size of stimuli vs. attention
// it is the primary test of Reynolds & Heeger 2009 attentional dynamics.
// Presents two gabor filters at .25 vs. .75 positions, one of which is in attn focus.
// focus is LIP input at end, with -1 feature.
// small LIP = 0.09, large LIP = 0.30
// small V1 = 0.08, large V1 = 0.12
var StimAttnSizeDebug = Stims{
	// small input, large attention
	StimSet{"InS_AtL_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	// large input, small attention
	StimSet{"InL_AtS_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
}

// StimAttnSizeC2UP has contrasts C2 and up
var StimAttnSizeC2Up = Stims{
	// small input, large attention
	StimSet{"InS_AtL_C2", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.2}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.2}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C4", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.4}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.4}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C5", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.5}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.5}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C7", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.7}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.7}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C8", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.8}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.8}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	StimSet{"InS_AtL_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.08, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.30, 0}}},
	// large input, small attention
	StimSet{"InL_AtS_C2", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.2}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.2}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C3", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.3}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C4", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.4}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.4}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C5", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.5}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.5}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C6", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.6}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C7", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.7}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.7}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C8", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.8}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.8}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
	StimSet{"InL_AtS_C9", []Stim{Stim{mat32.Vec2{.25, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.75, .5}, 2, 0.12, 0.9}, Stim{mat32.Vec2{.25, .5}, -1, 0.09, 0}}},
}
