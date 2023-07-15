// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/prjn"
)

// HipConfig have the hippocampus size and connectivity parameters
type HipConfig struct {
	EC2Size      evec.Vec2i `nest:"+" desc:"size of EC2"`
	EC3NPool     evec.Vec2i `nest:"+" desc:"number of EC3 pools (outer dimension)"`
	EC3NNrn      evec.Vec2i `nest:"+" desc:"number of neurons in one EC3 pool"`
	CA1NNrn      evec.Vec2i `nest:"+" desc:"number of neurons in one CA1 pool"`
	CA3Size      evec.Vec2i `nest:"+" desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	InToEc2PCon  float32    `desc:"percent connectivity from Input to EC2"`
	Ec3ToEc2PCon float32    `desc:"percent connectivity from EC3 to EC2"`
	Ec2ToDgPCon  float32    `desc:"percent connectivity from EC2 to DG"`
	Ec2ToCa3PCon float32    `desc:"percent connectivity from EC2 to CA3"`
	Ca3ToCa1PCon float32    `desc:"percent connectivity from CA3 to CA1"`
	DgToCa3PCon  float32    `desc:"percent connectivity into CA3 from DG"`

	Ec2LatRadius int     `desc:"lateral radius in EC2"`
	Ec2LatSigma  float32 `desc:"lateral sigma in EC2"`

	ECPctAct     float32 `desc:"percent activation in EC pool"`
	MossyDel     float32 `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32 `desc:"delta in mossy strength for testing (relative to base param)"`
	ThetaLow     float32 `desc:"theta low value"`
	ThetaHigh    float32 `desc:"theta low value"`
	MemThr       float64 `desc:"memory threshold"`
}

func (hip *HipConfig) Defaults() {
	// patgen
	hip.ECPctAct = 0.2

	// size
	hip.EC2Size.Set(21, 21) // 21
	hip.EC3NPool.Set(2, 3)
	hip.EC3NNrn.Set(7, 7)
	hip.CA1NNrn.Set(10, 10) // using MedHip now
	hip.CA3Size.Set(20, 20) // using MedHip now
	hip.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hip.Ec2ToDgPCon = 0.25
	hip.Ec2ToCa3PCon = 0.25
	hip.Ca3ToCa1PCon = 0.25
	hip.DgToCa3PCon = 0.02
	hip.InToEc2PCon = 0.25
	hip.Ec3ToEc2PCon = 0.1 // 0.1 for EC3-EC2 in WintererMaierWoznyEtAl17, not sure about Input-EC2

	// lateral
	hip.Ec2LatRadius = 2
	hip.Ec2LatSigma = 2

	// theta EDL in CA1
	hip.ThetaLow = 0.9 // doesn't have strong effect at low NTrials but shouldn't go too low (e.g., 0.3)
	hip.ThetaHigh = 1

	// EDL in CA3
	hip.MossyDel = 4
	hip.MossyDelTest = 3

	// memory threshold
	hip.MemThr = 0.34
}

// AddHip adds a new Hippocampal network for episodic memory.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddHip(ctx *Context, hip *HipConfig, space float32) (ec2, ec3, dg, ca3, ca1, ec5 *Layer) {
	// in = net.AddLayer4D("Input", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, InputLayer)

	// Trisynaptic Pathway (TSP)
	ec2 = net.AddLayer2D("EC2", hip.EC2Size.Y, hip.EC2Size.X, SuperLayer)
	dg = net.AddLayer2D("DG", int(float32(hip.CA3Size.Y)*hip.DGRatio), int(float32(hip.CA3Size.X)*hip.DGRatio), SuperLayer)
	ca3 = net.AddLayer2D("CA3", hip.CA3Size.Y, hip.CA3Size.X, SuperLayer)

	// Monosynaptic Pathway (MSP)
	ec3 = net.AddLayer4D("EC3", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, SuperLayer)
	ec3.SetClass("EC")
	ca1 = net.AddLayer4D("CA1", hip.EC3NPool.Y, hip.EC3NPool.X, hip.CA1NNrn.Y, hip.CA1NNrn.X, SuperLayer)
	ec5 = net.AddLayer4D("EC5", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, TargetLayer) // clamped in plus phase
	ec5.SetClass("EC")

	// Input and ECs connections
	onetoone := prjn.NewOneToOne()
	// inToEc2 := prjn.NewUnifRnd()
	// inToEc2.PCon = hip.InToEc2PCon
	Ec3ToEc2 := prjn.NewUnifRnd()
	Ec3ToEc2.PCon = hip.Ec3ToEc2PCon
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hip.DgToCa3PCon
	// net.ConnectLayers(in, ec2, inToEc2, ForwardPrjn)
	// net.ConnectLayers(in, ec3, onetoone, ForwardPrjn)
	net.ConnectLayers(ec3, ec2, Ec3ToEc2, ForwardPrjn)
	net.ConnectLayers(ec5, ec3, onetoone, BackPrjn)

	// recurrent inhbition in EC2
	lat := prjn.NewCircle()
	lat.TopoWts = true
	lat.Radius = hip.Ec2LatRadius
	lat.Sigma = hip.Ec2LatSigma
	inh := net.ConnectLayers(ec2, ec2, lat, InhibPrjn)
	inh.SetClass("InhibLateral")

	// TSP connections
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hip.Ec2ToDgPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hip.Ec2ToCa3PCon
	ca3ToCa1 := prjn.NewUnifRnd()
	ca3ToCa1.PCon = hip.Ca3ToCa1PCon
	full := prjn.NewFull()
	net.ConnectLayers(ec2, dg, ppathDG, HipPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ec2, ca3, ppathCA3, HipPrjn).SetClass("PPath")
	net.ConnectLayers(ca3, ca3, full, HipPrjn).SetClass("PPath")
	net.ConnectLayers(dg, ca3, mossy, ForwardPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ca3, ca1, ca3ToCa1, HipPrjn).SetClass("HippoCHL")

	// MSP connections
	pool1to1 := prjn.NewPoolOneToOne()
	net.ConnectLayers(ec3, ca1, pool1to1, HipPrjn).SetClass("EcCa1Prjn")     // HipPrjn makes wt linear
	net.ConnectLayers(ca1, ec5, pool1to1, ForwardPrjn).SetClass("EcCa1Prjn") // doesn't work w/ HipPrjn
	net.ConnectLayers(ec5, ca1, pool1to1, HipPrjn).SetClass("EcCa1Prjn")     // HipPrjn makes wt linear

	// positioning
	// ec2.PlaceAbove(in)
	ec3.PlaceRightOf(ec2, space)
	ec5.PlaceRightOf(ec3, space)
	dg.PlaceAbove(ec2)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, space)

	return
}
