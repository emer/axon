// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/prjn"
)

// AddHip adds a new Hippocampal network for episodic memory.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddHip(ctx *Context, latRadius int, latSigma float32, ec2SizeY, ec2SizeX, ec3NPoolY, ec3NPoolX, ec3NNrnY, ec3NNrnX, dgSizeY, dgSizeX, ca3SizeY, ca3SizeX, ca1NNrnY, ca1NNrnX int, inToEc2PCon, ec2ToCa3PCon, ec2ToDgPCon, dgToCa3PCon, ca3ToCa1PCon, ec3ToEc2PCon, space float32) (in, ec2, ec3, dg, ca3, ca1, ec5 *Layer) {
	in = net.AddLayer4D("Input", ec3NPoolY, ec3NPoolX, ec3NNrnY, ec3NNrnX, InputLayer)
	
	// Trisynaptic Pathway (TSP)
	ec2 = net.AddLayer2D("EC2", ec2SizeY, ec2SizeX, SuperLayer)
	dg = net.AddLayer2D("DG", dgSizeY, dgSizeX, SuperLayer)
	ca3 = net.AddLayer2D("CA3", ca3SizeY, ca3SizeX, SuperLayer)

	// Monosynaptic Pathway (MSP)
	ec3 = net.AddLayer4D("EC3", ec3NPoolY, ec3NPoolX, ec3NNrnY, ec3NNrnX, SuperLayer)
	ec3.SetClass("EC")
	ca1 = net.AddLayer4D("CA1", ec3NPoolY, ec3NPoolX, ca1NNrnY, ca1NNrnX, SuperLayer)
	ec5 = net.AddLayer4D("EC5", ec3NPoolY, ec3NPoolX, ec3NNrnY, ec3NNrnX, TargetLayer) // clamped in plus phase
	ec5.SetClass("EC")

	// Input and ECs connections
	onetoone := prjn.NewOneToOne()
	inToEc2 := prjn.NewUnifRnd()
	inToEc2.PCon = inToEc2PCon
	Ec3ToEc2 := prjn.NewUnifRnd()
	Ec3ToEc2.PCon = ec3ToEc2PCon
	mossy := prjn.NewUnifRnd()
	mossy.PCon = dgToCa3PCon
	net.ConnectLayers(in, ec2, inToEc2, ForwardPrjn)
	net.ConnectLayers(in, ec3, onetoone, ForwardPrjn)
	net.ConnectLayers(ec3, ec2, Ec3ToEc2, ForwardPrjn)
	net.ConnectLayers(ec5, ec3, onetoone, BackPrjn)

	// recurrent inhbition in EC2
	lat := prjn.NewCircle()
	lat.TopoWts = true
	lat.Radius = latRadius
	lat.Sigma = latSigma
	inh := net.ConnectLayers(ec2, ec2, lat, InhibPrjn)
	inh.SetClass("InhibLateral")

	// TSP connections
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = ec2ToDgPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = ec2ToCa3PCon
	ca3ToCa1 := prjn.NewUnifRnd()
	ca3ToCa1.PCon = ca3ToCa1PCon
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
	ec2.PlaceAbove(in)
	ec3.PlaceRightOf(ec2, space)
	ec5.PlaceRightOf(ec3, space)
	dg.PlaceAbove(ec2)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, space)
	
	return
}
