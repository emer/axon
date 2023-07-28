// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/prjn"
)

// HipConfig have the hippocampus size and connectivity parameters
type HipConfig struct {
	// model size
	EC2Size  evec.Vec2i `nest:"+" desc:"size of EC2"`
	EC3NPool evec.Vec2i `nest:"+" desc:"number of EC3 pools (outer dimension)"`
	EC3NNrn  evec.Vec2i `nest:"+" desc:"number of neurons in one EC3 pool"`
	CA1NNrn  evec.Vec2i `nest:"+" desc:"number of neurons in one CA1 pool"`
	CA3Size  evec.Vec2i `nest:"+" desc:"size of CA3"`
	DGRatio  float32    `def:"2.236" desc:"size of DG / CA3"`

	// pcon
	EC3ToEC2PCon float32 `def:"0.1" desc:"percent connectivity from EC3 to EC2"`
	EC2ToDGPCon  float32 `def:"0.25" desc:"percent connectivity from EC2 to DG"`
	EC2ToCA3PCon float32 `def:"0.25" desc:"percent connectivity from EC2 to CA3"`
	CA3ToCA1PCon float32 `def:"0.25" desc:"percent connectivity from CA3 to CA1"`
	DGToCA3PCon  float32 `def:"0.02" desc:"percent connectivity into CA3 from DG"`

	EC2LatRadius int     `desc:"lateral radius of connectivity in EC2"`
	EC2LatSigma  float32 `desc:"lateral gaussian sigma in EC2 for how quickly weights fall off with distance"`

	MossyDelta     float32 `def:"1" desc:"proportion of full mossy fiber strength (PrjnScale.Rel) for CA3 EDL in training, applied at the start of a trial to reduce DG -> CA3 strength.  1 = fully reduce strength, .5 = 50% reduction, etc"`
	MossyDeltaTest float32 `def:"0.75" desc:"proportion of full mossy fiber strength (PrjnScale.Rel) for CA3 EDL in testing, applied during 2nd-3rd quarters to reduce DG -> CA3 strength.  1 = fully reduce strength, .5 = 50% reduction, etc"`
	ThetaLow       float32 `def:"0.9" desc:"low theta modulation value for temporal difference EDL -- sets PrjnScale.Rel on CA1 <-> EC prjns consistent with Theta phase model"`
	ThetaHigh      float32 `def:"1" desc:"high theta modulation value for temporal difference EDL -- sets PrjnScale.Rel on CA1 <-> EC prjns consistent with Theta phase model"`
}

func (hip *HipConfig) Defaults() {
	// size
	hip.EC2Size.Set(21, 21) // 21
	hip.EC3NPool.Set(2, 3)
	hip.EC3NNrn.Set(7, 7)
	hip.CA1NNrn.Set(10, 10) // using MedHip now
	hip.CA3Size.Set(20, 20) // using MedHip now
	hip.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hip.EC2ToDGPCon = 0.25
	hip.EC2ToCA3PCon = 0.25
	hip.CA3ToCA1PCon = 0.25
	hip.DGToCA3PCon = 0.02
	hip.EC3ToEC2PCon = 0.1 // 0.1 for EC3-EC2 in WintererMaierWoznyEtAl17, not sure about Input-EC2

	// lateral
	hip.EC2LatRadius = 2
	hip.EC2LatSigma = 2

	hip.MossyDelta = 1
	hip.MossyDeltaTest = .75
	hip.ThetaLow = 0.9
	hip.ThetaHigh = 1
}

// AddHip adds a new Hippocampal network for episodic memory.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddHip(ctx *Context, hip *HipConfig, space float32) (ec2, ec3, dg, ca3, ca1, ec5 *Layer) {
	// Trisynaptic Pathway (TSP)
	ec2 = net.AddLayer2D("EC2", hip.EC2Size.Y, hip.EC2Size.X, SuperLayer)
	ec2.SetRepIdxsShape(emer.Layer2DRepIdxs(ec2, 10))
	dg = net.AddLayer2D("DG", int(float32(hip.CA3Size.Y)*hip.DGRatio), int(float32(hip.CA3Size.X)*hip.DGRatio), SuperLayer)
	dg.SetRepIdxsShape(emer.Layer2DRepIdxs(dg, 10))
	ca3 = net.AddLayer2D("CA3", hip.CA3Size.Y, hip.CA3Size.X, SuperLayer)
	ca3.SetRepIdxsShape(emer.Layer2DRepIdxs(ca3, 10))

	// Monosynaptic Pathway (MSP)
	ec3 = net.AddLayer4D("EC3", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, SuperLayer)
	ec3.SetClass("EC")
	ec3.SetRepIdxsShape(emer.CenterPoolIdxs(ec3, 2), emer.CenterPoolShape(ec3, 2))
	ca1 = net.AddLayer4D("CA1", hip.EC3NPool.Y, hip.EC3NPool.X, hip.CA1NNrn.Y, hip.CA1NNrn.X, SuperLayer)
	ca1.SetRepIdxsShape(emer.CenterPoolIdxs(ca1, 2), emer.CenterPoolShape(ca1, 2))
	ec5 = net.AddLayer4D("EC5", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, TargetLayer) // clamped in plus phase
	ec5.SetClass("EC")
	ec5.SetRepIdxsShape(emer.CenterPoolIdxs(ec5, 2), emer.CenterPoolShape(ec5, 2))

	// Input and ECs connections
	onetoone := prjn.NewOneToOne()
	ec3Toec2 := prjn.NewUnifRnd()
	ec3Toec2.PCon = hip.EC3ToEC2PCon
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hip.DGToCA3PCon
	net.ConnectLayers(ec3, ec2, ec3Toec2, ForwardPrjn)
	net.ConnectLayers(ec5, ec3, onetoone, BackPrjn)

	// recurrent inhbition in EC2
	lat := prjn.NewCircle()
	lat.TopoWts = true
	lat.Radius = hip.EC2LatRadius
	lat.Sigma = hip.EC2LatSigma
	inh := net.ConnectLayers(ec2, ec2, lat, InhibPrjn)
	inh.SetClass("InhibLateral")

	// TSP connections
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hip.EC2ToDGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hip.EC2ToCA3PCon
	ca3ToCA1 := prjn.NewUnifRnd()
	ca3ToCA1.PCon = hip.CA3ToCA1PCon
	full := prjn.NewFull()
	net.ConnectLayers(ec2, dg, ppathDG, HipPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ec2, ca3, ppathCA3, HipPrjn).SetClass("PPath")
	net.ConnectLayers(ca3, ca3, full, HipPrjn).SetClass("PPath")
	net.ConnectLayers(dg, ca3, mossy, ForwardPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ca3, ca1, ca3ToCA1, HipPrjn).SetClass("HippoCHL")

	// MSP connections
	pool1to1 := prjn.NewPoolOneToOne()
	net.ConnectLayers(ec3, ca1, pool1to1, HipPrjn).SetClass("EcCA1Prjn")     // HipPrjn makes wt linear
	net.ConnectLayers(ca1, ec5, pool1to1, ForwardPrjn).SetClass("EcCA1Prjn") // doesn't work w/ HipPrjn
	net.ConnectLayers(ec5, ca1, pool1to1, HipPrjn).SetClass("EcCA1Prjn")     // HipPrjn makes wt linear

	// positioning
	ec3.PlaceRightOf(ec2, space)
	ec5.PlaceRightOf(ec3, space)
	dg.PlaceAbove(ec2)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, space)

	return
}

// ConfigLoopsHip configures the hippocampal looper and should be included in ConfigLoops in model to make sure hip loops is configured correctly
// see hip.go for an instance of implementation of this function
func (net *Network) ConfigLoopsHip(ctx *Context, man *looper.Manager, hip *HipConfig, pretrain *bool) {

	var tmpVals []float32

	ec3 := net.AxonLayerByName("EC3")
	ec5 := net.AxonLayerByName("EC5")
	ca1 := net.AxonLayerByName("CA1")
	ca3 := net.AxonLayerByName("CA3")
	dg := net.AxonLayerByName("DG")
	dgFmEc2 := dg.SendName("EC2")
	ca1FmEc3 := ca1.SendName("EC3")
	ca1FmCa3 := ca1.SendName("CA3")
	ca3FmDg := ca3.SendName("DG")
	ca3FmEc2 := ca3.SendName("EC2")
	ca3FmCa3 := ca3.SendName("CA3")

	dgPjScale := ca3FmDg.Params.PrjnScale.Rel
	ca1FmCa3Abs := ca1FmCa3.Params.PrjnScale.Abs

	// configure events -- note that events are shared between Train, Test
	// so only need to do it once on Train
	mode := etime.Train
	stack := man.Stacks[mode]
	cyc, _ := stack.Loops[etime.Cycle]
	minusStart, _ := cyc.EventByName("MinusPhase")
	minusStart.OnEvent.Add("HipMinusPhase:Start", func() {
		if *pretrain {
			dgFmEc2.Params.Learn.Learn = 0
			ca3FmEc2.Params.Learn.Learn = 0
			ca3FmCa3.Params.Learn.Learn = 0
			ca1FmCa3.Params.Learn.Learn = 0
			ca1FmCa3.Params.PrjnScale.Abs = 0
		} else {
			dgFmEc2.Params.Learn.Learn = 1
			ca3FmEc2.Params.Learn.Learn = 1
			ca3FmCa3.Params.Learn.Learn = 1
			ca1FmCa3.Params.Learn.Learn = 1
			ca1FmCa3.Params.PrjnScale.Abs = ca1FmCa3Abs
		}
		ca1FmEc3.Params.PrjnScale.Rel = hip.ThetaHigh
		ca1FmCa3.Params.PrjnScale.Rel = hip.ThetaLow

		ca3FmDg.Params.PrjnScale.Rel = dgPjScale * (1 - hip.MossyDelta) // turn off DG input to CA3 in first quarter

		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	beta1, _ := cyc.EventByName("Beta1")
	beta1.OnEvent.Add("Hip:Beta1", func() {
		ca1FmEc3.Params.PrjnScale.Rel = hip.ThetaLow
		ca1FmCa3.Params.PrjnScale.Rel = hip.ThetaHigh
		if man.Mode == etime.Test {
			ca3FmDg.Params.PrjnScale.Rel = dgPjScale * (1 - hip.MossyDeltaTest)
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	plus, _ := cyc.EventByName("PlusPhase")
	// note: critical for this to come before std start
	plus.OnEvent.InsertBefore("PlusPhase:Start", "HipPlusPhase:Start", func() {
		ca3FmDg.Params.PrjnScale.Rel = dgPjScale // restore at the beginning of plus phase for CA3 EDL
		ca1FmEc3.Params.PrjnScale.Rel = hip.ThetaHigh
		ca1FmCa3.Params.PrjnScale.Rel = hip.ThetaLow
		if man.Mode == etime.Train { // clamp EC5 from Input
			for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
				ec3.UnitVals(&tmpVals, "Act", int(di))
				ec5.ApplyExt1D32(ctx, di, tmpVals)
			}
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
		net.ApplyExts(ctx) // essential for GPU
	})

	trl := stack.Loops[etime.Trial]
	trl.OnEnd.Prepend("HipPlusPhase:End", func() {
		ca1FmCa3.Params.PrjnScale.Rel = hip.ThetaHigh
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
}
