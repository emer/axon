// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/core/tensor/stats/stats"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

// HipConfig have the hippocampus size and connectivity parameters
type HipConfig struct {

	// size of EC2
	EC2Size vecint.Vector2i `nest:"+"`

	// number of EC3 pools (outer dimension)
	EC3NPool vecint.Vector2i `nest:"+"`

	// number of neurons in one EC3 pool
	EC3NNrn vecint.Vector2i `nest:"+"`

	// number of neurons in one CA1 pool
	CA1NNrn vecint.Vector2i `nest:"+"`

	// size of CA3
	CA3Size vecint.Vector2i `nest:"+"`

	// size of DG / CA3
	DGRatio float32 `default:"2.236"`

	// percent connectivity from EC3 to EC2
	EC3ToEC2PCon float32 `default:"0.1"`

	// percent connectivity from EC2 to DG
	EC2ToDGPCon float32 `default:"0.25"`

	// percent connectivity from EC2 to CA3
	EC2ToCA3PCon float32 `default:"0.25"`

	// percent connectivity from CA3 to CA1
	CA3ToCA1PCon float32 `default:"0.25"`

	// percent connectivity into CA3 from DG
	DGToCA3PCon float32 `default:"0.02"`

	// lateral radius of connectivity in EC2
	EC2LatRadius int

	// lateral gaussian sigma in EC2 for how quickly weights fall off with distance
	EC2LatSigma float32

	// proportion of full mossy fiber strength (PathScale.Rel) for CA3 EDL in training, applied at the start of a trial to reduce DG -> CA3 strength.  1 = fully reduce strength, .5 = 50% reduction, etc
	MossyDelta float32 `default:"1"`

	// proportion of full mossy fiber strength (PathScale.Rel) for CA3 EDL in testing, applied during 2nd-3rd quarters to reduce DG -> CA3 strength.  1 = fully reduce strength, .5 = 50% reduction, etc
	MossyDeltaTest float32 `default:"0.75"`

	// low theta modulation value for temporal difference EDL -- sets PathScale.Rel on CA1 <-> EC paths consistent with Theta phase model
	ThetaLow float32 `default:"0.9"`

	// high theta modulation value for temporal difference EDL -- sets PathScale.Rel on CA1 <-> EC paths consistent with Theta phase model
	ThetaHigh float32 `default:"1"`

	// flag for clamping the EC5 from EC5ClampSrc
	EC5Clamp bool `default:"true"`

	// source layer for EC5 clamping activations in the plus phase -- biologically it is EC3 but can use an Input layer if available
	EC5ClampSrc string `default:"EC3"`

	// clamp the EC5 from EC5ClampSrc during testing as well as training -- this will overwrite any target values that might be used in stats (e.g., in the basic hip example), so it must be turned off there
	EC5ClampTest bool `default:"true"`

	// threshold for binarizing EC5 clamp values -- any value above this is clamped to 1, else 0 -- helps produce a cleaner learning signal.  Set to 0 to not perform any binarization.
	EC5ClampThr float32 `default:"0.1"`
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

	hip.EC5Clamp = true
	hip.EC5ClampSrc = "EC3"
	hip.EC5ClampTest = true
	hip.EC5ClampThr = 0.1
}

// AddHip adds a new Hippocampal network for episodic memory.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddHip(ctx *Context, hip *HipConfig, space float32) (ec2, ec3, dg, ca3, ca1, ec5 *Layer) {
	// Trisynaptic Pathway (TSP)
	ec2 = net.AddLayer2D("EC2", hip.EC2Size.Y, hip.EC2Size.X, SuperLayer)
	ec2.SetSampleIndexesShape(emer.Layer2DSampleIndexes(ec2, 10))
	dg = net.AddLayer2D("DG", int(float32(hip.CA3Size.Y)*hip.DGRatio), int(float32(hip.CA3Size.X)*hip.DGRatio), SuperLayer)
	dg.SetSampleIndexesShape(emer.Layer2DSampleIndexes(dg, 10))
	ca3 = net.AddLayer2D("CA3", hip.CA3Size.Y, hip.CA3Size.X, SuperLayer)
	ca3.SetSampleIndexesShape(emer.Layer2DSampleIndexes(ca3, 10))

	// Monosynaptic Pathway (MSP)
	ec3 = net.AddLayer4D("EC3", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, SuperLayer)
	ec3.AddClass("EC")
	ec3.SetSampleIndexesShape(emer.CenterPoolIndexes(ec3, 2), emer.CenterPoolShape(ec3, 2))
	ca1 = net.AddLayer4D("CA1", hip.EC3NPool.Y, hip.EC3NPool.X, hip.CA1NNrn.Y, hip.CA1NNrn.X, SuperLayer)
	ca1.SetSampleIndexesShape(emer.CenterPoolIndexes(ca1, 2), emer.CenterPoolShape(ca1, 2))
	if hip.EC5Clamp {
		ec5 = net.AddLayer4D("EC5", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, TargetLayer) // clamped in plus phase
	} else {
		ec5 = net.AddLayer4D("EC5", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, SuperLayer)
	}
	ec5.AddClass("EC")
	ec5.SetSampleIndexesShape(emer.CenterPoolIndexes(ec5, 2), emer.CenterPoolShape(ec5, 2))

	// Input and ECs connections
	onetoone := paths.NewOneToOne()
	ec3Toec2 := paths.NewUniformRand()
	ec3Toec2.PCon = hip.EC3ToEC2PCon
	mossy := paths.NewUniformRand()
	mossy.PCon = hip.DGToCA3PCon
	net.ConnectLayers(ec3, ec2, ec3Toec2, ForwardPath)
	net.ConnectLayers(ec5, ec3, onetoone, BackPath)

	// recurrent inhbition in EC2
	lat := paths.NewCircle()
	lat.TopoWeights = true
	lat.Radius = hip.EC2LatRadius
	lat.Sigma = hip.EC2LatSigma
	inh := net.ConnectLayers(ec2, ec2, lat, InhibPath)
	inh.AddClass("InhibLateral")

	// TSP connections
	ppathDG := paths.NewUniformRand()
	ppathDG.PCon = hip.EC2ToDGPCon
	ppathCA3 := paths.NewUniformRand()
	ppathCA3.PCon = hip.EC2ToCA3PCon
	ca3ToCA1 := paths.NewUniformRand()
	ca3ToCA1.PCon = hip.CA3ToCA1PCon
	full := paths.NewFull()
	net.ConnectLayers(ec2, dg, ppathDG, HipPath).AddClass("HippoCHL")
	net.ConnectLayers(ec2, ca3, ppathCA3, HipPath).AddClass("PPath")
	net.ConnectLayers(ca3, ca3, full, HipPath).AddClass("PPath")
	net.ConnectLayers(dg, ca3, mossy, ForwardPath).AddClass("HippoCHL")
	net.ConnectLayers(ca3, ca1, ca3ToCA1, HipPath).AddClass("HippoCHL")

	// MSP connections
	pool1to1 := paths.NewPoolOneToOne()
	net.ConnectLayers(ec3, ca1, pool1to1, HipPath).AddClass("EcCA1Path")     // HipPath makes wt linear
	net.ConnectLayers(ca1, ec5, pool1to1, ForwardPath).AddClass("EcCA1Path") // doesn't work w/ HipPath
	net.ConnectLayers(ec5, ca1, pool1to1, HipPath).AddClass("EcCA1Path")     // HipPath makes wt linear

	// positioning
	ec3.PlaceRightOf(ec2, space)
	ec5.PlaceRightOf(ec3, space)
	dg.PlaceAbove(ec2)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, space)

	return
}

// ConfigLoopsHip configures the hippocampal looper and should be included in ConfigLoops
// in model to make sure hip loops is configured correctly.
// see hip.go for an instance of implementation of this function.
// ec5ClampFrom specifies the layer to clamp EC5 plus phase values from:
// EC3 is the biological source, but can use Input layer for simple testing net.
func (net *Network) ConfigLoopsHip(ctx *Context, man *looper.Manager, hip *HipConfig, pretrain *bool) {
	var tmpValues []float32

	clampSrc := net.LayerByName(hip.EC5ClampSrc)
	ec5 := net.LayerByName("EC5")
	ca1 := net.LayerByName("CA1")
	ca3 := net.LayerByName("CA3")
	dg := net.LayerByName("DG")
	dgFromEc2 := errors.Log1(dg.RecvPathBySendName("EC2")).(*Path)
	ca1FromEc3 := errors.Log1(ca1.RecvPathBySendName("EC3")).(*Path)
	ca1FromCa3 := errors.Log1(ca1.RecvPathBySendName("CA3")).(*Path)
	ca3FromDg := errors.Log1(ca3.RecvPathBySendName("DG")).(*Path)
	ca3FromEc2 := errors.Log1(ca3.RecvPathBySendName("EC2")).(*Path)
	ca3FromCa3 := errors.Log1(ca3.RecvPathBySendName("CA3")).(*Path)

	dgPjScale := ca3FromDg.Params.PathScale.Rel
	ca1FromCa3Abs := ca1FromCa3.Params.PathScale.Abs

	// configure events -- note that events are shared between Train, Test
	// so only need to do it once on Train
	mode := etime.Train
	stack := man.Stacks[mode]
	cyc, _ := stack.Loops[etime.Cycle]
	minusStart := cyc.EventByName("MinusPhase")
	minusStart.OnEvent.Add("HipMinusPhase:Start", func() {
		if *pretrain {
			dgFromEc2.Params.Learn.Learn = 0
			ca3FromEc2.Params.Learn.Learn = 0
			ca3FromCa3.Params.Learn.Learn = 0
			ca1FromCa3.Params.Learn.Learn = 0
			ca1FromCa3.Params.PathScale.Abs = 0
		} else {
			dgFromEc2.Params.Learn.Learn = 1
			ca3FromEc2.Params.Learn.Learn = 1
			ca3FromCa3.Params.Learn.Learn = 1
			ca1FromCa3.Params.Learn.Learn = 1
			ca1FromCa3.Params.PathScale.Abs = ca1FromCa3Abs
		}
		ca1FromEc3.Params.PathScale.Rel = hip.ThetaHigh
		ca1FromCa3.Params.PathScale.Rel = hip.ThetaLow

		ca3FromDg.Params.PathScale.Rel = dgPjScale * (1 - hip.MossyDelta) // turn off DG input to CA3 in first quarter

		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	beta1 := cyc.EventByName("Beta1")
	beta1.OnEvent.Add("Hip:Beta1", func() {
		ca1FromEc3.Params.PathScale.Rel = hip.ThetaLow
		ca1FromCa3.Params.PathScale.Rel = hip.ThetaHigh
		if man.Mode == etime.Test {
			ca3FromDg.Params.PathScale.Rel = dgPjScale * (1 - hip.MossyDeltaTest)
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	plus := cyc.EventByName("PlusPhase")

	// note: critical for this to come before std start
	plus.OnEvent.InsertBefore("PlusPhase:Start", "HipPlusPhase:Start", func() {
		ca3FromDg.Params.PathScale.Rel = dgPjScale // restore at the beginning of plus phase for CA3 EDL
		ca1FromEc3.Params.PathScale.Rel = hip.ThetaHigh
		ca1FromCa3.Params.PathScale.Rel = hip.ThetaLow
		// clamp EC5 from clamp source (EC3 typically)
		if hip.EC5Clamp {
			if mode != etime.Test || hip.EC5ClampTest {
				for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
					clampSrc.UnitValues(&tmpValues, "Act", int(di))
					if hip.EC5ClampThr > 0 {
						stats.Binarize32(tmpValues, tensor.NewFloat64Scalar(hip.EC5ClampThr))
					}
					ec5.ApplyExt1D32(ctx, di, tmpValues)
				}
			}
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
		net.ApplyExts(ctx) // essential for GPU
	})

	trl := stack.Loops[etime.Trial]
	trl.OnEnd.Prepend("HipPlusPhase:End", func() {
		ca1FromCa3.Params.PathScale.Rel = hip.ThetaHigh
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
}
