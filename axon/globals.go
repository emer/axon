// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start globals

// GlobalVars are network-wide variables, such as neuromodulators, reward, drives, etc
// including the state for the Rubicon phasic dopamine model. These are stored
// in the Network.Globals float32 slice and corresponding global GPU slice.
type GlobalVars int32 //enums:enum

const (
	/////////////////////////////////////////
	// Reward

	// Rew is the external reward value.  Must also set HasRew flag when Rew is set,
	// otherwise it is ignored. This is computed by the Rubicon algorithm from US
	// inputs set by Net.Rubicon methods, and can be directly set in simpler RL cases.
	GvRew GlobalVars = iota

	// HasRew must be set to true (1) when an external reward / US input is present,
	// otherwise Rew is ignored.  This is also set when Rubicon BOA model gives up.
	// This drives ACh release in the Rubicon model.
	GvHasRew

	// RewPred is the reward prediction, computed by a special reward prediction layer,
	// e.g., the VSPatch layer in the Rubicon algorithm.
	GvRewPred

	// PrevPred is previous time step reward prediction, e.g., for TDPredLayer
	GvPrevPred

	// HadRew is HasRew state from the previous trial, copied from HasRew in NewState.
	// Used for updating Effort, Urgency at start of new trial.
	GvHadRew

	/////////////////////////////////////////
	// NeuroMod neuromodulators

	// DA is phasic dopamine that drives learning moreso than performance,
	// representing reward prediction error, signaled as phasic
	// increases or decreases in activity relative to a tonic baseline, which is
	// represented by a value of 0.  Released by the VTA (ventral tegmental area),
	// or SNc (substantia nigra pars compacta).
	GvDA

	// DAtonic is tonic dopamine, which has modulatory instead of learning effects.
	// Increases can drive greater propensity to engage in activities by biasing Go
	// vs No pathways in the basal ganglia, for example as a function of Urgency.
	GvDAtonic

	// ACh is acetylcholine, activated by salient events, particularly at the onset
	// of a reward / punishment outcome (US), or onset of a conditioned stimulus (CS).
	// Driven by BLA -> PPtg that detects changes in BLA activity, via LDTLayer type.
	GvACh

	// NE is norepinepherine -- not yet in use
	GvNE

	// Ser is serotonin -- not yet in use
	GvSer

	// AChRaw is raw ACh value used in updating global ACh value by LDTLayer.
	GvAChRaw

	// GoalMaint is the normalized (0-1) goal maintenance activity,
	// set in ApplyRubicon function at start of trial.
	// Drives top-down inhibition of LDT layer / ACh activity.
	GvGoalMaint

	/////////////////////////////////////////
	// VSMatrix gating and Rubicon Rew flags

	// VSMatrixJustGated is VSMatrix just gated (to engage goal maintenance
	// in PFC areas), set at end of plus phase.  This excludes any gating
	// happening at time of US.
	GvVSMatrixJustGated

	// VSMatrixHasGated is VSMatrix has gated since the last time HasRew was set
	// (US outcome received or expected one failed to be received).
	GvVSMatrixHasGated

	// CuriosityPoolGated is true if VSMatrixJustGated and the first pool
	// representing the curiosity / novelty drive gated. This can change the
	// giving up Effort.Max parameter.
	GvCuriosityPoolGated

	/////////////////////////////////////////
	// Time, Effort & Urgency

	// Time is the raw time counter, incrementing upward during goal engaged window.
	// This is also copied directly into NegUS[0] which tracks time, but we maintain
	// a separate effort value to make it clearer.
	GvTime

	// Effort is the raw effort counter, incrementing upward for each effort step
	// during goal engaged window.
	// This is also copied directly into NegUS[1] which tracks effort, but we maintain
	// a separate effort value to make it clearer.
	GvEffort

	// UrgencyRaw is the raw effort for urgency, incrementing upward from effort
	// increments per step when _not_ goal engaged.
	GvUrgencyRaw

	// Urgency is the overall urgency activity level (normalized 0-1),
	// computed from logistic function of GvUrgencyRaw.  This drives DAtonic
	// activity to increasingly bias Go firing.
	GvUrgency

	/////////////////////////////////////////
	// US / PV

	// HasPosUS indicates has positive US on this trial,
	// drives goal accomplishment logic and gating.
	GvHasPosUS

	// HadPosUS is state from the previous trial (copied from HasPosUS in NewState).
	GvHadPosUS

	// NegUSOutcome indicates that a phasic negative US stimulus was experienced,
	// driving phasic ACh, VSMatrix gating to reset current goal engaged plan (if any),
	// and phasic dopamine based on the outcome.
	GvNegUSOutcome

	// HadNegUSOutcome is state from the previous trial (copied from NegUSOutcome
	// in NewState)
	GvHadNegUSOutcome

	// PVposSum is the total weighted positive valence primary value
	// = sum of Weight * USpos * Drive
	GvPVposSum

	// PVpos is the normalized positive valence primary value
	// = (1 - 1/(1+PVposGain * PVposSum))
	GvPVpos

	// PVnegSum is the total weighted negative valence primary values including costs
	// = sum of Weight * Cost + Weight * USneg
	GvPVnegSum

	// PVpos is the normalized negative valence primary values, including costs
	// = (1 - 1/(1+PVnegGain * PVnegSum))
	GvPVneg

	// PVposEst is the estimated PVpos final outcome value
	// decoded from the network PVposFinal layer
	GvPVposEst

	// PVposVar is the estimated variance or uncertainty in the PVpos
	// final outcome value decoded from the network PVposFinal layer
	GvPVposVar

	// PVnegEst is the estimated PVneg final outcome value
	// decoded from the network PVnegFinal layer
	GvPVnegEst

	// PVnegVar is the estimated variance or uncertainty in the PVneg
	// final outcome value decoded from the network PVnegFinal layer
	GvPVnegVar

	// GoalDistEst is the estimate of distance to the goal, in trial step units,
	// decreasing down to 0 as the goal approaches.
	GvGoalDistEst

	// GoalDistPrev is the previous estimate of distance to the goal,
	// in trial step units, decreasing down to 0 as the goal approaches.
	GvGoalDistPrev

	// ProgressRate is the negative time average change in GoalDistEst,
	// i.e., positive values indicate continued approach to the goal,
	// while negative values represent moving away from the goal.
	GvProgressRate

	// GiveUpUtility is total GiveUp weight as a function of Cost
	GvGiveUpUtility

	// ContUtility is total Continue weight as a function of expected positive outcome PVposEst
	GvContUtility

	// GiveUpTiming is total GiveUp weight as a function of VSPatchPosSum * (1 - VSPatchPosVar)
	GvGiveUpTiming

	// ContTiming is total Continue weight as a function of (1 - VSPatchPosSum) * VSPatchPosVar
	GvContTiming

	// GiveUpProgress is total GiveUp weight as a function of ProgressRate
	GvGiveUpProgress

	// ContProgress is total Continue weight as a function of ProgressRate
	GvContProgress

	// GiveUpSum is total GiveUp weight: Utility + Timing + Progress
	GvGiveUpSum

	// ContSum is total Continue weight: Utility + Timing + Progress
	GvContSum

	// GiveUpProb is the probability of giving up: 1 / (1 + (GvContSum / GvGiveUpSum))
	GvGiveUpProb

	// GiveUp is true if a reset was triggered probabilistically based on GiveUpProb
	GvGiveUp

	// GaveUp is copy of GiveUp from previous trial
	GvGaveUp

	/////////////////////////////////////////
	// VSPatch prediction of PVpos net value

	// VSPatchPos is the net shunting input from VSPatch (PosD1, named PVi in original Rubicon)
	// computed as the Max of US-specific VSPatch saved values, subtracting D1 - D2.
	// This is also stored as GvRewPred.
	GvVSPatchPos

	// VSPatchPosThr is a thresholded version of GvVSPatchPos,
	// applying Rubicon.LHb.VSPatchNonRewThr threshold for non-reward trials.
	// This is the version used for computing DA.
	GvVSPatchPosThr

	// VSPatchPosRPE is the reward prediction error for the VSPatchPos reward prediction
	// without any thresholding applied, and only for PV events.
	// This is used to train the VSPatch, assuming a local feedback circuit that does
	// not have the effective thresholding used for the broadcast critic signal that
	// trains the rest of the network.
	GvVSPatchPosRPE

	// VSPatchPosSum is the sum of VSPatchPos over goal engaged trials,
	// representing the integrated prediction that the US is going to occur
	GvVSPatchPosSum

	// VSPatchPosPrev is the previous trial VSPatchPosSum
	GvVSPatchPosPrev

	// VSPatchPosVar is the integrated temporal variance of VSPatchPos over goal engaged trials,
	// which determines when the VSPatchPosSum has stabilized
	GvVSPatchPosVar

	/////////////////////////////////////////
	// LHb lateral habenula component of the Rubicon model -- does all US processing

	// computed LHb activity level that drives dipping / pausing of DA firing,
	// when VSPatch pos prediction > actual PV reward drive
	// or PVneg > PVpos
	GvLHbDip

	// LHbBurst is computed LHb activity level that drives bursts of DA firing,
	// when actual PV reward drive > VSPatch pos prediction
	GvLHbBurst

	// LHbPVDA is GvLHbBurst - GvLHbDip -- the LHb contribution to DA,
	// reflecting PV and VSPatch (PVi), but not the CS (LV) contributions
	GvLHbPVDA

	/////////////////////////////////////////
	// Amygdala CS / LV variables

	// CeMpos is positive valence central nucleus of the amygdala (CeM)
	// LV (learned value) activity, reflecting
	// |BLAposAcqD1 - BLAposExtD2|_+ positively rectified.
	// CeM sets Raw directly.  Note that a positive US onset even with no
	// active Drive will be reflected here, enabling learning about unexpected outcomes.
	GvCeMpos

	// CeMneg is negative valence central nucleus of the amygdala (CeM)
	// LV (learned value) activity, reflecting
	// |BLAnegAcqD2 - BLAnegExtD1|_+ positively rectified.  CeM sets Raw directly
	GvCeMneg

	/////////////////////////////////////////
	// VTA ventral tegmental area dopamine release

	// VtaDA is overall dopamine value reflecting all of the different inputs
	GvVtaDA

	/////////////////////////////////////////
	// Cost is Time, Effort etc costs

	// Cost are Time, Effort, etc costs, as normalized version of corresponding raw.
	// NCosts of them
	GvCost

	// CostRaw are raw, linearly incremented negative valence US outcomes,
	// this value is also integrated together with all US vals for PVneg
	GvCostRaw

	/////////////////////////////////////////
	// USneg is negative valence US
	//   allocated for Nitems

	// USneg are negative valence US outcomes, normalized version of raw.
	// NNegUSs of them
	GvUSneg

	// USnegRaw are raw, linearly incremented negative valence US outcomes,
	// this value is also integrated together with all US vals for PVneg
	GvUSnegRaw

	///////////////////////////////////////////////////////////
	// USpos, VSPatch

	// Drives are current drive state, updated with optional homeostatic
	// exponential return to baseline values.
	GvDrives

	// USpos are current positive-valence drive-satisfying input(s)
	// (unconditioned stimuli = US)
	GvUSpos

	// VSPatch is current reward predicting VSPatch (PosD1) values.
	GvVSPatchD1

	// VSPatch is current reward predicting VSPatch (PosD2) values.
	GvVSPatchD2

	// OFCposPTMaint is activity level of given OFCposPT maintenance pool
	// used in anticipating potential USpos outcome value.
	GvOFCposPTMaint

	// VSMatrixPoolGated indicates whether given VSMatrix pool gated
	// this is reset after last goal accomplished -- records gating since then.
	GvVSMatrixPoolGated

	// IMPORTANT: if GvVSMatrixPoolGated is not the last, need to update gosl defn below
)

//gosl:end globals

//gosl:hlsl globals
/*
static const GlobalVars GlobalVarsN = GvVSMatrixPoolGated + 1;
*/
//gosl:end globals
