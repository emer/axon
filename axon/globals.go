// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/goki/ki/kit"

//go:generate stringer -type=GlobalVars

var KiT_GlobalVars = kit.Enums.AddEnum(GlobalVarsN, kit.NotBitFlag, nil)

func (ev GlobalVars) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *GlobalVars) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

//gosl: start globals

// GlobalVars are network-wide variables, such as neuromodulators, reward, drives, etc
// including the state for the PVLV phasic dopamine model.
type GlobalVars int32

const (
	/////////////////////////////////////////
	// Reward

	// Rew is reward value -- this is set here in the Context struct, and the RL Rew layer grabs it from there -- must also set HasRew flag when rew is set -- otherwise is ignored.
	GvRew GlobalVars = iota

	// HasRew must be set to true when a reward is present -- otherwise Rew is ignored.  Also set when PVLV BOA model gives up.  This drives ACh release in the PVLV model.
	GvHasRew

	// RewPred is reward prediction -- computed by a special reward prediction layer
	GvRewPred

	// PrevPred is previous time step reward prediction -- e.g., for TDPredLayer
	GvPrevPred

	// HadRew is HasRew state from the previous trial -- copied from HasRew in NewState -- used for updating Effort, Urgency at start of new trial
	GvHadRew

	/////////////////////////////////////////
	// NeuroMod neuromodulators

	// DA is dopamine -- represents reward prediction error, signaled as phasic increases or decreases in activity relative to a tonic baseline, which is represented by a value of 0.  Released by the VTA -- ventral tegmental area, or SNc -- substantia nigra pars compacta.
	GvDA

	// ACh is acetylcholine -- activated by salient events, particularly at the onset of a reward / punishment outcome (US), or onset of a conditioned stimulus (CS).  Driven by BLA -> PPtg that detects changes in BLA activity, via LDTLayer type
	GvACh

	// NE is norepinepherine -- not yet in use
	GvNE

	// Ser is serotonin -- not yet in use
	GvSer

	// AChRaw is raw ACh value used in updating global ACh value by LDTLayer
	GvAChRaw

	// NotMaint is activity of the PTNotMaintLayer -- drives top-down inhibition of LDT layer / ACh activity.
	GvNotMaint

	/////////////////////////////////////////
	// VSMatrix gating and PVLV Rew flags

	// VSMatrixJustGated is VSMatrix just gated (to engage goal maintenance in PFC areas), set at end of plus phase -- this excludes any gating happening at time of US
	GvVSMatrixJustGated

	// VSMatrixHasGated is VSMatrix has gated since the last time HasRew was set (US outcome received or expected one failed to be received
	GvVSMatrixHasGated

	// CuriosityPoolGated is true if VSMatrixJustGated and the first pool representing the curiosity / novelty drive gated -- this can change the giving up Effort.Max parameter.
	GvCuriosityPoolGated

	/////////////////////////////////////////
	// Time, Effort & Urgency

	// Time is raw time counter, incrementing upward during goal engaged window.
	// This is also copied directly into NegUS[0] which tracks time, but we maintain
	// a separate effort value to make it clearer.
	GvTime

	// Effort is raw effort counter -- incrementing upward for each effort step
	// during goal engaged window.
	// This is also copied directly into NegUS[1] which tracks effort, but we maintain
	// a separate effort value to make it clearer.
	GvEffort

	// UrgencyRaw is raw effort for urgency -- incrementing upward from effort
	// increments per step when _not_ goal engaged
	GvUrgencyRaw

	// Urgency is the overall urgency activity level (normalized 0-1),
	// computed from logistic function of GvUrgencyRaw
	GvUrgency

	/////////////////////////////////////////
	// US / PV

	// HasPosUS indicates has positive US on this trial -- drives goal accomplishment logic
	// and gating.
	GvHasPosUS

	// HadPosUS is state from the previous trial (copied from HasPosUS in NewState).
	GvHadPosUS

	// NegUSOutcome indicates that a strong negative US stimulus was experienced,
	// driving phasic ACh, VSMatrix gating to reset current goal engaged plan (if any),
	// and phasic dopamine based on the outcome.
	GvNegUSOutcome

	// HadNegUSOutcome is state from the previous trial (copied from NegUSOutcome in NewState)
	GvHadNegUSOutcome

	// PVposSum is total weighted positive valence primary value = sum of Weight * USpos * Drive
	GvPVposSum

	// PVpos is normalized positive valence primary value = (1 - 1/(1+PVPosGain * PVposSum))
	GvPVpos

	// PVnegSum is total weighted negative valence primary value = sum of Weight * USneg
	GvPVnegSum

	// PVpos is normalized negative valence primary value = (1 - 1/(1+PVNegGain * PVnegSum))
	GvPVneg

	// PVposEst is the estimated PVpos value based on OFCposUSPT and VSMatrix gating
	GvPVposEst

	// PVposEstSum is the sum that goes into computing estimated PVpos
	// value based on OFCposUSPT and VSMatrix gating
	GvPVposEstSum

	// PVposEstDisc is the discounted version of PVposEst, subtracting VSPatchPosSum,
	// which represents the accumulated expectation of PVpos to this point.
	GvPVposEstDisc

	// GiveUpDiff is the difference: PVposEstDisc - PVneg representing the
	// expected positive outcome up to this point.  When this turns negative,
	// the chance of giving up goes up proportionally, as a logistic
	// function of this difference.
	GvGiveUpDiff

	// GiveUpProb is the probability from the logistic function of GiveUpDiff
	GvGiveUpProb

	// GiveUp is true if a reset was triggered probabilistically based on GiveUpProb
	GvGiveUp

	// GaveUp is copy of GiveUp from previous trial
	GvGaveUp

	/////////////////////////////////////////
	// VSPatch prediction of PVpos net value

	// VSPatchPos is net shunting input from VSPatch (PosD1, named PVi in original PVLV)
	// computed as the Max of US-specific VSPatch saved values.
	// This is also stored as GvRewPred.
	GvVSPatchPos

	// VSPatchPosPrev is the previous-trial version of VSPatchPos -- for adjusting the
	// VSPatchThr threshold
	GvVSPatchPosPrev

	// VSPatchPosSum is the sum of VSPatchPos over goal engaged trials,
	// representing the integrated prediction that the US is going to occur
	GvVSPatchPosSum

	/////////////////////////////////////////
	// LHb lateral habenula component of the PVLV model -- does all US processing

	// computed LHb activity level that drives dipping / pausing of DA firing,
	// when VSPatch pos prediction > actual PV reward drive
	// or PVNeg > PVPos
	GvLHbDip

	// LHbBurst is computed LHb activity level that drives bursts of DA firing, when actual PV reward drive > VSPatch pos prediction
	GvLHbBurst

	// LHbPVDA is GvLHbBurst - GvLHbDip -- the LHb contribution to DA, reflecting PV and VSPatch (PVi), but not the CS (LV) contributions
	GvLHbPVDA

	/////////////////////////////////////////
	// Amygdala CS / LV variables

	// CeMpos is positive valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLAPosAcqD1 - BLAPosExtD2|_+ positively rectified.  CeM sets Raw directly.  Note that a positive US onset even with no active Drive will be reflected here, enabling learning about unexpected outcomes
	GvCeMpos

	// CeMneg is negative valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLANegAcqD2 - BLANegExtD1|_+ positively rectified.  CeM sets Raw directly
	GvCeMneg

	/////////////////////////////////////////
	// VTA ventral tegmental area dopamine release

	// VtaDA is overall dopamine value reflecting all of the different inputs
	GvVtaDA

	/////////////////////////////////////////
	// USneg is negative valence US
	//   allocated for Nitems

	// USneg are negative valence US outcomes -- normalized version of raw,
	// NNegUSs of them
	GvUSneg

	// USnegRaw are raw, linearly incremented negative valence US outcomes,
	// this value is also integrated together with all US vals for PVneg
	GvUSnegRaw

	///////////////////////////////////////////////////////////
	// USpos, VSPatch

	// Drives is current drive state -- updated with optional homeostatic exponential return to baseline values
	GvDrives

	// USpos is current positive-valence drive-satisfying input(s) (unconditioned stimuli = US)
	GvUSpos

	// VSPatch is current reward predicting VSPatch (PosD1) values
	GvVSPatch

	// VSPatch is previous reward predicting VSPatch (PosD1) values
	GvVSPatchPrev

	// OFCposUSPTMaint is activity level of given OFCposUSPT maintenance pool
	// used in anticipating potential USpos outcome value
	GvOFCposUSPTMaint

	// VSMatrixPoolGated indicates whether given VSMatrix pool gated
	// this is reset after last goal accomplished -- records gating since then.
	GvVSMatrixPoolGated

	GlobalVarsN
)

//gosl: end globals
