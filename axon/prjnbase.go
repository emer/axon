// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"log"

	"cogentcore.org/core/giv"
	"cogentcore.org/core/mat32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
)

// index naming:
// syi =  prjn-relative synapse index (per existing usage)
// syni = network-relative synapse index -- add SynStIdx to syi

// PrjnBase contains the basic structural information for specifying a projection of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The same struct token is added to the Recv and Send layer prjn lists, and it manages everything
// about the connectivity, and methods on the Prjn handle all the relevant computation.
// The Base does not have algorithm-specific methods and parameters, so it can be easily
// reused for different algorithms, and cleanly separates the algorithm-specific code.
// Any dependency on the algorithm-level Prjn can be captured in the AxonPrjn interface,
// accessed via the AxonPrj field.
type PrjnBase struct {

	// we need a pointer to ourselves as an AxonPrjn, which can always be used to extract the true underlying type of object when prjn is embedded in other structs -- function receivers do not have this ability so this is necessary.
	AxonPrj AxonPrjn `copy:"-" json:"-" xml:"-" view:"-"`

	// inactivate this projection -- allows for easy experimentation
	Off bool

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string

	// can record notes about this projection here
	Notes string

	// sending layer for this projection
	Send *Layer

	// receiving layer for this projection
	Recv *Layer

	// pattern of connectivity
	Pat prjn.Pattern `tableview:"-"`

	// type of projection -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)
	Typ PrjnTypes

	// default parameters that are applied prior to user-set parameters -- these are useful for specific functionality in specialized brain areas (e.g., PVLV, BG etc) not associated with a prjn type, which otherwise is used to hard-code initial default parameters -- typically just set to a literal map.
	DefParams params.Params `tableview:"-"`

	// provides a history of parameters applied to the layer
	ParamsHistory params.HistoryImpl `tableview:"-"`

	// average and maximum number of recv connections in the receiving layer
	RecvConNAvgMax minmax.AvgMax32 `tableview:"-" inactive:"+" view:"inline"`

	// average and maximum number of sending connections in the sending layer
	SendConNAvgMax minmax.AvgMax32 `tableview:"-" inactive:"+" view:"inline"`

	// start index into global Synapse array:
	SynStIdx uint32 `view:"-"`

	// number of synapses in this projection
	NSyns uint32 `view:"-"`

	// starting offset and N cons for each recv neuron, for indexing into the RecvSynIdx array of indexes into the Syns synapses, which are organized sender-based.  This is locally-managed during build process, but also copied to network global PrjnRecvCons slice for GPU usage.
	RecvCon []StartN `view:"-"`

	// index into Syns synaptic state for each sending unit and connection within that, for the sending projection which does not own the synapses, and instead indexes into recv-ordered list
	RecvSynIdx []uint32 `view:"-"`

	// for each recv synapse, this is index of *sending* neuron  It is generally preferable to use the Synapse SendIdx where needed, instead of this slice, because then the memory access will be close by other values on the synapse.
	RecvConIdx []uint32 `view:"-"`

	// starting offset and N cons for each sending neuron, for indexing into the Syns synapses, which are organized sender-based.  This is locally-managed during build process, but also copied to network global PrjnSendCons slice for GPU usage.
	SendCon []StartN `view:"-"`

	// index of other neuron that receives the sender's synaptic input, ordered by the sending layer's order of units as the outer loop, and SendCon.N receiving units within that.  It is generally preferable to use the Synapse RecvIdx where needed, instead of this slice, because then the memory access will be close by other values on the synapse.
	SendConIdx []uint32 `view:"-"`

	// Ge or Gi conductance ring buffer for each neuron, accessed through Params.Com.ReadIdx, WriteIdx -- scale * weight is added with Com delay offset -- a subslice from network PrjnGBuf. Uses int-encoded float values for faster GPU atomic integration
	GBuf []int32 `view:"-"`

	// projection-level synaptic conductance values, integrated by prjn before being integrated at the neuron level, which enables the neuron to perform non-linear integration as needed -- a subslice from network PrjnGSyn.
	GSyns []float32 `view:"-"`
}

// emer.Prjn interface

// Init MUST be called to initialize the prjn's pointer to itself as an emer.Prjn
// which enables the proper interface methods to be called.
func (pj *PrjnBase) Init(prjn emer.Prjn) {
	pj.AxonPrj = prjn.(AxonPrjn)
}

func (pj *PrjnBase) TypeName() string { return "Prjn" } // always, for params..
func (pj *PrjnBase) Class() string    { return pj.PrjnTypeName() + " " + pj.Cls }
func (pj *PrjnBase) Name() string {
	return pj.Send.Name() + "To" + pj.Recv.Name()
}
func (pj *PrjnBase) SetClass(cls string) emer.Prjn         { pj.Cls = cls; return pj.AxonPrj }
func (pj *PrjnBase) AddClass(cls string)                   { pj.Cls = params.AddClass(pj.Cls, cls) }
func (pj *PrjnBase) SetPattern(pat prjn.Pattern) emer.Prjn { pj.Pat = pat; return pj.AxonPrj }
func (pj *PrjnBase) SetType(typ emer.PrjnType) emer.Prjn   { pj.Typ = PrjnTypes(typ); return pj.AxonPrj }
func (pj *PrjnBase) Label() string                         { return pj.Name() }
func (pj *PrjnBase) RecvLay() emer.Layer                   { return pj.Recv }
func (pj *PrjnBase) SendLay() emer.Layer                   { return pj.Send }
func (pj *PrjnBase) Pattern() prjn.Pattern                 { return pj.Pat }
func (pj *PrjnBase) Type() emer.PrjnType                   { return emer.PrjnType(pj.Typ) }
func (pj *PrjnBase) PrjnTypeName() string                  { return pj.Typ.String() }
func (pj *PrjnBase) IsOff() bool                           { return pj.Off }

// SetOff individual projection. Careful: Layer.SetOff(true) will reactivate all prjns of that layer,
// so prjn-level lesioning should always be done last.
func (pj *PrjnBase) SetOff(off bool) { pj.Off = off }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pj *PrjnBase) Connect(slay, rlay *Layer, pat prjn.Pattern, typ PrjnTypes) {
	pj.Send = slay
	pj.Recv = rlay
	pj.Pat = pat
	pj.Typ = typ
}

// Validate tests for non-nil settings for the projection -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (pj *PrjnBase) Validate(logmsg bool) error {
	emsg := ""
	if pj.Pat == nil {
		emsg += "Pat is nil; "
	}
	if pj.Recv == nil {
		emsg += "Recv is nil; "
	}
	if pj.Send == nil {
		emsg += "Send is nil; "
	}
	if emsg != "" {
		err := errors.New(emsg)
		if logmsg {
			log.Println(emsg)
		}
		return err
	}
	return nil
}

// RecvSynIdxs returns the receiving synapse indexes for given recv unit index
// within the receiving layer, to be iterated over for recv-based processing.
func (pj *PrjnBase) RecvSynIdxs(ri uint32) []uint32 {
	rcon := pj.RecvCon[ri]
	return pj.RecvSynIdx[rcon.Start : rcon.Start+rcon.N]
}

// Build constructs the full connectivity among the layers.
// Calls Validate and returns error if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
// Does NOT allocate synapses -- these are set by Network from global slice.
func (pj *PrjnBase) Build() error {
	if pj.Off {
		return nil
	}
	err := pj.Validate(true)
	if err != nil {
		return err
	}
	ssh := pj.Send.Shape()
	rsh := pj.Recv.Shape()
	sendn, recvn, cons := pj.Pat.Connect(ssh, rsh, pj.Recv == pj.Send)
	slen := ssh.Len()
	rlen := rsh.Len()
	tcons := pj.SetConStartN(&pj.SendCon, &pj.SendConNAvgMax, sendn)
	tconr := pj.SetConStartN(&pj.RecvCon, &pj.RecvConNAvgMax, recvn)
	if tconr != tcons {
		log.Printf("%v programmer error: total recv cons %v != total send cons %v\n", pj.String(), tconr, tcons)
	}
	// these are large allocs, as number of connections tends to be ~quadratic
	// These indexes are not used in GPU computation -- only for CPU side.
	pj.RecvConIdx = make([]uint32, tconr)
	pj.RecvSynIdx = make([]uint32, tcons)
	pj.SendConIdx = make([]uint32, tcons)

	sconN := make([]uint32, slen) // temporary mem needed to tracks cur n of sending cons

	cbits := cons.Values
	for ri := 0; ri < rlen; ri++ {
		rbi := ri * slen // recv bit index
		rcon := pj.RecvCon[ri]
		rci := uint32(0)
		for si := 0; si < slen; si++ {
			if !cbits.Index(rbi + si) { // no connection
				continue
			}
			if rci >= rcon.N {
				log.Printf("%v programmer error: recv target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), rcon.N, ri, si)
				break
			}
			pj.RecvConIdx[rcon.Start+rci] = uint32(si)

			sci := sconN[si]
			scon := pj.SendCon[si]
			if sci >= scon.N {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), scon.N, ri, si)
				break
			}
			pj.SendConIdx[scon.Start+sci] = uint32(ri)
			pj.RecvSynIdx[rcon.Start+rci] = scon.Start + sci
			(sconN[si])++
			rci++
		}
	}
	return nil
}

// SetConStartN sets the *Con StartN values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pj *PrjnBase) SetConStartN(con *[]StartN, avgmax *minmax.AvgMax32, tn *etensor.Int32) uint32 {
	ln := tn.Len()
	tnv := tn.Values
	*con = make([]StartN, ln)
	idx := uint32(0)
	avgmax.Init()
	for i := 0; i < ln; i++ {
		nv := uint32(tnv[i])
		(*con)[i] = StartN{N: nv, Start: idx}
		idx += nv
		avgmax.UpdateVal(float32(nv), int32(i))
	}
	avgmax.CalcAvg()
	return idx
}

// String satisfies fmt.Stringer for prjn
func (pj *PrjnBase) String() string {
	str := ""
	if pj.Recv == nil {
		str += "recv=nil; "
	} else {
		str += pj.Recv.Name() + " <- "
	}
	if pj.Send == nil {
		str += "send=nil"
	} else {
		str += pj.Send.Name()
	}
	if pj.Pat == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + pj.Pat.Name()
	}
	return str
}

// ParamsHistoryReset resets parameter application history
func (pj *PrjnBase) ParamsHistoryReset() {
	pj.ParamsHistory.ParamsHistoryReset()
}

// ParamsApplied is just to satisfy History interface so reset can be applied
func (pj *PrjnBase) ParamsApplied(sel *params.Sel) {
	pj.ParamsHistory.ParamsApplied(sel)
}

// ApplyParams applies given parameter style Sheet to this projection.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (pj *PrjnBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(pj.AxonPrj, setMsg)
	// note: must use AxonPrj to get to actual Prjn, which then uses Styler interface
	// to return the Params struct.
	if app {
		pj.AxonPrj.UpdateParams()
	}
	return app, err
}

// ApplyDefParams applies DefParams default parameters if set
// Called by Prjn.Defaults()
func (pj *PrjnBase) ApplyDefParams() {
	if pj.DefParams == nil {
		return
	}
	err := pj.DefParams.Apply(pj.AxonPrj, false)
	if err != nil {
		log.Printf("programmer error -- fix DefParams: %s\n", err)
	}
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (pj *PrjnBase) NonDefaultParams() string {
	pth := pj.Recv.Name() + "." + pj.Name() // redundant but clearer..
	nds := giv.StructNonDefFieldsStr(pj.AxonPrj.AsAxon().Params, pth)
	return nds
}

func (pj *PrjnBase) SynVarNames() []string {
	return SynapseVarNames
}

// SynVarProps returns properties for variables
func (pj *PrjnBase) SynVarProps() map[string]string {
	return SynapseVarProps
}

// SynIdx returns the index of the synapse between given send, recv unit indexes
// (1D, flat indexes, layer relative).
// Returns -1 if synapse not found between these two neurons.
// Requires searching within connections for sending unit.
func (pj *PrjnBase) SynIdx(sidx, ridx int) int {
	if sidx >= len(pj.SendCon) {
		return -1
	}
	scon := pj.SendCon[sidx]
	if scon.N == 0 {
		return -1
	}
	firstRi := int(pj.SendConIdx[scon.Start])
	lastRi := int(pj.SendConIdx[scon.Start+scon.N-1])
	if ridx < firstRi || ridx > lastRi { // fast reject -- prjns are always in order!
		return -1
	}
	// start at index proportional to ri relative to rist
	up := int32(0)
	if lastRi > firstRi {
		up = int32(float32(scon.N) * float32(ridx-firstRi) / float32(lastRi-firstRi))
	}
	dn := up - 1

	for {
		doing := false
		if up < int32(scon.N) {
			doing = true
			sconi := int32(scon.Start) + up
			if int(pj.SendConIdx[sconi]) == ridx {
				return int(sconi)
			}
			up++
		}
		if dn >= 0 {
			doing = true
			sconi := int32(scon.Start) + dn
			if int(pj.SendConIdx[sconi]) == ridx {
				return int(sconi)
			}
			dn--
		}
		if !doing {
			break
		}
	}
	return -1
}

// SynVarIdx returns the index of given variable within the synapse,
// according to *this prjn's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *PrjnBase) SynVarIdx(varNm string) (int, error) {
	return SynapseVarByName(varNm)
}

// SynVarNum returns the number of synapse-level variables
// for this prjn.  This is needed for extending indexes in derived types.
func (pj *PrjnBase) SynVarNum() int {
	return len(SynapseVarNames)
}

// Syn1DNum returns the number of synapses for this prjn as a 1D array.
// This is the max idx for SynVal1D and the number of vals set by SynVals.
func (pj *PrjnBase) Syn1DNum() int {
	return int(pj.NSyns)
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods.
func (pj *PrjnBase) SynVal1D(varIdx int, synIdx int) float32 {
	if synIdx < 0 || synIdx >= int(pj.NSyns) {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= pj.SynVarNum() {
		return mat32.NaN()
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIdx + uint32(synIdx)
	if varIdx < int(SynapseVarsN) {
		return SynV(ctx, syni, SynapseVars(varIdx))
	} else {
		return SynCaV(ctx, syni, 0, SynapseCaVars(varIdx-int(SynapseVarsN))) // data = 0 def
	}
}

// SynVals sets values of given variable name for each synapse,
// using the natural ordering of the synapses (sender based for Axon),
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pj *PrjnBase) SynVals(vals *[]float32, varNm string) error {
	vidx, err := pj.AxonPrj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	ns := int(pj.NSyns)
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[0:ns]
	}
	slay := pj.Send
	i := 0
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pj.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			(*vals)[i] = pj.AxonPrj.SynVal1D(vidx, i)
			i++
		}
	}
	return nil
}

// SynVal returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns mat32.NaN() for access errors (see SynValTry for error message)
func (pj *PrjnBase) SynVal(varNm string, sidx, ridx int) float32 {
	vidx, err := pj.AxonPrj.SynVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	syi := pj.SynIdx(sidx, ridx)
	return pj.AxonPrj.SynVal1D(vidx, syi)
}

// SynVal1DDi returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods.
// Includes Di data parallel index for data-parallel synaptic values.
func (pj *PrjnBase) SynVal1DDi(varIdx int, synIdx int, di int) float32 {
	if synIdx < 0 || synIdx >= int(pj.NSyns) {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= pj.SynVarNum() {
		return mat32.NaN()
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIdx + uint32(synIdx)
	if varIdx < int(SynapseVarsN) {
		return SynV(ctx, syni, SynapseVars(varIdx))
	} else {
		return SynCaV(ctx, syni, uint32(di), SynapseCaVars(varIdx-int(SynapseVarsN)))
	}
}

// SynValDi returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns mat32.NaN() for access errors (see SynValTry for error message)
// Includes Di data parallel index for data-parallel synaptic values.
func (pj *PrjnBase) SynValDi(varNm string, sidx, ridx int, di int) float32 {
	vidx, err := pj.AxonPrj.SynVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	syi := pj.SynIdx(sidx, ridx)
	return pj.SynVal1DDi(vidx, syi, di)
}
