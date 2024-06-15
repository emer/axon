// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"log"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

// index naming:
// syi =  path-relative synapse index (per existing usage)
// syni = network-relative synapse index -- add SynStIndex to syi

// PathBase contains the basic structural information for specifying a pathway of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The same struct token is added to the Recv and Send layer path lists, and it manages everything
// about the connectivity, and methods on the Path handle all the relevant computation.
// The Base does not have algorithm-specific methods and parameters, so it can be easily
// reused for different algorithms, and cleanly separates the algorithm-specific code.
// Any dependency on the algorithm-level Path can be captured in the AxonPath interface,
// accessed via the AxonPrj field.
type PathBase struct {

	// we need a pointer to ourselves as an AxonPath, which can always be used to extract the true underlying type of object when path is embedded in other structs -- function receivers do not have this ability so this is necessary.
	AxonPrj AxonPath `copier:"-" json:"-" xml:"-" display:"-"`

	// inactivate this pathway -- allows for easy experimentation
	Off bool

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string

	// can record notes about this pathway here
	Notes string

	// sending layer for this pathway
	Send *Layer

	// receiving layer for this pathway
	Recv *Layer

	// pattern of connectivity
	Pat paths.Pattern `tabledisplay:"-"`

	// type of pathway:  Forward, Back, Lateral, or extended type in specialized algorithms.
	// Matches against .Cls parameter styles (e.g., .Back etc)
	Typ PathTypes

	// default parameters that are applied prior to user-set parameters.
	// these are useful for specific functionality in specialized brain areas
	// (e.g., Rubicon, BG etc) not associated with a path type, which otherwise
	// is used to hard-code initial default parameters.
	// Typically just set to a literal map.
	DefParams params.Params `tabledisplay:"-"`

	// provides a history of parameters applied to the layer
	ParamsHistory params.HistoryImpl `tabledisplay:"-"`

	// average and maximum number of recv connections in the receiving layer
	RecvConNAvgMax minmax.AvgMax32 `tabledisplay:"-" edit:"-" display:"inline"`

	// average and maximum number of sending connections in the sending layer
	SendConNAvgMax minmax.AvgMax32 `tabledisplay:"-" edit:"-" display:"inline"`

	// start index into global Synapse array:
	SynStIndex uint32 `display:"-"`

	// number of synapses in this pathway
	NSyns uint32 `display:"-"`

	// starting offset and N cons for each recv neuron, for indexing into the RecvSynIndex array of indexes into the Syns synapses, which are organized sender-based.  This is locally managed during build process, but also copied to network global PathRecvCons slice for GPU usage.
	RecvCon []StartN `display:"-"`

	// index into Syns synaptic state for each sending unit and connection within that, for the sending pathway which does not own the synapses, and instead indexes into recv-ordered list
	RecvSynIndex []uint32 `display:"-"`

	// for each recv synapse, this is index of *sending* neuron  It is generally preferable to use the Synapse SendIndex where needed, instead of this slice, because then the memory access will be close by other values on the synapse.
	RecvConIndex []uint32 `display:"-"`

	// starting offset and N cons for each sending neuron, for indexing into the Syns synapses, which are organized sender-based.  This is locally managed during build process, but also copied to network global PathSendCons slice for GPU usage.
	SendCon []StartN `display:"-"`

	// index of other neuron that receives the sender's synaptic input, ordered by the sending layer's order of units as the outer loop, and SendCon.N receiving units within that.  It is generally preferable to use the Synapse RecvIndex where needed, instead of this slice, because then the memory access will be close by other values on the synapse.
	SendConIndex []uint32 `display:"-"`

	// Ge or Gi conductance ring buffer for each neuron, accessed through Params.Com.ReadIndex, WriteIndex -- scale * weight is added with Com delay offset -- a subslice from network PathGBuf. Uses int-encoded float values for faster GPU atomic integration
	GBuf []int32 `display:"-"`

	// pathway-level synaptic conductance values, integrated by path before being integrated at the neuron level, which enables the neuron to perform non-linear integration as needed -- a subslice from network PathGSyn.
	GSyns []float32 `display:"-"`
}

// emer.Path interface

// Init MUST be called to initialize the path's pointer to itself as an emer.Path
// which enables the proper interface methods to be called.
func (pj *PathBase) Init(path emer.Path) {
	pj.AxonPrj = path.(AxonPath)
}

func (pj *PathBase) TypeName() string { return "Path" } // always, for params..
func (pj *PathBase) Class() string    { return pj.PathTypeName() + " " + pj.Cls }
func (pj *PathBase) Name() string {
	return pj.Send.Name() + "To" + pj.Recv.Name()
}
func (pj *Path) AddClass(cls ...string) emer.Path {
	pj.Cls = params.AddClass(pj.Cls, cls...)
	return pj
}
func (pj *PathBase) SetPattern(pat paths.Pattern) emer.Path { pj.Pat = pat; return pj.AxonPrj }
func (pj *PathBase) SetType(typ emer.PathType) emer.Path    { pj.Typ = PathTypes(typ); return pj.AxonPrj }
func (pj *PathBase) Label() string                          { return pj.Name() }
func (pj *PathBase) RecvLay() emer.Layer                    { return pj.Recv }
func (pj *PathBase) SendLay() emer.Layer                    { return pj.Send }
func (pj *PathBase) Pattern() paths.Pattern                 { return pj.Pat }
func (pj *PathBase) Type() emer.PathType                    { return emer.PathType(pj.Typ) }
func (pj *PathBase) PathTypeName() string                   { return pj.Typ.String() }
func (pj *PathBase) IsOff() bool                            { return pj.Off }

// SetOff individual pathway. Careful: Layer.SetOff(true) will reactivate all paths of that layer,
// so path-level lesioning should always be done last.
func (pj *PathBase) SetOff(off bool) { pj.Off = off }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pj *PathBase) Connect(slay, rlay *Layer, pat paths.Pattern, typ PathTypes) {
	pj.Send = slay
	pj.Recv = rlay
	pj.Pat = pat
	pj.Typ = typ
}

// Validate tests for non-nil settings for the pathway -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (pj *PathBase) Validate(logmsg bool) error {
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

// RecvSynIndexes returns the receiving synapse indexes for given recv unit index
// within the receiving layer, to be iterated over for recv-based processing.
func (pj *PathBase) RecvSynIndexes(ri uint32) []uint32 {
	rcon := pj.RecvCon[ri]
	return pj.RecvSynIndex[rcon.Start : rcon.Start+rcon.N]
}

// Build constructs the full connectivity among the layers.
// Calls Validate and returns error if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
// Does NOT allocate synapses -- these are set by Network from global slice.
func (pj *PathBase) Build() error {
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
	pj.RecvConIndex = make([]uint32, tconr)
	pj.RecvSynIndex = make([]uint32, tcons)
	pj.SendConIndex = make([]uint32, tcons)

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
			pj.RecvConIndex[rcon.Start+rci] = uint32(si)

			sci := sconN[si]
			scon := pj.SendCon[si]
			if sci >= scon.N {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), scon.N, ri, si)
				break
			}
			pj.SendConIndex[scon.Start+sci] = uint32(ri)
			pj.RecvSynIndex[rcon.Start+rci] = scon.Start + sci
			(sconN[si])++
			rci++
		}
	}
	return nil
}

// SetConStartN sets the *Con StartN values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pj *PathBase) SetConStartN(con *[]StartN, avgmax *minmax.AvgMax32, tn *tensor.Int32) uint32 {
	ln := tn.Len()
	tnv := tn.Values
	*con = make([]StartN, ln)
	idx := uint32(0)
	avgmax.Init()
	for i := 0; i < ln; i++ {
		nv := uint32(tnv[i])
		(*con)[i] = StartN{N: nv, Start: idx}
		idx += nv
		avgmax.UpdateValue(float32(nv), int32(i))
	}
	avgmax.CalcAvg()
	return idx
}

// String satisfies fmt.Stringer for path
func (pj *PathBase) String() string {
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
func (pj *PathBase) ParamsHistoryReset() {
	pj.ParamsHistory.ParamsHistoryReset()
}

// ParamsApplied is just to satisfy History interface so reset can be applied
func (pj *PathBase) ParamsApplied(sel *params.Sel) {
	pj.ParamsHistory.ParamsApplied(sel)
}

// ApplyParams applies given parameter style Sheet to this pathway.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (pj *PathBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(pj.AxonPrj, setMsg)
	// note: must use AxonPrj to get to actual Path, which then uses Styler interface
	// to return the Params struct.
	if app {
		pj.AxonPrj.UpdateParams()
	}
	return app, err
}

// ApplyDefParams applies DefParams default parameters if set
// Called by Path.Defaults()
func (pj *PathBase) ApplyDefParams() {
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
func (pj *PathBase) NonDefaultParams() string {
	pth := pj.Recv.Name() + "." + pj.Name() // redundant but clearer..
	_ = pth
	// nds := views.StructNonDefFieldsStr(pj.AxonPrj.AsAxon().Params, pth)
	// todo: see layerbase for new impl
	return "todo need to do"
}

func (pj *PathBase) SynVarNames() []string {
	return SynapseVarNames
}

// SynVarProps returns properties for variables
func (pj *PathBase) SynVarProps() map[string]string {
	return SynapseVarProps
}

// SynIndex returns the index of the synapse between given send, recv unit indexes
// (1D, flat indexes, layer relative).
// Returns -1 if synapse not found between these two neurons.
// Requires searching within connections for sending unit.
func (pj *PathBase) SynIndex(sidx, ridx int) int {
	if sidx >= len(pj.SendCon) {
		return -1
	}
	scon := pj.SendCon[sidx]
	if scon.N == 0 {
		return -1
	}
	firstRi := int(pj.SendConIndex[scon.Start])
	lastRi := int(pj.SendConIndex[scon.Start+scon.N-1])
	if ridx < firstRi || ridx > lastRi { // fast reject -- paths are always in order!
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
			if int(pj.SendConIndex[sconi]) == ridx {
				return int(sconi)
			}
			up++
		}
		if dn >= 0 {
			doing = true
			sconi := int32(scon.Start) + dn
			if int(pj.SendConIndex[sconi]) == ridx {
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

// SynVarIndex returns the index of given variable within the synapse,
// according to *this path's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *PathBase) SynVarIndex(varNm string) (int, error) {
	return SynapseVarByName(varNm)
}

// SynVarNum returns the number of synapse-level variables
// for this paths.  This is needed for extending indexes in derived types.
func (pj *PathBase) SynVarNum() int {
	return len(SynapseVarNames)
}

// Syn1DNum returns the number of synapses for this path as a 1D array.
// This is the max idx for SynVal1D and the number of vals set by SynValues.
func (pj *PathBase) Syn1DNum() int {
	return int(pj.NSyns)
}

// SynVal1D returns value of given variable index (from SynVarIndex) on given SynIndex.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods.
func (pj *PathBase) SynVal1D(varIndex int, synIndex int) float32 {
	if synIndex < 0 || synIndex >= int(pj.NSyns) {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= pj.SynVarNum() {
		return math32.NaN()
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIndex + uint32(synIndex)
	if varIndex < int(SynapseVarsN) {
		return SynV(ctx, syni, SynapseVars(varIndex))
	} else {
		return SynCaV(ctx, syni, 0, SynapseCaVars(varIndex-int(SynapseVarsN))) // data = 0 def
	}
}

// SynValues sets values of given variable name for each synapse,
// using the natural ordering of the synapses (sender based for Axon),
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pj *PathBase) SynValues(vals *[]float32, varNm string) error {
	vidx, err := pj.AxonPrj.SynVarIndex(varNm)
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
// Returns math32.NaN() for access errors (see SynValTry for error message)
func (pj *PathBase) SynValue(varNm string, sidx, ridx int) float32 {
	vidx, err := pj.AxonPrj.SynVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	syi := pj.SynIndex(sidx, ridx)
	return pj.AxonPrj.SynVal1D(vidx, syi)
}

// SynVal1DDi returns value of given variable index (from SynVarIndex) on given SynIndex.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods.
// Includes Di data parallel index for data-parallel synaptic values.
func (pj *PathBase) SynVal1DDi(varIndex int, synIndex int, di int) float32 {
	if synIndex < 0 || synIndex >= int(pj.NSyns) {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= pj.SynVarNum() {
		return math32.NaN()
	}
	ctx := &pj.Recv.Network.Ctx
	syni := pj.SynStIndex + uint32(synIndex)
	if varIndex < int(SynapseVarsN) {
		return SynV(ctx, syni, SynapseVars(varIndex))
	} else {
		return SynCaV(ctx, syni, uint32(di), SynapseCaVars(varIndex-int(SynapseVarsN)))
	}
}

// SynValDi returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns math32.NaN() for access errors (see SynValTry for error message)
// Includes Di data parallel index for data-parallel synaptic values.
func (pj *PathBase) SynValDi(varNm string, sidx, ridx int, di int) float32 {
	vidx, err := pj.AxonPrj.SynVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	syi := pj.SynIndex(sidx, ridx)
	return pj.SynVal1DDi(vidx, syi, di)
}
