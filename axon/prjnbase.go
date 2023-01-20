// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"errors"
	"log"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
)

// PrjnBase contains the basic structural information for specifying a projection of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The exact same struct object is added to the Recv and Send layers, and it manages everything
// about the connectivity, and methods on the Prjn handle all the relevant computation.
type PrjnBase struct {
	AxonPrj         AxonPrjn        `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an AxonPrjn, which can always be used to extract the true underlying type of object when prjn is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Off             bool            `desc:"inactivate this projection -- allows for easy experimentation"`
	Cls             string          `desc:"Class is for applying parameter styles, can be space separated multple tags"`
	Notes           string          `desc:"can record notes about this projection here"`
	Send            emer.Layer      `desc:"sending layer for this projection"`
	Recv            emer.Layer      `desc:"receiving layer for this projection -- the emer.Layer interface can be converted to the specific Layer type you are using, e.g., rlay := prjn.Recv.(*axon.Layer)"`
	Pat             prjn.Pattern    `desc:"pattern of connectivity"`
	Typ             emer.PrjnType   `desc:"type of projection -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)"`
	RecvConN        []uint32        `view:"-" desc:"number of recv connections for each neuron in the receiving layer, as a flat list"`
	RecvConNAvgMax  minmax.AvgMax32 `inactive:"+" desc:"average and maximum number of recv connections in the receiving layer"`
	RecvConIdxStart []uint32        `view:"-" desc:"starting index into ConIdx list for each neuron in receiving layer -- just a list incremented by ConN"`
	RecvConIdx      []uint32        `view:"-" desc:"index of other neuron on sending side of projection, ordered by the receiving layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`
	RecvSynIdx      []uint32        `view:"-" desc:"index of synaptic state values for each recv unit x connection, for the receiver projection which does not own the synapses, and instead indexes into sender-ordered list"`
	SendConN        []uint32        `view:"-" desc:"number of sending connections for each neuron in the sending layer, as a flat list"`
	SendConNAvgMax  minmax.AvgMax32 `inactive:"+" desc:"average and maximum number of sending connections in the sending layer"`
	SendConIdxStart []uint32        `view:"-" desc:"starting index into ConIdx list for each neuron in sending layer -- just a list incremented by ConN"`
	SendConIdx      []uint32        `view:"-" desc:"index of other neuron on receiving side of projection, ordered by the sending layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`
	Syns            []Synapse       `desc:"synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SendConIdx array"`

	// misc state variables below:
	GBuf  []float32   `desc:"Ge or Gi conductance ring buffer for each neuron * Gidx.Len, accessed through Gidx, and length Gidx.Len in size per neuron -- scale * weight is added with Com delay offset."`
	PIBuf []float32   `desc:"pooled inhibition ring buffer for each pool * Gidx.Len, accessed through Gidx, and length Gidx.Len in size per pool in receiving layer."`
	PIdxs []uint32    `desc:"indexes of subpool for each receiving neuron, for aggregating PIBuf -- this is redundant with Neuron.Subpool but provides faster local access in SendSpike."`
	GVals []PrjnGVals `desc:"[recv neurons] projection-level synaptic conductance values, integrated by prjn before being integrated at the neuron level, which enables the neuron to perform non-linear integration as needed."`
}

// emer.Prjn interface

// Init MUST be called to initialize the prjn's pointer to itself as an emer.Prjn
// which enables the proper interface methods to be called.
func (pj *PrjnBase) Init(prjn emer.Prjn) {
	pj.AxonPrj = prjn.(AxonPrjn)
}

func (pj *PrjnBase) TypeName() string { return "Prjn" } // always, for params..
func (pj *PrjnBase) Class() string    { return pj.AxonPrj.PrjnTypeName() + " " + pj.Cls }
func (pj *PrjnBase) Name() string {
	return pj.Send.Name() + "To" + pj.Recv.Name()
}
func (pj *PrjnBase) SetClass(cls string) emer.Prjn         { pj.Cls = cls; return pj.AxonPrj }
func (pj *PrjnBase) SetPattern(pat prjn.Pattern) emer.Prjn { pj.Pat = pat; return pj.AxonPrj }
func (pj *PrjnBase) SetType(typ emer.PrjnType) emer.Prjn   { pj.Typ = typ; return pj.AxonPrj }
func (pj *PrjnBase) Label() string                         { return pj.Name() }
func (pj *PrjnBase) RecvLay() emer.Layer                   { return pj.Recv }
func (pj *PrjnBase) SendLay() emer.Layer                   { return pj.Send }
func (pj *PrjnBase) Pattern() prjn.Pattern                 { return pj.Pat }
func (pj *PrjnBase) Type() emer.PrjnType                   { return pj.Typ }
func (pj *PrjnBase) PrjnTypeName() string                  { return pj.Typ.String() }
func (pj *PrjnBase) IsOff() bool                           { return pj.Off }

// SetOff individual projection. Careful: Layer.SetOff(true) will reactivate all prjns of that layer,
// so prjn-level lesioning should always be done last.
func (pj *PrjnBase) SetOff(off bool) { pj.Off = off }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pj *PrjnBase) Connect(slay, rlay emer.Layer, pat prjn.Pattern, typ emer.PrjnType) {
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

// BuildBase constructs the full connectivity among the layers as specified in this projection.
// Calls Validate and returns false if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
func (pj *PrjnBase) BuildBase() error {
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
	tcons := pj.SetNIdxSt(&pj.SendConN, &pj.SendConNAvgMax, &pj.SendConIdxStart, sendn)
	tconr := pj.SetNIdxSt(&pj.RecvConN, &pj.RecvConNAvgMax, &pj.RecvConIdxStart, recvn)
	if tconr != tcons {
		log.Printf("%v programmer error: total recv cons %v != total send cons %v\n", pj.String(), tconr, tcons)
	}
	// these are large allocs, as number of connections tends to be quadratic
	pj.RecvConIdx = make([]uint32, tconr)
	pj.RecvSynIdx = make([]uint32, tconr)
	pj.SendConIdx = make([]uint32, tcons)

	sconN := make([]uint32, slen) // temporary mem needed to tracks cur n of sending cons

	cbits := cons.Values
	for ri := 0; ri < rlen; ri++ {
		rbi := ri * slen        // recv bit index
		rtcn := pj.RecvConN[ri] // number of cons
		rst := pj.RecvConIdxStart[ri]
		rci := uint32(0)
		for si := 0; si < slen; si++ {
			if !cbits.Index(rbi + si) { // no connection
				continue
			}
			sst := pj.SendConIdxStart[si]
			if rci >= rtcn {
				log.Printf("%v programmer error: recv target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), rtcn, ri, si)
				break
			}
			pj.RecvConIdx[rst+rci] = uint32(si)

			sci := sconN[si]
			stcn := pj.SendConN[si]
			if sci >= stcn {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), stcn, ri, si)
				break
			}
			pj.SendConIdx[sst+sci] = uint32(ri)
			pj.RecvSynIdx[rst+rci] = sst + sci
			(sconN[si])++
			rci++
		}
	}
	return nil
}

// SetNIdxSt sets the *ConN and *ConIdxSt values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pj *PrjnBase) SetNIdxSt(n *[]uint32, avgmax *minmax.AvgMax32, idxst *[]uint32, tn *etensor.Int32) uint32 {
	ln := tn.Len()
	tnv := tn.Values
	*n = make([]uint32, ln)
	*idxst = make([]uint32, ln)
	idx := uint32(0)
	avgmax.Init()
	for i := 0; i < ln; i++ {
		nv := uint32(tnv[i])
		(*n)[i] = nv
		(*idxst)[i] = idx
		idx += nv
		avgmax.UpdateVal(float32(nv), i)
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

// ApplyParams applies given parameter style Sheet to this projection.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (pj *PrjnBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(pj.AxonPrj, setMsg) // essential to go through AxonPrj
	if app {
		pj.AxonPrj.UpdateParams()
	}
	return app, err
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (pj *PrjnBase) NonDefaultParams() string {
	pth := pj.Recv.Name() + "." + pj.Name() // redundant but clearer..
	nds := giv.StructNonDefFieldsStr(pj.AxonPrj, pth)
	return nds
}

func (pj *PrjnBase) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (pj *PrjnBase) SynVarProps() map[string]string {
	return SynapseVarProps
}

// SynIdx returns the index of the synapse between given send, recv unit indexes
// (1D, flat indexes). Returns -1 if synapse not found between these two neurons.
// Requires searching within connections for receiving unit.
func (pj *PrjnBase) SynIdx(sidx, ridx int) int {
	if sidx >= len(pj.SendConN) {
		return -1
	}
	nc := int(pj.SendConN[sidx])
	st := int(pj.SendConIdxStart[sidx])
	for ci := 0; ci < nc; ci++ {
		ri := int(pj.SendConIdx[st+ci])
		if ri != ridx {
			continue
		}
		return int(st + ci)
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
	return len(SynapseVars)
}

// Syn1DNum returns the number of synapses for this prjn as a 1D array.
// This is the max idx for SynVal1D and the number of vals set by SynVals.
func (pj *PrjnBase) Syn1DNum() int {
	return len(pj.Syns)
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *PrjnBase) SynVal1D(varIdx int, synIdx int) float32 {
	if synIdx < 0 || synIdx >= len(pj.Syns) {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= pj.SynVarNum() {
		return mat32.NaN()
	}
	sy := &pj.Syns[synIdx]
	return sy.VarByIndex(varIdx)
}

// SynVals sets values of given variable name for each synapse, using the natural ordering
// of the synapses (sender based for Axon),
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pj *PrjnBase) SynVals(vals *[]float32, varNm string) error {
	vidx, err := pj.AxonPrj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	ns := len(pj.Syns)
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[0:ns]
	}
	for i := range pj.Syns {
		(*vals)[i] = pj.AxonPrj.SynVal1D(vidx, i)
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
	synIdx := pj.SynIdx(sidx, ridx)
	return pj.AxonPrj.SynVal1D(vidx, synIdx)
}

// Build constructs the full connectivity among the layers as specified in this projection.
// Calls PrjnBase.BuildBase and then allocates the synaptic values in Syns accordingly.
func (pj *PrjnBase) Build() error {
	if err := pj.BuildBase(); err != nil {
		return err
	}
	// this is a large alloc, as number of syns is typically large
	pj.Syns = make([]Synapse, len(pj.SendConIdx))
	rlay := pj.Recv.(AxonLayer).AsAxon()
	rlen := rlay.Shape().Len()
	pj.GVals = make([]PrjnGVals, rlen)
	pj.PIdxs = make([]uint32, rlen)
	for ni := range rlay.Neurons {
		pj.PIdxs[ni] = rlay.Neurons[ni].SubPool
	}
	return nil
}
