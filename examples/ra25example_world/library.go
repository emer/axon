package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/params"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"log"
	"time"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4 // TODO(refactor): Logs library

func ToggleLayersOff(net *axon.Network, layerNames []string, off bool) { // TODO(refactor): move to library
	for _, lnm := range layerNames {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func NewRndSeed(randomSeed *int64) { // TODO(refactor): to library
	*randomSeed = time.Now().UnixNano()
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func ApplyInputs(net *deep.Network, en WorldInterface, states, layers []string) { // TODO(refactor): library code
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	for i, lnm := range layers {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		pats := en.ObserveWithShape(states[i], lyi.Shape().Shp)
		lyi.Shape().Strides()
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func ApplyInputsWithStrideAndShape(net *axon.Network, en WorldInterface, states, layers []string) { // TODO(refactor): library code
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	for i, lnm := range layers {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		pats := en.ObserveWithShapeStride(states[i], lyi.Shape().Shp, lyi.Shape().Strides())
		lyi.Shape().Strides()
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func SaveWeights(fileName string, net *deep.Network) { // TODO(refactor): library code
	fnm := fileName
	fmt.Printf("Saving Weights to: %v\n", fnm)
	net.SaveWtsJSON(gi.FileName(fnm))
}

////////////////////////////////////////////////////////////////////
//  MPI code

// MPIInit initializes MPI
func MPIInit(useMPI *bool) *mpi.Comm { // TODO(refactor): library code
	mpi.Init()
	var err error

	comm, err := mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		*useMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
	return comm
}

// MPIFinalize finalizes MPI
func MPIFinalize(useMPI bool) { // TODO(refactor): library code
	if useMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
func CollectDWts(net *axon.Network, allWeightChanges *[]float32) { // TODO(refactor): axon library code
	net.CollectDWts(allWeightChanges)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func MPIWtFmDWt(comm *mpi.Comm, net *deep.Network, useMPI bool, allWeightChanges *[]float32, sumWeights *[]float32, time axon.Time) { // TODO(refactor): axon library code

	if useMPI {
		CollectDWts(&net.Network, allWeightChanges)
		ndw := len(*allWeightChanges)
		if len(*sumWeights) != ndw {
			*sumWeights = make([]float32, ndw)
		}
		comm.AllReduceF32(mpi.OpSum, *sumWeights, *allWeightChanges)
		net.SetDWts(*sumWeights, mpi.WorldSize())
	}
	net.WtFmDWt(&time)
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func ParamsName(paramset string) string { // TODO(refactor): library code
	if paramset == "" {
		return "Base"
	}
	return paramset
}

// TODO(refactor): Fix "Network" and "Sim" as arguments below

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParams(sheet string, setMsg bool, net *deep.Network, params *params.Sets, paramName string, ss interface{}) error { // TODO(refactor): Move to library, take in names as args
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := SetParamsSet("Base", sheet, setMsg, net, params, ss)
	if paramName != "" && paramName != "Base" {
		err = SetParamsSet(paramName, sheet, setMsg, net, params, ss)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParamsSet(setNm string, sheet string, setMsg bool, net *deep.Network, params *params.Sets, ss interface{}) error { // TODO(refactor): library, take in names as args
	pset, err := params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func RunName(tag string, paramName string) string { // TODO(refactor): library code
	if tag != "" {
		return tag + "_" + ParamsName(paramName)
	} else {
		return ParamsName(paramName)
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func RunEpochName(run, epc int) string { // TODO(refactor): library
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func WeightsFileName(netName, tag, paramName string, run, epc int) string { // TODO(refactor): library
	return netName + "_" + RunName(tag, paramName) + "_" + RunEpochName(run, epc) + ".wts.gz"
}

// LogFileName returns default log file name
func LogFileName(netName, lognm, tag, paramName string) string { // TODO(refactor): library
	return netName + "_" + RunName(tag, paramName) + "_" + lognm + ".tsv"
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func ValsTsr(tensorDictionary *map[string]*etensor.Float32, name string) *etensor.Float32 { // TODO(refactor): library code

	if *tensorDictionary == nil {
		*tensorDictionary = make(map[string]*etensor.Float32)
	}
	tsr, ok := (*tensorDictionary)[name]
	if !ok {
		tsr = &etensor.Float32{}
		(*tensorDictionary)[name] = tsr
	}
	return tsr
}

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func HogDead(net *deep.Network, lnm string) (hog, dead float64) { // TODO(refactor): library stats code
	ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	n := len(ly.Neurons)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActAvg > 0.3 {
			hog += 1
		} else if nrn.ActAvg < 0.01 {
			dead += 1
		}
	}
	hog /= float64(n)
	dead /= float64(n)
	return
}
