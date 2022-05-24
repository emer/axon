package axon

import (
	"fmt"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/params"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/gi"
	"time"
)

// AddDefaultLoopSimLogic adds some sim related logic to looper.Manager. It makes some assumptions about how the loop stack is set up which may cause it to fail.
func AddDefaultLoopSimLogic(manager *looper.Manager, time *Time, net *Network) {
	// Net Cycle
	for m, _ := range manager.Stacks {
		manager.Stacks[m].Loops[etime.Cycle].Main.Add("Axon:Cycle:RunAndIncrement", func() {
			net.Cycle(time)
			time.CycleInc()
		})
	}
	// Weight updates.
	// Note that the substring "UpdateNetView" in the name is important here, because it's checked in AddDefaultGUICallbacks.
	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Axon:LoopSegment:UpdateWeights", func() {
		net.DWt(time)
		// TODO Need to update net view here to accurately display weight changes.
		net.WtFmDWt(time)
	})

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t
			loop.OnStart.Add(curMode.String()+":"+curTime.String()+":"+"SetTimeVal", func() {
				time.Mode = curMode.String()
			})
		}
	}
}

// SendActionAndStep takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func SendActionAndStep(net *Network, ev agent.WorldInterface) {
	// Iterate over all Target (output) layers
	actions := map[string]agent.Action{}
	for _, lnm := range net.LayersByClass(emer.Target.String()) {
		ly := net.LayerByName(lnm).(AxonLayer).AsAxon()
		vt := &etensor.Float32{}      // TODO Maybe make this more efficient by holding a copy of the right size?
		ly.UnitValsTensor(vt, "ActM") // ActM is neuron activity
		actions[lnm] = agent.Action{Vector: vt, ActionShape: &agent.SpaceSpec{
			ContinuousShape: vt.Shp,
			Stride:          vt.Strd,
			Min:             0,
			Max:             1,
		}}
	}
	_, debug := ev.StepWorld(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}

// ToggleLayersOff can be used to disable layers in a Network, for example if you are doing an ablation study.
func ToggleLayersOff(net *Network, layerNames []string, off bool) {
	for _, lnm := range layerNames {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func ApplyInputs(net *Network, en agent.WorldInterface, layer string, patfunc func(spec agent.SpaceSpec) etensor.Tensor) {
	lyi := net.LayerByName(layer)
	lyi.(AxonLayer).InitExt() // Clear any existing inputs
	if lyi == nil {
		return
	}
	ly := lyi.(AxonLayer).AsAxon()
	ss := agent.SpaceSpec{ContinuousShape: lyi.Shape().Shp, Stride: lyi.Shape().Strd}
	pats := patfunc(ss)
	if pats != nil {
		ly.ApplyExt(pats)
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func SaveWeights(fileName string, net *Network) {
	fnm := fileName
	fmt.Printf("Saving Weights to: %v\n", fnm)
	net.SaveWtsJSON(gi.FileName(fnm))
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParams(sheet string, setMsg bool, net *Network, params *params.Sets, paramName string, ss interface{}) error {
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
func SetParamsSet(setNm string, sheet string, setMsg bool, net *Network, params *params.Sets, ss interface{}) error {
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

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr and under dead threshold
func HogDead(net *Network, lnm string) (hog, dead float64) {
	ly := net.LayerByName(lnm).(AxonLayer).AsAxon()
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

// AddPlusAndMinusPhases adds the minus and plus phases of the theta cycle, which help the network learn.
func AddPlusAndMinusPhases(manager *looper.Manager, time *Time, net *Network) {
	// The minus and plus phases of the theta cycle, which help the network learn.
	minusPhase := looper.Event{Name: "MinusPhase", AtCtr: 0}
	minusPhase.OnEvent.Add("Sim:MinusPhase:Start", func() {
		time.PlusPhase = false
		time.NewPhase(false)
	})
	plusPhase := looper.Event{Name: "PlusPhase", AtCtr: 150}
	plusPhase.OnEvent.Add("Sim:MinusPhase:End", func() { net.MinusPhase(time) })
	plusPhase.OnEvent.Add("Sim:PlusPhase:Start", func() {
		time.PlusPhase = true
		time.NewPhase(true)
	})
	plusPhaseEnd := looper.Event{Name: "PlusPhaseEnd", AtCtr: 199}
	plusPhaseEnd.OnEvent.Add("Sim:PlusPhase:End", func() { net.PlusPhase(time) })
	// Add both to train and test, by copy
	manager.AddEventAllModes(etime.Cycle, minusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhaseEnd)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run.
func NewRndSeed(randomSeed *int64) {
	*randomSeed = time.Now().UnixNano()
}

// AddCommonLogItemsForOutputLayers helps the UserInterface add common logging items based on the output layers of an axon Network.
func AddCommonLogItemsForOutputLayers(ui *egui.UserInterface) {
	// Record time in logs
	for m, st := range ui.Looper.Stacks {
		mode := m
		for t, l := range st.Loops {
			time := t
			loop := l
			ui.Logs.AddItem(&elog.Item{
				Name: t.String(),
				Type: etensor.INT64,
				Plot: elog.DFalse,
				Write: elog.WriteMap{
					etime.Scopes([]etime.Modes{mode}, []etime.Times{time}): func(ctx *elog.Context) {
						ctx.SetInt(loop.Counter.Cur)
					}}})
		}
	}

	// Error for output layers
	for _, olnm := range ui.Network.LayersByClass(emer.Target.String()) {
		out := ui.Network.LayerByName(olnm).(AxonLayer).AsAxon()

		// TODO These should be computed at the Trial, not Cycle level
		baseComputeLevel := etime.Trial
		found := false
		cosDiffMap := elog.WriteMap{}
		pctErrMap := elog.WriteMap{}
		trlCorrMap := elog.WriteMap{}
		for m, st := range ui.Looper.Stacks {
			for iter := len(st.Order) - 1; iter >= 0; iter-- {
				i := iter // For closures
				t := st.Order[i]
				if st.Order[iter] == baseComputeLevel {
					found = true // Subsequent layers can do aggregation.
					cosDiffMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat32(out.CosDiff.Cos)
					}
					pctErrMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat64(out.PctUnitErr())
					}
					trlCorrMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						pcterr := out.PctUnitErr()
						trlCorr := 1
						if pcterr > 0 {
							trlCorr = 0
						}
						ctx.SetFloat64(float64(trlCorr))
					}
				} else if found {
					// All other, less frequent, timescales are an aggregate
					for _, wm := range []elog.WriteMap{cosDiffMap, pctErrMap, trlCorrMap} {
						wm[etime.Scope(m, t)] = func(ctx *elog.Context) {
							ctx.SetAgg(ctx.Mode, st.Order[i+1], agg.AggMean)
						}
					}
				}
			}
		}

		// Add it to the list.
		ui.Logs.AddItem(&elog.Item{
			Name:   olnm + "CosSim",
			Type:   etensor.FLOAT64,
			Plot:   elog.DTrue,
			Range:  minmax.F64{Min: 0, Max: 1},
			FixMax: elog.DTrue,
			Write:  cosDiffMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  olnm + "PctErr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DTrue,
			Write: pctErrMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  olnm + "UnitCorr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DTrue,
			Write: trlCorrMap})
	}
}
