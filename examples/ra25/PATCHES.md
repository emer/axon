# 4/2022: Update to Looper

The new https://github.com/emer/emergent/tree/master/looper framework enables single-stepping at any level (including Cycles finally) and simplifies code management.  Here are the changes to incorporate it, in terms of an edited diff, skipping most of the brute force renaming of elog -> etime for all Times and Modes enum values.

Also, the critical `ConfigLoops` function is not shown here -- see ra25.go, and in general, the latest ra25.go is the official source -- these diffs are just helpful hints as to what other changes were needed.

```Go
 type Sim struct {
+	Loops        looper.Set       `desc:"contains looper control loops for running sim"`

@@ -131,6 +135,7 @@ func (ss *Sim) Config() {
 	ss.ConfigEnv()
 	ss.ConfigNet(ss.Net)
 	ss.ConfigLogs()
+	ss.ConfigLoops()
 }

 func (ss *Sim) ConfigEnv() {
@@ -142,16 +147,19 @@ func (ss *Sim) ConfigEnv() {
 		ss.NZeroStop = 5
 	}
 
	// Can be called multiple times -- don't re-create
	var trn, tst *envlp.FixedTable
	if len(ss.Envs) == 0 {
		trn = &envlp.FixedTable{}
		tst = &envlp.FixedTable{}
	} else {
		trn = ss.Envs["Train"].(*envlp.FixedTable)
		tst = ss.Envs["Test"].(*envlp.FixedTable)
	}
    
 	trn.Nm = "TrainEnv"
 	trn.Dsc = "training params and state"
-	trn.Table = etable.NewIdxView(ss.Pats)
+	trn.Config(etable.NewIdxView(ss.Pats), etime.Train.String())
+	trn.Counter(etime.Run).Max = ss.MaxRuns
+	trn.Counter(etime.Epoch).Max = ss.MaxEpcs
 	trn.Validate()
-	trn.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
 
 	tst.Nm = "TestEnv"
 	tst.Dsc = "testing params and state"
-	tst.Table = etable.NewIdxView(ss.Pats)
+	tst.Config(etable.NewIdxView(ss.Pats), etime.Test.String())
 	tst.Sequential = true
+	tst.Counter(etime.Epoch).Max = 1
 	tst.Validate()
 
 	// note: to create a train / test split of pats, do this:
@@ -160,8 +168,14 @@ func (ss *Sim) ConfigEnv() {
 	// trn.Table = splits.Splits[0]
 	// tst.Table = splits.Splits[1]
 
-	ss.TrainEnv.Init(0)
-	ss.TestEnv.Init(0)
+	trn.Init()
+	tst.Init()
+	ss.Envs.Add(trn, tst)
+}
+
+// Env returns the relevant environment based on Time Mode
+func (ss *Sim) Env() envlp.Env {
+	return ss.Envs[ss.Time.Mode]
 }
 
 func (ss *Sim) ConfigNet(net *axon.Network) {
@@ -229,7 +243,7 @@ func (ss *Sim) Init() {
 
 // InitRndSeed initializes the random seed based on current training run number
 func (ss *Sim) InitRndSeed() {
-	run := ss.TrainEnv.Run.Cur
+	run := ss.Envs["Train"].Counter(etime.Run).Cur
 	rand.Seed(ss.RndSeeds[run])
 }
 
@@ -242,260 +256,193 @@ func (ss *Sim) NewRndSeed() {
 	}
 }
 
-func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
-	switch viewUpdt {
-	case axon.Cycle:
-		ss.GUI.UpdateNetView()
-	case axon.FastSpike:
-		if ss.Time.Cycle%10 == 0 {
-			ss.GUI.UpdateNetView()
-		}
-	case axon.GammaCycle:
-		if ss.Time.Cycle%25 == 0 {
-			ss.GUI.UpdateNetView()
-		}
-	case axon.AlphaCycle:
-		if ss.Time.Cycle%100 == 0 {
-			ss.GUI.UpdateNetView()
-		}
+// UpdateNetViewCycle is updating within Cycle level
+func (ss *Sim) UpdateNetViewCycle() {
+	if !ss.ViewOn {
+		return
+	}
+	viewUpdt := ss.TrainUpdt
+	if ss.Time.Testing {
+		viewUpdt = ss.TestUpdt
 	}
+	ss.GUI.UpdateNetViewCycle(viewUpdt, ss.Time.Cycle)
 }
 
+// UpdateNetViewTime updates net view based on given time scale
+// in relation to view update settings.
+func (ss *Sim) UpdateNetViewTime(time etime.Times) {
+	if !ss.ViewOn {
+		return
+	}
 	viewUpdt := ss.TrainUpdt
-	if !train {
+	if ss.Time.Testing {
 		viewUpdt = ss.TestUpdt
 	}



-// ApplyInputs applies input patterns from given envirbonment.
+// ApplyInputs applies input patterns from given environment.
 // It is good practice to have this be a separate method with appropriate
 // args so that it can be used for various different contexts
 // (training, testing, etc).
-func (ss *Sim) ApplyInputs(en env.Env) {
+func (ss *Sim) ApplyInputs() {
+	ev := ss.Env()
 	// ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
 	// going to the same layers, but good practice and cheap anyway
 
 	lays := []string{"Input", "Output"}
 	for _, lnm := range lays {
 		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
-		pats := en.State(ly.Nm)
+		pats := ev.State(ly.Nm)
 		if pats != nil {
 			ly.ApplyExt(pats)
 		}
 	}
 }

 
 // NewRun intializes a new run of the model, using the TrainEnv.Run counter
 // for the new run value
 func (ss *Sim) NewRun() {
 	ss.InitRndSeed()
-	run := ss.TrainEnv.Run.Cur
-	ss.TrainEnv.Init(run)
-	ss.TestEnv.Init(run)
+	ss.Envs["Train"].Init()
+	ss.Envs["Test"].Init()
 	ss.Time.Reset()
 	ss.Net.InitWts()
 	ss.InitStats()
-	ss.StatCounters(true)
-	ss.Logs.ResetLog(elog.Train, elog.Epoch)
-	ss.Logs.ResetLog(elog.Test, elog.Epoch)
+	ss.StatCounters()
+	ss.Logs.ResetLog(etime.Train, etime.Epoch)
+	ss.Logs.ResetLog(etime.Test, etime.Epoch)
 	ss.NeedsNewRun = false
 }
 

-// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
-func (ss *Sim) RunTestAll() {
-	ss.GUI.StopNow = false
-	ss.TestAll()
-	ss.Stopped()
+	ss.Envs["Test"].Init()
+	tst := ss.Loops.Stack(etime.Test)
+	tst.Init()
+	tst.Run()
 }
 
 /////////////////////////////////////////////////////////////////////////
@@ -605,15 +506,12 @@ func (ss *Sim) InitStats() {
 
 // StatCounters saves current counters to Stats, so they are available for logging etc
 // Also saves a string rep of them to the GUI, if the GUI is active
-func (ss *Sim) StatCounters(train bool) {
-	ev := ss.TrainEnv
-	if !train {
-		ev = ss.TestEnv
-	}
-	ss.Stats.SetInt("Run", ss.TrainEnv.Run.Cur)
-	ss.Stats.SetInt("Epoch", ss.TrainEnv.Epoch.Cur)
-	ss.Stats.SetInt("Trial", ev.Trial.Cur)
-	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
+func (ss *Sim) StatCounters() {
+	ev := ss.Env()
+	if ev == nil {
+		return
+	}
+	ev.CtrsToStats(&ss.Stats)
 	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
 	ss.GUI.NetViewText = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCosDiff"})
 }

 
 // Log is the main logging function, handles special things for different scopes
-func (ss *Sim) Log(mode elog.EvalModes, time elog.Times) {
+func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
+	ss.Time.Mode = mode.String()
+	ss.StatCounters()
 	dt := ss.Logs.Table(mode, time)
 	row := dt.Rows
 	switch {
-	case mode == elog.Test && time == elog.Epoch:
+	case mode == etime.Test && time == etime.Epoch:
+		ss.Stats.SetInt("Epoch", ss.Envs["Train"].Counter(etime.Epoch).Cur)
 		ss.LogTestErrors()
-	case mode == elog.Train && time == elog.Epoch:
-		epc := ss.TrainEnv.Epoch.Cur
-		if (ss.PCAInterval > 0) && ((epc-1)%ss.PCAInterval == 0) { // -1 so runs on first epc
+	case mode == etime.Train && time == etime.Epoch:
+		epc := ss.Envs["Train"].Counter(etime.Epoch).Cur
+		if ss.PCAInterval > 0 && epc%ss.PCAInterval == 0 {
 			ss.PCAStats()
 		}

-	case mode == elog.Train && time == elog.Trial:
-		epc := ss.TrainEnv.Epoch.Cur
+	case mode == etime.Train && time == etime.Trial:
+		epc := ss.Envs["Train"].Counter(etime.Epoch).Cur
 		if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
-			ss.Log(elog.Analyze, elog.Trial)
+			ss.Log(etime.Analyze, etime.Trial)
 		}
 	}
 }
 
-// RasterRec updates spike raster record for given cycle
-func (ss *Sim) RasterRec(cyc int) {
-	ss.Stats.RasterRec(ss.Net, cyc, "Spike")
+// RasterRec updates spike raster record for current Time.Cycle
+func (ss *Sim) RasterRec() {
+	ss.Stats.RasterRec(ss.Net, ss.Time.Cycle, "Spike")
 }
 
 // RunName returns a name for this run that combines Tag and Params -- add this to
@@ -747,7 +648,7 @@ func (ss *Sim) RunEpochName(run, epc int) string {
 
 // WeightsFileName returns default current weights file name
 func (ss *Sim) WeightsFileName() string {
-	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
+	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.Envs["Train"].Counter(etime.Run).Cur, ss.Envs["Train"].Counter(etime.Epoch).Cur) + ".wts"
 }
 
 // LogFileName returns default log file name
@@ -763,8 +664,10 @@ func (ss *Sim) ConfigGui() *gi.Window {
 	title := "Leabra Random Associator"
 	ss.GUI.MakeWindow(ss, "ra25", title, `This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
 	ss.GUI.CycleUpdateInterval = 10
-	ss.GUI.NetView.Params.MaxRecs = 300
-	ss.GUI.NetView.SetNet(ss.Net)
+
+	nv := ss.GUI.AddNetView("NetView")
+	nv.Params.MaxRecs = 300
+	nv.SetNet(ss.Net)
 
 	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
 	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
@@ -786,119 +689,18 @@ func (ss *Sim) ConfigGui() *gi.Window {
 			ss.GUI.UpdateWindow()
 		},
 	})
-	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
-		Icon:    "run",
-		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
-		Active:  egui.ActiveStopped,
-		Func: func() {
-			if !ss.GUI.IsRunning {
-				ss.GUI.IsRunning = true
-				ss.GUI.ToolBar.UpdateActions()
-				go ss.Train()
-			}
-		},
-	})
+
 	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
 		Icon:    "stop",
-		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
+		Tooltip: "Interrupts running.  running / stepping picks back up where it left off.",
 		Active:  egui.ActiveRunning,
 		Func: func() {
 			ss.Stop()
 		},
 	})
-	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
...
-	})
+	ss.GUI.AddLooperCtrl(ss.Loops.Stack(etime.Train))
+	ss.GUI.AddLooperCtrl(ss.Loops.Stack(etime.Test))
 
@@ -992,10 +794,11 @@ func (ss *Sim) CmdArgs() {
 		fmt.Printf("Saving final weights per run\n")
 	}
 	fmt.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
-	ss.TrainEnv.Run.Set(ss.StartRun)
-	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
+	rc := ss.Envs["Train"].Counter(etime.Run)
+	rc.Set(ss.StartRun)
+	rc.Max = ss.StartRun + ss.MaxRuns
 	ss.NewRun()
-	ss.Train()
+	ss.Loops.Run(etime.Train)
 
 	ss.Logs.CloseLogFiles()
```
 
