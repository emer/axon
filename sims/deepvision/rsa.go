// Copyright (c) 2020, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

import (
	"os"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/core"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensor/tmath"
	"cogentcore.org/lab/tensorcore"
	"cogentcore.org/lab/tensorfs"
)

var (
	Debug = false

	// 20 Object categs: IMPORTANT: do not change the order of this list as it is used
	// in various places as the cannonical ordering for e.g., Expt1 data
	Objs = []string{
		"banana",
		"layercake",
		"trafficcone",
		"sailboat",
		"trex",
		"person",
		"guitar",
		"tablelamp",
		"doorknob",
		"handgun",
		"donut",
		"chair",
		"slrcamera",
		"elephant",
		"piano",
		"fish",
		"car",
		"heavycannon",
		"stapler",
		"motorcycle",
	}

	ObjIdxs map[string]int

	// CanonicalCats is best-fitting 5-category leabra ("Centroid")
	CanonicalCats = map[string]string{
		"banana":      "1-pyramid",
		"layercake":   "1-pyramid",
		"trafficcone": "1-pyramid",
		"sailboat":    "1-pyramid",
		"trex":        "1-pyramid",
		"person":      "2-vertical",
		"guitar":      "2-vertical",
		"tablelamp":   "2-vertical",
		"doorknob":    "3-round",
		"donut":       "3-round",
		"handgun":     "3-round",
		"chair":       "3-round",
		"slrcamera":   "4-box",
		"elephant":    "4-box",
		"piano":       "4-box",
		"fish":        "4-box",
		"car":         "5-horiz",
		"heavycannon": "5-horiz",
		"stapler":     "5-horiz",
		"motorcycle":  "5-horiz",
	}

	CanonicalGroups []string // CanonicalCats with repeats all blank, for grouped labels

	rsaStatNames = []string{"RSAvsV1", "RSAvsTE", "RSAvsExpt"} // , "MeanCentroid", "MeanBasic", "MeanExpt"}
)

// StatRSA returns a Stats function that records RSA:
// representational similarity analysis stats.
func (ss *Sim) StatRSA(layers ...string) func(mode Modes, level Levels, phase StatsPhase) {
	net := ss.Net
	return func(mode Modes, level Levels, phase StatsPhase) {
		if level < Trial {
			return
		}
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		interval := ss.Config.Run.RSAInterval
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String()).Dir("RSA")
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		ndata := int(net.Context().NData)
		for li, lnm := range layers {
			if level == Trial {
				for di := range ndata {
					ev := ss.Envs.ByModeDi(mode, di).(*Obj3DSacEnv)
					tick := ev.Tick.Cur
					if tick == 2 { // using tick 2 for all data
						ss.rsaTrial(curModeDir, lnm, ev.CurCat, di)
					}
				}
				continue // no actual stats at trial level
			}
			for si, stnm := range rsaStatNames {
				name := lnm + "_" + stnm
				tsr := levelDir.Float64(name)
				if phase == Start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						s.On = true
					})
					continue
				}
				switch level {
				case Epoch:
					hasNew := false
					if interval > 0 && trnEpc%interval == 0 {
						hasNew = true
						if li == 0 && si == 0 {
							ss.rsaEpoch(curModeDir, layers...) // puts results in curModeDir
						}
					}
					var stat float64
					nr := tsr.DimSize(0)
					if nr > 0 {
						stat = tsr.FloatRow(nr-1, 0)
					}
					if hasNew {
						stat = curModeDir.Float64(name).Float1D(0)
					}
					tsr.AppendRowFloat(float64(stat))
				case Run:
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

func (ss *Sim) RSAInit() {
	curModeDir := ss.Current.Dir(Train.String()).Dir("RSA")
	nc := len(Objs)

	smat := curModeDir.Float64("Expt_Smat", nc, nc)
	errors.Log(tensor.OpenCSV(smat, fsx.Filename("expt1_simat.csv"), tensor.Comma))
	mx := stats.Max(tensor.As1D(smat))
	tmath.DivOut(smat, mx, smat)

	ObjIdxs = make(map[string]int, nc)
	CanonicalGroups = make([]string, nc)
	lstcat := ""
	for i, o := range Objs {
		ObjIdxs[o] = i
		cat := CanonicalCats[o]
		if cat != lstcat {
			CanonicalGroups[i] = cat
			lstcat = cat
		}
	}
}

var SimMatGridStyle = func(s *tensorcore.GridStyle) {
	s.TopZero = true
	s.Range.SetMin(0).SetMax(1)
	s.ColorMap = core.ColorMapName("Viridis")
	s.GridFill = 1
	s.DimExtra = 0.15
}

func (ss *Sim) RSAGUI() {
	ss.rsaSimMatGrid("Expt_Smat")
}

func (ss *Sim) rsaSimMatGrid(nm string) {
	curModeDir := ss.Current.Dir(Train.String()).Dir("RSA")
	tbs := ss.GUI.Tabs.AsLab()
	_, idx := tbs.CurrentTab()

	smat := curModeDir.Float64(nm)
	tensorcore.AddGridStylerTo(smat, SimMatGridStyle)
	tg := tbs.TensorGrid(strings.TrimSuffix(nm, "_Smat"), smat)
	tg.RowLabels = CanonicalGroups
	tg.ColumnLabels = CanonicalGroups
	tbs.SelectTabIndex(idx)
}

// RSASaveRActs saves running average activation data to tar file.
func (ss *Sim) RSASaveRActs(fname string) error {
	f, err := os.Create(fname)
	if errors.Log(err) != nil {
		return err
	}
	defer f.Close()
	curModeDir := ss.Current.Dir(Train.String()).Dir("RSA")
	err = tensorfs.Tar(f, curModeDir.Dir("RAvgs"), true, nil) // gz
	return errors.Log(err)
}

// RSAOpenRActs opens running average activation data from tar file.
func (ss *Sim) RSAOpenRActs(fname fsx.Filename) error { //types:add
	f, err := os.Open(string(fname))
	if errors.Log(err) != nil {
		return err
	}
	defer f.Close()
	curModeDir := ss.Current.Dir(Train.String()).Dir("RSA")
	err = errors.Log(tensorfs.Untar(f, curModeDir.Dir("RAvgs"), true))
	if err == nil {
		ss.RSAStats()
	}
	return err
}

// rsaTrial accumulates running-average activations for layer, object in _Ravg
func (ss *Sim) rsaTrial(curModeDir *tensorfs.Node, lnm, obj string, di int) {
	avgDt := 0.1
	avgDtC := 1 - avgDt
	ly := ss.Net.LayerByName(lnm)
	atsr := curModeDir.Dir("RAvgs").Dir(lnm).Float64(obj, ly.Shape.Sizes...)

	varName := "Act"
	vtsr := curModeDir.Float32(lnm+"_"+varName, ly.Shape.Sizes...)
	ly.UnitValuesTensor(vtsr, varName, di)

	nn := int(ly.Shape.Len())
	for lni := range nn {
		act := vtsr.Float1D(lni)
		avg := atsr.Float1D(lni)
		avg = avgDtC*avg + avgDt*act
		atsr.SetFloat1D(avg, lni)
	}
}

// RSAStats runs stats on current data, displaying grids.
// This is run on saved data, in GUI.
func (ss *Sim) RSAStats() {
	curModeDir := ss.Current.Dir(Train.String()).Dir("RSA")
	ravgs := curModeDir.Dir("RAvgs")
	var slays []string
	ravgs.NodesFunc(func(nd *tensorfs.Node) bool {
		slays = append(slays, nd.Name())
		return false
	})
	// fmt.Println("slays:", slays)
	ss.rsaEpoch(curModeDir, slays...)
	for _, lnm := range slays {
		ss.rsaSimMatGrid(lnm + "_Smat")
	}
}

// rsaEpoch computes all stats at epoch level
func (ss *Sim) rsaEpoch(curModeDir *tensorfs.Node, layers ...string) {
	// first get everything per-layer
	rsaSimMats(curModeDir, layers...)

	nc := len(Objs)
	v1Smat := curModeDir.Float64("V1m_Smat", nc, nc)
	teSmat := curModeDir.Float64("TE_Smat")
	exSmat := curModeDir.Float64("Expt_Smat")

	for _, lnm := range layers {
		smat := curModeDir.Float64(lnm+"_Smat", nc, nc)
		v1snm := lnm + "_" + rsaStatNames[0]
		v1sim := 1.0
		if lnm != "V1m" {
			v1sim = metric.Correlation(v1Smat, smat).Float1D(0)
		}
		curModeDir.Float64(v1snm, 1).SetFloat1D(v1sim, 0)

		tesnm := lnm + "_" + rsaStatNames[1]
		tesim := 1.0
		if teSmat.Len() == nc*nc && lnm != "TE" {
			tesim = metric.Correlation(teSmat, smat).Float1D(0)
		}
		curModeDir.Float64(tesnm, 1).SetFloat1D(tesim, 0)

		exsnm := lnm + "_" + rsaStatNames[2]
		exsim := metric.Correlation(exSmat, smat).Float1D(0)
		curModeDir.Float64(exsnm, 1).SetFloat1D(exsim, 0)
	}
}

// rsaSimMats computes the similarity matrixes from running average acts
func rsaSimMats(curModeDir *tensorfs.Node, layers ...string) {
	nc := len(Objs)
	for _, lnm := range layers {
		// Canonical simat
		smat := curModeDir.Float64(lnm+"_Smat", nc, nc)
		adir := curModeDir.Dir("RAvgs").Dir(lnm)
		for ci, obj := range Objs {
			atsr := tensor.As1D(adir.Float64(obj))
			if atsr.Len() == 0 {
				continue
			}
			for oci := ci + 1; oci < nc; oci++ {
				oobj := Objs[oci]
				otsr := tensor.As1D(adir.Float64(oobj))
				sim := 0.0
				if otsr.Len() > 0 {
					sim = metric.InvCorrelation(atsr, otsr).Float1D(0)
				}
				smat.SetFloat(sim, ci, oci)
				smat.SetFloat(sim, oci, ci)
			}
		}
	}
}

// // RSA handles representational similarity analysis
// type RSA struct {
// 	Cats       []string                 `desc:"category names for each row of simmat / activation table -- call SetCats"`
// 	Sims       map[string]*simat.SimMat `desc:"similarity matricies for each layer"`
// 	V1Sims     []float64                `desc:"similarity for each layer relative to V1"`
// 	CatDists   []float64                `desc:"AvgContrastDist for each layer under CanonicalCats centroid meta categories"`
// 	BasicDists []float64                `desc:"AvgBasicDist for each layer -- basic-level distances"`
// 	ExptDists  []float64                `desc:"AvgExptDist for each layer -- distances from expt data"`
// 	Cat5Sims   map[string]*simat.SimMat `desc:"similarity matricies for each layer, organized into CanonicalCats and sorted"`
// 	Cat5Objs   map[string]*[]string     `desc:"corresponding ordering of objects in sorted Cat5Sims lists"`
// 	PermNCats  map[string]int           `desc:"number of categories remaining after permutation from LbaCat"`
// 	PermDists  map[string]float64       `desc:"avg contrast dist for permutation"`
// }
//
// // SetCats sets the categories from given list of category/object_file names
// func (rs *RSA) SetCats(objs []string) {
// 	rs.Cats = make([]string, 0, 20*20)
// 	for _, ob := range objs {
// 		cat := strings.Split(ob, "/")[0]
// 		rs.Cats = append(rs.Cats, cat)
// 	}
// }
//
// func (rs *RSA) SimByName(cn string) *simat.SimMat {
// 	sm, ok := rs.Sims[cn]
// 	if !ok || sm == nil {
// 		sm = &simat.SimMat{}
// 		rs.Sims[cn] = sm
// 	}
// 	return sm
// }
//
// func (rs *RSA) Cat5SimByName(cn string) *simat.SimMat {
// 	sm, ok := rs.Cat5Sims[cn]
// 	if !ok || sm == nil {
// 		sm = &simat.SimMat{}
// 		rs.Cat5Sims[cn] = sm
// 	}
// 	return sm
// }
//
// func (rs *RSA) Cat5ObjByName(cn string) *[]string {
// 	sm, ok := rs.Cat5Objs[cn]
// 	if !ok || sm == nil {
// 		nsm := sliceclone.String(rs.Cats)
// 		sm = &nsm
// 		rs.Cat5Objs[cn] = sm
// 	}
// 	return sm
// }
//
// // StatsFmActs computes RSA stats from given acts table, for given columns (layer names)
// func (rs *RSA) StatsFmActs(acts *etable.Table, lays []string) {
// 	tick := 2 // use this tick for analyses..
// 	tix := etable.NewIdxView(acts)
// 	tix.Filter(func(et *etable.Table, row int) bool {
// 		tck := int(et.CellFloat("Tick", row))
// 		return tck == tick
// 	})
//
// 	expt := rs.SimByName("Expt1")
//
// 	for i, cn := range lays {
// 		sm := rs.SimByName(cn)
// 		rs.SimMatFmActs(sm, tix, cn)
//
// 		osm := rs.SimByName(cn + "_Obj")
// 		rs.ObjSimMat(osm, sm, rs.Cats)
//
// 		dist := metric.CrossEntropy64(osm.Mat.(*tensor.Float64).Values, expt.Mat.(*tensor.Float64).Values)
// 		rs.ExptDists[i] = dist
// 	}
//
// 	v1sm := rs.Sims["V1m"]
// 	v1sm64 := v1sm.Mat.(*tensor.Float64)
// 	for i, cn := range lays {
// 		osm := rs.SimByName(cn)
//
// 		rs.CatDists[i] = -rs.AvgContrastDist(osm, rs.Cats, CanonicalCats)
// 		rs.BasicDists[i] = rs.AvgBasicDist(osm, rs.Cats)
//
// 		if v1sm == osm {
// 			rs.V1Sims[i] = 1
// 			continue
// 		}
// 		osm64 := osm.Mat.(*tensor.Float64)
// 		rs.V1Sims[i] = metric.Correlation64(osm64.Values, v1sm64.Values)
// 	}
// 	cat5s := []string{"TE"}
// 	for _, cn := range cat5s {
// 		rs.StatsSortPermuteCat5(cn)
// 	}
// }
//
// func (rs *RSA) StatsSortPermuteCat5(laynm string) {
// 	sm := rs.SimByName(laynm)
// 	if len(sm.Rows) == 0 {
// 		return
// 	}
// 	sm5 := rs.Cat5SimByName(laynm)
// 	obj := rs.CatSortSimMat(sm, sm5, rs.Cats, CanonicalCats, true, laynm+"_LbaCat")
// 	obj5 := rs.Cat5ObjByName(laynm)
// 	copy(*obj5, obj)
// 	pnm := laynm + "perm"
// 	pcats, ncat, pdist := rs.PermuteCatTest(sm, rs.Cats, CanonicalCats, pnm)
// 	sm5p := rs.Cat5SimByName(pnm)
// 	objp := rs.CatSortSimMat(sm, sm5p, rs.Cats, pcats, true, pnm)
// 	obj5p := rs.Cat5ObjByName(pnm)
// 	copy(*obj5p, objp)
// 	rs.PermNCats[laynm] = ncat
// 	rs.PermDists[laynm] = pdist
// }
//
// // ConfigSimMat sets meta data
// func (rs *RSA) ConfigSimMat(sm *simat.SimMat) {
// 	smat := sm.Mat.(*tensor.Float64)
// 	smat.SetMetaData("max", "2")
// 	smat.SetMetaData("min", "0")
// 	smat.SetMetaData("colormap", "Viridis")
// 	smat.SetMetaData("grid-fill", "1")
// 	smat.SetMetaData("dim-extra", "0.5")
// }
//
// // OpenSimMat opens a saved sim mat for given layer name,
// // using given cat strings per row of sim mat
// func (rs *RSA) OpenSimMat(laynm string, fname gi.FileName) {
// 	sm := rs.SimByName(laynm)
// 	no := len(rs.Cats)
// 	sm.Init()
// 	rs.ConfigSimMat(sm)
// 	smat := sm.Mat.(*tensor.Float64)
// 	smat.SetShape([]int{no, no}, nil, nil)
// 	err := tensor.OpenCSV(smat, fname, etable.Tab.Rune())
// 	if err != nil {
// 		log.Println(err)
// 		return
// 	}
// 	sm.Rows = simat.BlankRepeat(rs.Cats)
// 	sm.Cols = sm.Rows
// 	rs.StatsSortPermuteCat5(laynm)
// 	rs.PermDists[laynm+"_BasicDist"] = rs.AvgBasicDist(sm, rs.Cats)
//
// 	expt := rs.SimByName("Expt1")
//
// 	osm := rs.SimByName(laynm + "_Obj")
// 	rs.ObjSimMat(osm, sm, rs.Cats)
// 	dist := metric.CrossEntropy64(osm.Mat.(*tensor.Float64).Values, expt.Mat.(*tensor.Float64).Values)
// 	rs.PermDists[laynm+"_ExptDist"] = dist
//
// }
//
// // CatSortSimMat takes an input sim matrix and categorizes the items according to given cats
// // and then sorts items within that according to their average within - between cat similarity.
// // contrast = use within - between metric, otherwise just within
// // returns the new ordering of objects (like nms but sorted according to new sort)
// func (rs *RSA) CatSortSimMat(insm *simat.SimMat, osm *simat.SimMat, nms []string, catmap map[string]string, contrast bool, name string) []string {
// 	no := len(insm.Rows)
// 	sch := etable.Schema{
// 		{"Cat", tensor.STRING, nil, nil},
// 		{"Dist", tensor.FLOAT64, nil, nil},
// 		{"Obj", tensor.STRING, nil, nil},
// 	}
// 	dt := &etable.Table{}
// 	dt.SetFromSchema(sch, no)
// 	cats := dt.Cols[0].(*tensor.String).Values
// 	dists := dt.Cols[1].(*tensor.Float64).Values
// 	objs := dt.Cols[2].(*tensor.String).Values
// 	for i, nm := range nms {
// 		cats[i] = catmap[nm]
// 		objs[i] = nm
// 	}
// 	smatv := insm.Mat.(*tensor.Float64).Values
// 	avgCtrstDist := 0.0
// 	for ri := 0; ri < no; ri++ {
// 		roff := ri * no
// 		aid := 0.0
// 		ain := 0
// 		abd := 0.0
// 		abn := 0
// 		rc := cats[ri]
// 		for ci := 0; ci < no; ci++ {
// 			if ri == ci {
// 				continue
// 			}
// 			cc := cats[ci]
// 			d := smatv[roff+ci]
// 			if cc == rc {
// 				aid += d
// 				ain++
// 			} else {
// 				abd += d
// 				abn++
// 			}
// 		}
// 		if ain > 0 {
// 			aid /= float64(ain)
// 		}
// 		if abn > 0 {
// 			abd /= float64(abn)
// 		}
// 		dval := aid
// 		if contrast {
// 			dval -= abd
// 		}
// 		dists[ri] = dval
// 		avgCtrstDist += (1 - aid) - (1 - abd)
// 	}
// 	avgCtrstDist /= float64(no)
// 	ix := etable.NewIdxView(dt)
// 	ix.SortColNames([]string{"Cat", "Dist"}, true) // ascending
// 	osm.Init()
// 	osm.Mat.CopyShapeFrom(insm.Mat)
// 	osm.Mat.CopyMetaData(insm.Mat)
// 	rs.ConfigSimMat(osm)
// 	omatv := osm.Mat.(*tensor.Float64).Values
// 	bcols := make([]string, no)
// 	last := ""
// 	for sri := 0; sri < no; sri++ {
// 		sroff := sri * no
// 		ri := ix.Idxs[sri]
// 		roff := ri * no
// 		cat := cats[ri]
// 		if cat != last {
// 			bcols[sri] = cat
// 			last = cat
// 		}
// 		// bcols[sri] = nms[ri] // uncomment this to see all the names
// 		for sci := 0; sci < no; sci++ {
// 			ci := ix.Idxs[sci]
// 			d := smatv[roff+ci]
// 			omatv[sroff+sci] = d
// 		}
// 	}
// 	osm.Rows = bcols
// 	osm.Cols = bcols
// 	if Debug {
// 		fmt.Printf("%v  avg contrast dist: %.4f\n", name, avgCtrstDist)
// 	}
// 	sobjs := make([]string, no)
// 	for i := 0; i < no; i++ {
// 		nm := nms[ix.Idxs[i]]
// 		sobjs[i] = catmap[nm] + ": " + nm
// 	}
// 	return sobjs
// }
//
// // AvgContrastDist computes average contrast dist over given cat map
// // nms gives the base category names for each row in the simat, which is
// // then used to lookup the meta category in the catmap, which is used
// // for determining the within vs. between category status.
// func (rs *RSA) AvgContrastDist(insm *simat.SimMat, nms []string, catmap map[string]string) float64 {
// 	no := len(insm.Rows)
// 	smatv := insm.Mat.(*tensor.Float64).Values
// 	avgd := 0.0
// 	for ri := 0; ri < no; ri++ {
// 		roff := ri * no
// 		aid := 0.0
// 		ain := 0
// 		abd := 0.0
// 		abn := 0
// 		rnm := nms[ri]
// 		rc := catmap[rnm]
// 		for ci := 0; ci < no; ci++ {
// 			if ri == ci {
// 				continue
// 			}
// 			cnm := nms[ci]
// 			cc := catmap[cnm]
// 			d := smatv[roff+ci]
// 			if cc == rc {
// 				aid += d
// 				ain++
// 			} else {
// 				abd += d
// 				abn++
// 			}
// 		}
// 		if ain > 0 {
// 			aid /= float64(ain)
// 		}
// 		if abn > 0 {
// 			abd /= float64(abn)
// 		}
// 		avgd += aid - abd
// 	}
// 	avgd /= float64(no)
// 	return avgd
// }
//
// // AvgBasicDist computes average distance within basic-level categories given by nms
// func (rs *RSA) AvgBasicDist(insm *simat.SimMat, nms []string) float64 {
// 	no := len(insm.Rows)
// 	smatv := insm.Mat.(*tensor.Float64).Values
// 	avgd := 0.0
// 	ain := 0
// 	for ri := 0; ri < no; ri++ {
// 		roff := ri * no
// 		rnm := nms[ri]
// 		for ci := 0; ci < ri; ci++ {
// 			cnm := nms[ci]
// 			d := smatv[roff+ci]
// 			if rnm == cnm {
// 				avgd += d
// 				ain++
// 			}
// 		}
// 	}
// 	if ain > 0 {
// 		avgd /= float64(ain)
// 	}
// 	return avgd
// }
//
// // PermuteCatTest takes an input sim matrix and tries all one-off permutations relative to given
// // initial set of categories, and computes overall average constrast distance for each
// // selects categs with lowest dist and iterates until no better permutation can be found.
// // returns new map, number of categories used in new map, and the avg contrast distance for it
// func (rs *RSA) PermuteCatTest(insm *simat.SimMat, nms []string, catmap map[string]string, desc string) (map[string]string, int, float64) {
// 	if Debug {
// 		fmt.Printf("\n#########\n%v\n", desc)
// 	}
// 	catm := map[string]int{} // list of categories and index into catnms
// 	catnms := []string{}
// 	for _, nm := range nms {
// 		cat := catmap[nm]
// 		if _, has := catm[cat]; !has {
// 			catm[cat] = len(catnms)
// 			catnms = append(catnms, cat)
// 		}
// 	}
// 	ncats := len(catnms)
//
// 	itrmap := make(map[string]string)
// 	for k, v := range catmap {
// 		itrmap[k] = v
// 	}
//
// 	std := rs.AvgContrastDist(insm, nms, catmap)
// 	if Debug {
// 		fmt.Printf("std: %.4f  starting\n", std)
// 	}
//
// 	for itr := 0; itr < 100; itr++ {
// 		std = rs.AvgContrastDist(insm, nms, itrmap)
//
// 		effmap := make(map[string]string)
// 		mind := 100.0
// 		mindnm := ""
// 		mindcat := ""
// 		for _, nm := range nms { // go over each item
// 			cat := itrmap[nm]
// 			for oc := 0; oc < ncats; oc++ { // go over alternative categories
// 				ocat := catnms[oc]
// 				if ocat == cat {
// 					continue
// 				}
// 				for k, v := range itrmap {
// 					if k == nm {
// 						effmap[k] = ocat // switch
// 					} else {
// 						effmap[k] = v
// 					}
// 				}
// 				avgd := rs.AvgContrastDist(insm, nms, effmap)
// 				if avgd < mind {
// 					mind = avgd
// 					mindnm = nm
// 					mindcat = ocat
// 				}
// 				// if avgd < std {
// 				// 	fmt.Printf("Permute test better than std dist: %v  min dist: %v  for name: %v  in cat: %v\n", std, avgd, nm, ocat)
// 				// }
// 			}
// 		}
// 		if mind >= std {
// 			break
// 		}
// 		if Debug {
// 			fmt.Printf("itr %v std: %.4f  min: %.4f  name: %v  cat: %v\n", itr, std, mind, mindnm, mindcat)
// 		}
// 		itrmap[mindnm] = mindcat // make the switch
// 	}
// 	if Debug {
// 		fmt.Printf("std: %.4f  final\n", std)
// 	}
//
// 	nCatUsed := 0
// 	for oc := 0; oc < ncats; oc++ {
// 		cat := catnms[oc]
// 		if Debug {
// 			fmt.Printf("%v\n", cat)
// 		}
// 		nin := 0
// 		for _, nm := range Objs {
// 			ct := itrmap[nm]
// 			if ct == cat {
// 				nin++
// 				if Debug {
// 					fmt.Printf("\t%v\n", nm)
// 				}
// 			}
// 		}
// 		if nin > 0 {
// 			nCatUsed++
// 		}
// 	}
// 	return itrmap, nCatUsed, -std
// }
//
// // ObjSimMat compresses full simat into a much smaller per-object sim mat
// func (rs *RSA) ObjSimMat(osm *simat.SimMat, fsm *simat.SimMat, nms []string) {
// 	fsmat := fsm.Mat.(*tensor.Float64)
//
// 	ono := len(Objs)
// 	osm.Init()
// 	osmat := osm.Mat.(*tensor.Float64)
// 	osmat.SetShape([]int{ono, ono}, nil, nil)
// 	osm.Rows = LbaCatsBlanks
// 	osm.Cols = LbaCatsBlanks
// 	osmat.SetMetaData("max", "1")
// 	osmat.SetMetaData("min", "0")
// 	osmat.SetMetaData("colormap", "Viridis")
// 	osmat.SetMetaData("grid-fill", "1")
// 	osmat.SetMetaData("dim-extra", "0.15")
//
// 	nmat := &tensor.Float64{}
// 	nmat.SetShape([]int{ono, ono}, nil, nil)
//
// 	nf := len(nms)
// 	for ri := 0; ri < nf; ri++ {
// 		roi := ObjIdxs[nms[ri]]
// 		for ci := 0; ci < nf; ci++ {
// 			sidx := ri*nf + ci
// 			sval := fsmat.Values[sidx]
// 			coi := ObjIdxs[nms[ci]]
// 			oidx := roi*ono + coi
// 			if ri == ci {
// 				osmat.Values[oidx] = 0
// 			} else {
// 				osmat.Values[oidx] += sval
// 			}
// 			nmat.Values[oidx] += 1
// 		}
// 	}
// 	for ri := 0; ri < ono; ri++ {
// 		for ci := 0; ci < ono; ci++ {
// 			oidx := ri*ono + ci
// 			osmat.Values[oidx] /= nmat.Values[oidx]
// 		}
// 	}
// 	norm.DivNorm64(osmat.Values, norm.Max64)
// }
