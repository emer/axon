// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"
	"math"

	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
)

// Regression contains results and parameters for running a
// multivariate linear regression, supporting multiple independent
// and dependent variables.  Make a NewRegression and then do
// Run() on a tensor table.IndexView with the relevant data in
// columns of the table.  Batch-mode gradient descent is used
// and the relevant parameters can be altered from defaults before
// calling Run as needed.
type Regression struct {
	// Coeff are the coefficients to map from input independent variables
	// to the dependent variables.  The first, outer dimension is number of
	// dependent variables, and the second, inner dimension is number of
	// independent variables plus one for the offset (b) (last element).
	Coeff tensor.Float64

	// mean squared error for the regression
	MSE float64

	// R2 is the r^2 total variance accounted for by the linear regression,
	// for each dependent variable = 1 - (ErrVariance / ObsVariance)
	R2 []float64

	// Observed variance of each of the dependent variables to be predicted.
	ObsVariance []float64

	// Variance of the error residuals per dependent variables
	ErrVariance []float64

	//	optional names of the independent variables, for reporting results
	IndepNames []string

	//	optional names of the dependent variables, for reporting results
	DepNames []string

	///////////////////////////////////////////
	// Parameters of the regression:

	// ZeroOffset restricts the offset of the linear function to 0,
	// forcing it to pass through the origin.  Otherwise, a constant offset "b"
	// is fit during the regression process.
	ZeroOffset bool

	// learning rate parameter, which can be adjusted to reduce iterations based on
	// specific properties of the data, but the default is reasonable for most "typical" data.
	LRate float64 `default:"0.1"`

	// tolerance on difference in mean squared error (MSE) across iterations to stop
	// iterating and consider the result to be converged.
	StopTolerance float64 `default:"0.0001"`

	// Constant cost factor subtracted from weights, for the L1 norm or "Lasso"
	// regression.  This is good for producing sparse results but can arbitrarily
	// select one of multiple correlated independent variables.
	L1Cost float64

	// Cost factor proportional to the coefficient value, for the L2 norm or "Ridge"
	// regression.  This is good for generally keeping weights small and equally
	// penalizes correlated independent variables.
	L2Cost float64

	// CostStartIter is the iteration when we start applying the L1, L2 Cost factors.
	// It is often a good idea to have a few unconstrained iterations prior to
	// applying the cost factors.
	CostStartIter int `default:"5"`

	// maximum number of iterations to perform
	MaxIters int `default:"50"`

	///////////////////////////////////////////
	// Cached values from the table

	// Table of data
	Table *table.IndexView

	// tensor columns from table with the respective variables
	IndepVars, DepVars, PredVars, ErrVars tensor.Tensor

	// Number of independent and dependent variables
	NIndepVars, NDepVars int
}

func NewRegression() *Regression {
	rr := &Regression{}
	rr.Defaults()
	return rr
}

func (rr *Regression) Defaults() {
	rr.LRate = 0.1
	rr.StopTolerance = 0.001
	rr.MaxIters = 50
	rr.CostStartIter = 5
}

func (rr *Regression) init(nIv, nDv int) {
	rr.NIndepVars = nIv
	rr.NDepVars = nDv
	rr.Coeff.SetShape([]int{nDv, nIv + 1}, "DepVars", "IndepVars")
	rr.R2 = make([]float64, nDv)
	rr.ObsVariance = make([]float64, nDv)
	rr.ErrVariance = make([]float64, nDv)
	rr.IndepNames = make([]string, nIv)
	rr.DepNames = make([]string, nDv)
}

// SetTable sets the data to use from given indexview of table, where
// each of the Vars args specifies a column in the table, which can have either a
// single scalar value for each row, or a tensor cell with multiple values.
// predVars and errVars (predicted values and error values) are optional.
func (rr *Regression) SetTable(ix *table.IndexView, indepVars, depVars, predVars, errVars string) error {
	dt := ix.Table
	iv, err := dt.ColumnByNameTry(indepVars)
	if err != nil {
		return err
	}
	dv, err := dt.ColumnByNameTry(depVars)
	if err != nil {
		return err
	}
	var pv, ev tensor.Tensor
	if predVars != "" {
		pv, err = dt.ColumnByNameTry(predVars)
		if err != nil {
			return err
		}
	}
	if errVars != "" {
		ev, err = dt.ColumnByNameTry(errVars)
		if err != nil {
			return err
		}
	}
	if pv != nil && !pv.Shape().IsEqual(dv.Shape()) {
		return fmt.Errorf("predVars must have same shape as depVars")
	}
	if ev != nil && !ev.Shape().IsEqual(dv.Shape()) {
		return fmt.Errorf("errVars must have same shape as depVars")
	}
	_, nIv := iv.RowCellSize()
	_, nDv := dv.RowCellSize()
	rr.init(nIv, nDv)
	rr.Table = ix
	rr.IndepVars = iv
	rr.DepVars = dv
	rr.PredVars = pv
	rr.ErrVars = ev
	return nil
}

// Run performs the multi-variate linear regression using data SetTable function,
// learning linear coefficients and an overall static offset that best
// fits the observed dependent variables as a function of the independent variables.
// Initial values of the coefficients, and other parameters for the regression,
// should be set prior to running.
func (rr *Regression) Run() {
	ix := rr.Table
	iv := rr.IndepVars
	dv := rr.DepVars
	pv := rr.PredVars
	ev := rr.ErrVars

	if pv == nil {
		pv = dv.Clone()
	}
	if ev == nil {
		ev = dv.Clone()
	}

	nDv := rr.NDepVars
	nIv := rr.NIndepVars
	nCi := nIv + 1

	dc := rr.Coeff.Clone().(*tensor.Float64)

	lastItr := false
	sse := 0.0
	prevmse := 0.0
	n := ix.Len()
	norm := 1.0 / float64(n)
	lrate := norm * rr.LRate
	for itr := 0; itr < rr.MaxIters; itr++ {
		for i := range dc.Values {
			dc.Values[i] = 0
		}
		sse = 0
		if (itr+1)%10 == 0 {
			lrate *= 0.5
		}
		for i := 0; i < n; i++ {
			row := ix.Indexes[i]
			for di := 0; di < nDv; di++ {
				pred := 0.0
				for ii := 0; ii < nIv; ii++ {
					pred += rr.Coeff.Value([]int{di, ii}) * iv.FloatRowCell(row, ii)
				}
				if !rr.ZeroOffset {
					pred += rr.Coeff.Value([]int{di, nIv})
				}
				targ := dv.FloatRowCell(row, di)
				err := targ - pred
				sse += err * err
				for ii := 0; ii < nIv; ii++ {
					dc.Values[di*nCi+ii] += err * iv.FloatRowCell(row, ii)
				}
				if !rr.ZeroOffset {
					dc.Values[di*nCi+nIv] += err
				}
				if lastItr {
					pv.SetFloatRowCell(row, di, pred)
					if ev != nil {
						ev.SetFloatRowCell(row, di, err)
					}
				}
			}
		}
		for di := 0; di < nDv; di++ {
			for ii := 0; ii <= nIv; ii++ {
				if rr.ZeroOffset && ii == nIv {
					continue
				}
				idx := di*(nCi+1) + ii
				w := rr.Coeff.Values[idx]
				d := dc.Values[idx]
				sgn := 1.0
				if w < 0 {
					sgn = -1.0
				} else if w == 0 {
					sgn = 0
				}
				rr.Coeff.Values[idx] += lrate * (d - rr.L1Cost*sgn - rr.L2Cost*w)
			}
		}
		rr.MSE = norm * sse
		if lastItr {
			break
		}
		if itr > 0 {
			dmse := rr.MSE - prevmse
			if math.Abs(dmse) < rr.StopTolerance || itr == rr.MaxIters-2 {
				lastItr = true
			}
		}
		fmt.Println(itr, rr.MSE)
		prevmse = rr.MSE
	}

	obsMeans := make([]float64, nDv)
	errMeans := make([]float64, nDv)
	for i := 0; i < n; i++ {
		row := ix.Indexes[i]
		for di := 0; di < nDv; di++ {
			obsMeans[di] += dv.FloatRowCell(row, di)
			errMeans[di] += ev.FloatRowCell(row, di)
		}
	}
	for di := 0; di < nDv; di++ {
		obsMeans[di] *= norm
		errMeans[di] *= norm
		rr.ObsVariance[di] = 0
		rr.ErrVariance[di] = 0
	}
	for i := 0; i < n; i++ {
		row := ix.Indexes[i]
		for di := 0; di < nDv; di++ {
			o := dv.FloatRowCell(row, di) - obsMeans[di]
			rr.ObsVariance[di] += o * o
			e := ev.FloatRowCell(row, di) - errMeans[di]
			rr.ErrVariance[di] += e * e
		}
	}
	for di := 0; di < nDv; di++ {
		rr.ObsVariance[di] *= norm
		rr.ErrVariance[di] *= norm
		rr.R2[di] = 1.0 - (rr.ErrVariance[di] / rr.ObsVariance[di])
	}
}

// Variance returns a description of the variance accounted for by the regression
// equation, R^2, for each dependent variable, along with the variances of
// observed and errors (residuals), which are used to compute it.
func (rr *Regression) Variance() string {
	str := ""
	for di := range rr.R2 {
		if len(rr.DepNames) > di && rr.DepNames[di] != "" {
			str += rr.DepNames[di]
		} else {
			str += fmt.Sprintf("DV %d", di)
		}
		str += fmt.Sprintf("\tR^2: %8.6g\tR: %8.6g\tVar Err: %8.4g\t Obs: %8.4g\n", rr.R2[di], math.Sqrt(rr.R2[di]), rr.ErrVariance[di], rr.ObsVariance[di])
	}
	return str
}

// Coeffs returns a string describing the coefficients
func (rr *Regression) Coeffs() string {
	str := ""
	for di := range rr.NDepVars {
		if len(rr.DepNames) > di && rr.DepNames[di] != "" {
			str += rr.DepNames[di]
		} else {
			str += fmt.Sprintf("DV %d", di)
		}
		str += " = "
		for ii := 0; ii <= rr.NIndepVars; ii++ {
			str += fmt.Sprintf("\t%8.6g", rr.Coeff.Value([]int{di, ii}))
			if ii < rr.NIndepVars {
				str += " * "
				if len(rr.IndepNames) > ii && rr.IndepNames[di] != "" {
					str += rr.IndepNames[di]
				} else {
					str += fmt.Sprintf("IV_%d", ii)
				}
				str += " + "
			}
		}
		str += "\n"
	}
	return str
}
