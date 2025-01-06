# Linear approximation to synaptic calcium

This package computes a GLM linear regression against the CaBin time bins of CaSyn values to efficiently compute CaP and CaD at the synaptic level.

It is based on sims/kinasesim code, but is more focused on the specific task of computing the regression, sampling sender and receiver firing rates across minus and plus phases in a fully crossed manner.

To run, just do `go test`.

Edit the `linear_test.go` file to set different parameters.

Here are the current outputs for 200 cycles, Bins = 25, Plus = 50:

```
...
30 0.007846212365101512
CaP	R^2: 0.991438	R:  0.99571	Var Err: 0.004658	 Obs:    0.544
CaD	R^2: 0.996128	R: 0.998062	Var Err: 0.002602	 Obs:    0.672

CaP = 	0.0985391 * IV_0 + 	0.393702 * IV_1 + 	0.491684 * IV_2 + 	 0.59146 * IV_3 + 	0.692398 * IV_4 + 	0.792826 * IV_5 + 	 1.90145 * IV_6 + 	 3.00666 * IV_7 + 	       0
CaD = 	    0.35 * IV_0 + 	 0.65702 * IV_1 + 	0.956254 * IV_2 + 	 1.25657 * IV_3 + 	 1.25812 * IV_4 + 	 1.25885 * IV_5 + 	 1.13701 * IV_6 + 	 1.01161 * IV_7 + 	       0
```

linear.go has the initial coefficients, which are implemented in the `kinase.CaBinWts` function, set to:
```Go
	// NBins = 8, 200+50 cycles for CaSyn
	r.Coeff.Values = []float64{
		0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 1.9, 3.0, 0, // big at the end; insensitive to start
		0.35, 0.65, 0.95, 1.25, 1.25, 1.25, 1.125, 1.0, .0} // up and down
```
(note that the final 0 is the constant offset)

and for 300 cycles:
```Go
	// NBins = 12, 300+50 cycles for CaSyn
	// r.Coeff.Values = []float64{
	// 	0, 0, 0, 0, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 1.9, 3.0, 0, // big at the end; insensitive to start
	// 	0, 0, 0, 0, 0.35, 0.65, 0.95, 1.25, 1.25, 1.25, 1.125, 1.0, .0} // up and down
```

So 300 cycles just appends zeros to the start, and provides a very good fit.




