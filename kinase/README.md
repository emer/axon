# Kinase Learning Implementation

This implements central elements of the Kinase learning rule, including variables with associated time constants used for integrating calcium signals through the cascade of progressively longer time-integrals, from `Ca` -> `CaM` calmodulin (at `MTau`) -> CaMKII (`CaP` at `PTau` for LTP role) -> DAPK1 (`CaD` at `DTau` for LTD role).

See [kinaseq example](https://github.com/emer/axon/tree/main/examples/kinaseq) for an exploration of the implemented equations, and [kinase repository](https://github.com/ccnlab/kinase/tree/main/sims/kinase) for documentation and simulations about the biophysical basis of the equations.

# Time constants and Variables

* `MTau` (2 or 5) = calmodulin (`CaM`) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau.

* `PTau` (40) = LTP spike-driven Ca factor (`CaP`) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information.

* `DTau` (40) = LTD spike-driven Ca factor (`CaD`) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome).


# TODO

* remove NMDA and CaLrn from neuron if not used
* use neuron-level recv, send final CaSpk* values as regressors!  key!

# Closed-form Expression for cascading integration (not faster)

Rishi Chaudhri suggested the following approach:

[Wolfram Alpha Solution](https://www.wolframalpha.com/input?i=dx%2Fdt+%3D+-a*x%2C+dy%2Fdt+%3D+b*x+-+b*y%2C+dz%2Fdt+%3D+c*y+-+c*z)

But unfortunately all the FastExp calls end up being slower than directly computing the cascaded equations.

```go
// Equations for below, courtesy of Rishi Chaudhri:
// 
// CaAtT computes the 3 Ca values at (currentTime + ti), assuming 0
// new Ca incoming (no spiking). It uses closed-form exponential functions.
func (kp *CaDtParams) CaAtT(ti int32, caM, caP, caD *float32) {
	t := float32(ti)
	mdt := kp.MDt
	pdt := kp.PDt
	ddt := kp.DDt
	if kp.ExpAdj.IsTrue() { // adjust for discrete
		mdt *= 1.11
		pdt *= 1.03
		ddt *= 1.03
	}
	mi := *caM
	pi := *caP
	di := *caD

	*caM = mi * math32.FastExp(-t*mdt)

	em := math32.FastExp(t * mdt)
	ep := math32.FastExp(t * pdt)

	*caP = pi*math32.FastExp(-t*pdt) - (pdt*mi*math32.FastExp(-t*(mdt+pdt))*(em-ep))/(pdt-mdt)

	epd := math32.FastExp(t * (pdt + ddt))
	emd := math32.FastExp(t * (mdt + ddt))
	emp := math32.FastExp(t * (mdt + pdt))

	*caD = pdt*ddt*mi*math32.FastExp(-t*(mdt+pdt+ddt))*(ddt*(emd-epd)+(pdt*(epd-emp))+mdt*(emp-emd))/((mdt-pdt)*(mdt-ddt)*(pdt-ddt)) - ddt*pi*math32.FastExp(-t*(pdt+ddt))*(ep-math32.FastExp(t*ddt))/(ddt-pdt) + di*math32.FastExp(-t*ddt)
}
```

# 

