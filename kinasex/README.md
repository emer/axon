# KinasEx

This package contains experimental Kinase-based learning rules.

See https://github.com/emer/axon/tree/master/examples/kinaseq for exploration of the implemented equations, and https://github.com/ccnlab/kinase/tree/main/sims/kinase for biophysical basis of the equations.

In the initially implemented nomenclature (early 2022), the space of algorithms was enumerated in `kinase/rules.go` as follows:

```Go
const (
	// SynSpkCont implements synaptic-level Ca signals at an abstract level,
	// purely driven by spikes, not NMDA channel Ca, as a product of
	// sender and recv CaSyn values that capture the decaying Ca trace
	// from spiking, qualitatively as in the NMDA dynamics.  These spike-driven
	// Ca signals are integrated in a cascaded manner via CaM,
	// then CaP (reflecting CaMKII) and finally CaD (reflecting DAPK1).
	// It uses continuous learning based on temporary DWt (TDWt) values
	// based on the TWindow around spikes, which convert into DWt after
	// a pause in synaptic activity (no arbitrary ThetaCycle boundaries).
	// There is an option to compare with SynSpkTheta by only doing DWt updates
	// at the theta cycle level, in which case the key difference is the use of
	// TDWt, which can remove some variability associated with the arbitrary
	// timing of the end of trials.
	SynSpkCont Rules = iota

	// SynNMDACont is the same as SynSpkCont with NMDA-driven calcium signals
	// computed according to the very close approximation to the
	// Urakubo et al (2008) allosteric NMDA dynamics, then integrated at P vs. D
	// time scales.  This is the most biologically realistic yet computationally
	// tractable verseion of the Kinase learning algorithm.
	SynNMDACont

	// SynSpkTheta abstracts the SynSpkCont algorithm by only computing the
	// DWt change at the end of the ThetaCycle, instead of continuous updating.
	// This allows an optimized implementation that is roughly 1/3 slower than
	// the fastest NeurSpkTheta version, while still capturing much of the
	// learning dynamics by virtue of synaptic-level integration.
	SynSpkTheta

	// NeurSpkTheta uses neuron-level spike-driven calcium signals
	// integrated at P vs. D time scales -- this is the original
	// Leabra and Axon XCAL / CHL learning rule.
	// It exhibits strong sensitivity to final spikes and thus
	// high levels of variance.
	NeurSpkTheta
)
```

This package contains implementations of `SynSpkCont` and `SynNMDACont`.

