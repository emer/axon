# slow AHP (sAHP): Calcium 

This channel is driven by a M-type mechanism operating on calcium sensor pathways that have longer time constants, updated at the theta cycle level so that guarantees a long baseline already.  The logistic gating function operates with a high cutoff on the underlying Ca signal

The logistic operates on integrated Ca:

```Go
	co = (ca - Off)
	a = co / TauMax * (1 - exp(-co/Slope))
	b = -co / TauMax * (1 - exp(co/Slope))

   tau = 1 / (a + b)
   ninf = a / (a + b)
```

![Ninf from Ca](fig_sahp_ninf.png?raw=true "Ninf as a function of V (biological units)")

![Tau from Ca](fig_sahp_tau.png?raw=true "Tau as a function of V (biological units)")

![mAHP vs. fKNA](fig_mahp_vs_fkna.png?raw=true "mAHP vs. fast KNa (tau = 50, rise = 0.05)")

The `Time Run` plot (above) shows how the mAHP current develops over time in response to spiking, in comparison to the KNa function, with the default "fast" parameters of 50msec and rise = 0.05.  The mAHP is much more "anticipatory" due to the direct V sensitivity, whereas KNa is much more "reactive", responding only the the Na after spiking.


