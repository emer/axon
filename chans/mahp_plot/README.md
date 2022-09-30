# M-type voltage gated potassium channel: mAHP

This voltage-gated K channel, which is also inactivated by ACh muscarinic receptor activation, plays a role in medium time-scale afterhyperpolarization (mAHP).

Original formulation due to Mainen & Sejnowski (1996) is widely used, and used here.

```Go
	vo = (V - Voff)
	a = 0.001 * vo / (1 - exp(-vo/Vslope))
	b = -0.001 * vo / (1 - exp(vo/Vslope))

   tau = 1 / (a + b)
   ninf = a / (a + b)
```

![Ninf from V](fig_mahp_ninf.png?raw=true "Ninf as a function of V (biological units)")

![Tau from V](fig_mahp_tau.png?raw=true "Tau as a function of V (biological units)")

![mAHP vs. fKNA](fig_mahp_vs_fkna.png?raw=true "mAHP vs. fast KNa (tau = 50, rise = 0.05)")

The `Time Run` plot (above) shows how the mAHP current develops over time in response to spiking, in comparison to the KNa function, with the default "fast" parameters of 50msec and rise = 0.05.  The mAHP is much more "anticipatory" due to the direct V sensitivity, whereas KNa is much more "reactive", responding only the the Na after spiking.


