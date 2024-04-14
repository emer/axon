# A-type Potassium Channel

This voltage-gated K channel, found especially on dendrites of pyramidal neurons for example, has a narrow window of activation between the M and H gates, around -37 mV.  The M activating gate has a fast time constant on the scale of a msec, while the H inactivating gate has a linearly increasing time constant as a function of V.

It is particularly important for counteracting the excitatory effects of voltage gated calcium channels which can otherwise drive runaway excitatory currents.

```Go
	K = -1.8 - 1/(1+exp((vbio+40)/5))
```

```Go
    Alpha(V,K) = exp(0.03707 * K * (V - 1))
```

```Go
	Beta(V,K) = exp(0.01446 * K * (V - 1))
```

```Go
    H(V) = 1 / (1 + epx(0.1133 * (V + 56)))
```

```Go
    Htau(V) = 0.26 * (V + 50) > 2
```

```Go
    M(Alpha) = 1 / (1 + Alpha)
```

```Go
    Mtau(Alpha, Beta) = 1 + Beta / (0.5 * (1 + Alpha))
```

NOTE: to work properly with 1 msec Dt updating, the MTau adds a 1 -- otherwise MTau goes to high.

![G from V](fig_ak_chan_g_from_v.png?raw=true "G = M * H as a function of V (biological units)")

![M and H gating from V](fig_ak_chan_m_h_from_v.png?raw=true "M and H gating factors as a function of V (biological units)")

![Mtau from V](fig_ak_chan_mtau_from_v.png?raw=true "Mtau rate of change of M as a function of V (biological units)")

![Htau from V](fig_ak_chan_htau_from_v.png?raw=true "Htau rate of change of H as a function of V (biological units)")

![Gak and M over time](fig_ak_chan_time_plot.png?raw=true "Gak developing over time in response to simulated spiking potentials, with a baseline Vm of -50 -- H quickly inactivates.")

The `Time Run` plot (above) shows how the Gak current develops over time in response to spiking, which it tracks directly due to very fast M dynamics.  The H current inactivates significantly when the consistent Vm level (TimeVstart) is elevated -- e.g., -50 as shown in the figure.

# Simplified AKs channel

The `AKsParams` provides a much simpler, stateless version of the AK channel that is useful to just get a high-level cutoff to membrane potentials in the dendrites, e.g., as might otherwise emerge from the VGCC channel.

![Gaks from V](fig_aks_vs_ak.png?raw=true "Gak for simplified AK channel vs. Gak from full version")

This function is:
```Go
	if vbio > -37 { // flat response above cutoff -- real function goes back down..
		vbio = -37
	}
	return 0.076 / (1.0 + math32.FastExp(-0.075*(vbio+2)))
```



