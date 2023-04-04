# sKCa: Voltage-gated Calcium Channels

This plots the sKCa current function, which is describes the small-conductance calcium-activated potassium channel, using the equations described in Fujita et al (2012) based on Gunay et al (2008), (also Muddapu & Chakravarthy, 2021).  There is a gating factor M that depends on the Ca concentration, modeled using an X / (X + C50) form Hill equation:

```Go
M_hill(ca) = ca^h / (ca^h + c50^h)
```

This function is .5 when `ca == c50`, and the `h` default power (`Hill` param) of 4 makes it a sharply nonlinear function.

```Go
M(vm) = 1 / (1 + exp(-(vm + 37)))     // Tau = 3.6 msec
```

```Go
H(vm) = 1 / (1 + exp((vm + 41) * 2))  // Tau = 29 msec
```

```Go
sKCa(vm) = M^3 H * G(vm)
```

![M and H gating from V](fig_sKCa_m_h_from_v.png?raw=true "M and H gating factors as a function of V (biological units)")

The intersection of M and H produces a very narrow membrane potential window centered around -40 mV where the gates are both activated, with the fast updating (3.6 msec tau) and cubic dependence on M essentially shutting off the current very quickly when the action potential goes away.  The longer H time constant provides a slow adaptation-like dynamic that is sensitive to frequency (more adaptation at higher frequencies).

![M and H over time](fig_sKCa_time_plot.png?raw=true "M and H gating factors developing over time in response to simulated spiking potentials")

The `Time Run` plot (above) shows these dynamics playing out with a sequence of spiking.

Functionally, the narrow voltage window restricts the significance of this channel to the peri-spiking time window, where the Ca++ influx provides a bit of extra sustain on the trailing edge of the spike.  More importantly, the Ca++ provides a postsynaptic-only spiking signal for learning.

