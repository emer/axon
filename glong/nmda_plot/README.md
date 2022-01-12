# nmda

This plots the NMDA current function from Sanders et al, 2013 and Brunel & Wang (2001), which is most widely used active maintenance model.

See also: https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

Also used in: WeiWangWang12, FurmanWang08, NassarHelmersFrank18

Voltage dependence function is:

```Go
1 / (1 + 0.28*exp(-0.062 * vm))
```

![g_NMDA from V](fig_sandersetal_2013.png?raw=true "NMDA voltage gating conductance according to parameters from Sanders et al, 2013")



