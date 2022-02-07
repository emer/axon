# nmda

This plots the NMDA current function from Sanders et al, 2013 and Brunel & Wang (2001) (BW01), which is the most widely used active maintenance model.

See also: https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

Also used in: WeiWangWang12, FurmanWang08, NassarHelmersFrank18

Voltage dependence function based on Mg ion blocking is:

```Go
1 / (1 + (C/3.57)*exp(-0.062 * vm))
```

Jahr & Stevens (1990) originally derived this equation, using a Mg+ concentration C of 1 mM, so that factor = 0.28, used in BW01.  Urakubo et al. (2008) used a concentration of 1.5 mM, citing Froemke & Dan (2002).  Various other sources (Vink & Nechifor) indicate physiological ranges being around 1.2 - 1.4.  

![g_NMDA from V](fig_sandersetal_2013.png?raw=true "NMDA voltage gating conductance according to parameters from Sanders et al, 2013")

In addition to this v-dependent Mg factor, conductance depends on presynaptic glutamate release, which in the basic BW01 model just increments with spikes and decays with a time constant of about 100 msec.

The Urakubo et al (2008) model introduces allosteric dynamics, based on data from Froemke & Dan (2002), which we can capture simply using an inhibitory factor that increments with each spike and decays with a 100 msec time constant.  However, their model has an effective Glu-binding decay time of only 30 msec.


# References

* Jahr, C. E., & Stevens, C. F. (1990). A quantitative description of NMDA receptor-channel kinetic behavior. Journal of Neuroscience, 10(6), 1830–1837. https://doi.org/10.1523/JNEUROSCI.10-06-01830.1990

* Sanders, H., Berends, M., Major, G., Goldman, M. S., & Lisman, J. E. (2013). NMDA and GABAB (KIR) Conductances: The "Perfect Couple" for Bistability. Journal of Neuroscience, 33(2), 424–429. https://doi.org/10.1523/JNEUROSCI.1854-12.2013

* Vink, R., & Nechifor, M. (Eds.). (2011). Magnesium in the Central Nervous System. University of Adelaide Press. https://doi.org/10.1017/UPO9780987073051

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. *The Journal of Neuroscience, 28(13),* 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598

