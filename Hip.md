# Hippocampus

Package hip provides special hippocampus algorithms for implementing the Theta-phase hippocampus model described in [Zheng et al., 2022](#references) and [Ketz, Morkonda, & O'Reilly (2013)](#references).

Files: hip_{[net.go](axon/hip_net.go), [prjns.go](axon/hip_prjns.go)}

Timing of ThetaPhase dynamics -- based on quarter structure:

* **Q1:**   ECin -> CA1 -> ECout (CA3 -> CA1 off)  : ActQ1 = minus phase for auto-encoder
* **Q2,3:** CA3 -> CA1 -> ECout  (ECin -> CA1 off) : ActM = minus phase for recall
* **Q4:**   ECin -> CA1, ECin -> ECout (CA3 -> CA1 off, ECin -> CA1 on): ActP = plus phase for everything

```
[  q1      ][  q2  q3  ][     q4     ]
[ ------ minus ------- ][ -- plus -- ]
[   auto-  ][ recall-  ][ -- plus -- ]

  DG -> CA3 -> CA1
 /    /      /    \
[----ECin---] -> [ ECout ]

minus phase: ECout unclamped, driven by CA1
auto-   CA3 -> CA1 = 0, ECin -> CA1 = 1
recall- CA3 -> CA1 = 1, ECin -> CA1 = 0

plus phase: ECin -> ECout auto clamped
CA3 -> CA1 = 0, ECin -> CA1 = 1
(same as auto- -- training signal for CA3 -> CA1 is what EC would produce!
```

* `ActQ1` = auto encoder minus phase state (in both CA1 and ECout
        used in EcCa1Prjn as minus phase relative to ActP plus phase in CHL)
* `ActM` = recall minus phase (normal minus phase dynamics for CA3 recall learning)
* `ActP` = plus (serves as plus phase for both auto and recall)

learning just happens at end of trial as usual, but encoder projections use the ActQ1, ActM, ActP variables to learn on the right signals

# References

Papers also avail at https://ccnlab.org/pubs

* Ketz, N., Morkonda, S. G., & O’Reilly, R. C. (2013). Theta coordinated error-driven learning in the hippocampus. PLoS Computational Biology, 9, e1003067. http://www.ncbi.nlm.nih.gov/pubmed/23762019  [PDF](https://ccnlab.org/papers/KetzMorkondaOReilly13.pdf)

* Zheng, Y., Liu, X. L., Nishiyama, S., Ranganath, C., & O’Reilly, R. C. (2022). Correcting the hebbian mistake: Toward a fully error-driven hippocampus. PLOS Computational Biology, 18(10), e1010589. https://doi.org/10.1371/journal.pcbi.1010589 [PDF](https://ccnlab.org/papers/ZhengLiuNishiyamaEtAl22.pdf)


