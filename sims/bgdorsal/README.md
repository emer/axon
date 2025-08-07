# BG Dorsal

This is a test of the PCore model of basal ganglia (BG) function, in the **dorsolateral striatum** (DLS) and associated BG circuits. See also the [bgventral](../bgventral) for the simpler ventromedial striatum (VMS) model.

Full documentation is at [CompCogNeuro](https://CompCogNeuro.org/bg-dorsal-simulation) and the [basal ganglia](https://CompCogNeuro.org/basal-ganglia) page there. 

# Results

The params have been relatively thoroughly "tweaked" at this point: see `paramtweak.go` for tweaking code. Overall the model is remarkably robust to +/- 20% changes from default params.

The learned weights to the BG clearly show that it is disinhibiting the appropriate action at each step in the sequence.

## Combinatorics

| Act ^ Len | N       | Notes          |
|-----------|---------|----------------|
| 10^3      |   1,000 | acts = harder  |
| 6^4       |   1,296 | easy           |
| 5^5       |   3,125 | easy, std test |
| 6^5       |   7,776 | still quick    |
| 7^5       |  16,807 | some fail      |
| 8^5       |  32,768 | ?              |
| 9^5       |  59,049 | ?              |
| 10^5      | 100,000 | ?              |
| 6^6       |  46,656 | ~50% learn     |
| 7^6       | 117,649 | ?              |

## Aug 6, 2025

| Act ^ Len | Fail | Mean  | SDev | Min  | Q1  | Median | Q3   | Max   | Job no |
|-----------|------|-------|------|------|-----|--------|------|-------|--------| 
| 5^5       |    0 |  7.6  |  5.6 | 2    | 4   | 5      | 10   | 28    | 504    |
| 6^5       |    0 | 13.8  | 13.6 | 3    | 5   | 8.5    | 15   | 59    | 510    |
| 7^5       |    8 | 43.7  | 70   | 3    | 6   | 12.5   | 30.7 | 200   | 519    |
| 10^3      |      |       |      |      |     |        |      |       |        |


# TODO:

* STN pool-based instead of shared full

* Set number of cycles per trial in terms of BG motor gating timing: constant offset from onset of VM gating timing, with a cutoff for "nothing happening" trials.
    * Attempted an impl but is difficult with CaBins -- tried shifting bins but awk..

* "CL" not beneficial (implemented as direct MotorBS -> Matrix pathways): rel weight of 0.002 is OK but starts to actually impair above that.  Likely that a functional cerebellum is needed to make this useful.  Also, investigate other modulatory inputs to CL that might alter its signal.  Key ref for diffs between CL and PF: LaceyBolamMagill07: C. J. Lacey, J. P. Bolam, P. J. Magill, Novel and distinct operational principles of intralaminar thalamic neurons and their striatal pathways. J. Neurosci. 27, 4374–4384 (2007).

* Learning in the other parts of the pcore circuit -- might help auto-adjust the parameters.  Need to figure out what logical learning rules would look like.



