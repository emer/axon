# PCore BG Model: Pallidal Core model of BG

The core of this model is the Globus Pallidus (external segment), GPe, which plays a central role in integrating Go and NoGo signals from the striatum, in contrast to the standard, "classical" framework which focuses on the GPi or SNr as the primary locus of integration.

There are two recently-identified revisions to the standard circuitry diagram that drive this model (Suryanarayana et al, 2019; others):

* A distinction between outer (GPeOut) and inner (GPeIn) layers of the GPe, with both D1-dominant (Go) and D2-dominant (NoGo / No) projections into the GPe (the classical "direct" vs. "indirect" terminology thus not being quite as applicable).

* And a third, even more distinct arkypallidal (GPeTA) GPe layer.

Thus, the GPe circuitry is clearly capable of doing significant additional computation within this more elaborate circuitry, and it has also long been recognized as strongly interconnected with the subthalamic nucleus (STN), providing a critical additional dynamic.  This model provides a clear computational function for the GPe / STN complex within the larger BG system, with the following properties:

* GPeIn is the core of the core: it integrates Go and No striatal signals in consistent terms, with direct No inhibitory inputs, and indirect Go net excitatory inputs, inverted by way of double-inhibition through the GPeOut.  By having both of these signals converging on single neurons, the GPeIn can directly weigh the balance for and against a potential action.  Thus, in many ways, GPeIn is like the GPi / SNr of the classical model, except with the sign reversed (i.e., it is more active for a more net-Go balance).

* The GPeIn then projects inhibition to the GPeTA, which in turn drives the strongest source of broad, diffuse inhibition to the striatum: this provides the long-sought winner-take-all (WTA) action selection dynamic in the BG circuitry, by broadcasting back an effective inhibitory threshold that only the most strongly-activated striatal neurons can withstand.  In addition, the GPeTA sends a weaker inhibitory projection into the striatum, and having both of these is essential to prevent strong oscillatory dynamics.

* The GPi integrates the direct Go inhibition from the striatum, and the integrated Go vs. No balance from the GPeIn, which have the same sign and contribute synergistically to the GPi's release of inhibition on the thalamus, as in the standard model.  In our model, the integrated GPeIn input is stronger than the one from striatum Go pathway.

* The STN in our model plays two distinct functional roles, across two subtypes of neurons, defined in terms of differential connectivity patterns, which are known have the relevant diversity of connectivity (cites):

    + The STNp (pausing) neurons receive strong excitatory connections from the frontal cortex, and project to GPe (In and Out), while also receiving GPe inhibition.  The frontal input triggers a rapid inhibitory oscillation through the GPe circuits, which project strong inhibition in return to the STN's strong excitation.  This dynamic then produces a sustained inhibition of STNp firing, due to calcium-gated potassium channels (KCA, cites), which has been well documented in a range of preparations including, critically, awake-behaving animals (*in activo*) (Fujimoto & Kita, 1993; Magill et al, 2004).  The net effect of this burst / pause is to excite the GPe and then open up a window of time when it can then be free from driving STN inputs, to integrate the balance of Go vs. No pathways.  We see in our model that this produces a nice graded settling process over time within the pallidal core, such that the overall gating output reaction time (RT) reflects the degree of Go vs. No conflict: high conflict cases are significantly slower than unopposed Go cases, whilst the strength of Go overall also modulates RT (stronger Go = faster RT).

    + The STNs (stopping) neurons receive weaker excitatory inputs from frontal cortex, and project more to the GPi / SNr compared to GPe.  These neurons do not trigger the disinactivation required for strong levels of KCa channel opening, and instead experience a more gradual activation of KCa due to Ca influx over a more prolonged wave of activation.  Functionally, they are critical for stopping premature expression of the BG output through the GPi, before the core GPe integration has had a chance to unfold.  In this way, the STNs population functions like the more traditional view of STN overall, as a kind of global NoGo signal to prevent BG firing from being too "impulsive" (cites).  However, in this model, it is the GPe Go vs. No balancing that provides the more "considered", slower calculation, whereas the frontal cortex is assumed to play this role in the standard account.
    
* Both STN pathways recover from the KCa inhibition while inactivated, and their resumed activity excites the GP areas, terminating the window when the BG can functionally gate.  Furthermore, it is important for the STN neurons to experience a period with no driving frontal inputs, to reset the KCa channels to be able to support the STNp burst / pausing dynamics.  This has the desirable functional consequence of preventing any sustained frontal patterns from driving repeated BG gating of the same pathways again and again.  This provides a beneficial bias to keep the flow of action and cognition in a constant state of flux.

In summary, the GPe is the core integrator of the BG circuit, while the STN orchestrates the timing, opening the window for integration and preventing premature output.  The striatum is thus free to play its traditional role of learning to identify critical features supporting Go vs. No action weights, under the influence of phasic dopamine signals.  Because it is by far the largest BG structure, it is not well suited for WTA competition among alternative possible courses of action, or for integration between Go and No, which instead are much better supported by the compact and well-connected GPe core.

# Dynamics

![Net](figs/fig_pcore_netdyn_cyc.png?raw=true "Network gating dynamics")

The above figure shows the key stages unfolding over cycles within a standard Alpha cycle of 100 cycles.  Some of the time constants have been sped up to ensure everything occurs within this timeframe -- it may take longer in the real system.

* Cycle 1: tonic activation of GP and STN, just at onset of cortical inputs.  GPeTA is largely inhibited by GPeIn, consistent with neural recording.

* Cycle 7: STNp peaks at above the .9 threshold that KCa triggers pausing, while STNs rises more slowly, driving a slower accumulation of KCa.  STN has activated the GPe layers.

* Cycle 23: Offset of STNp enables GPe layers to settle into integration of MtxGo and MtxNo activations.  GPeTA sends broad inhibition to Matrix, such that only strongest are able to stay active.

* Cycle 51: STNs succumbs to slow accumulation of KCa, allowing full inhibition of GPi output from remaining MtxGo and GPeIn activity, disinhibiting the ventral thalamus (VThal), which then has a excitatory loop through PFCo output layers.

* Cycle 91: STN layers regain activation, resetting the circuit.  If PFC inputs do not turn off, the system will not re-gate, because the KCa channels are not fully reset.

![Net](figs/fig_pcore_rt_all_default.png?raw=true "Reaction time behavior")

The above figure shows the reaction time (cycles) to activate the thalamus above a firing threshold of .5, for a full sweep of ACC positive and negative values, which preferentially activate Go and No respectively.  The positive values are incremented by .1 in an outer loop, while the negative values are incremented by .1 within each level of positive.  Thus, you can see that as NoGo gets stronger, it competes against Go, causing an increase in reaction time, followed by a failure to gate at all.

# Electrophysiological data

Recent data: : majority of STN units show decreasing ramp prior to go cue, then small subset show brief phasic burst at Go then brief inhib window then *strong sustained activity* during / after movement.  This sustained activity will turn off gating window -- gating in PFCd can presumably do that and provide the final termination of the gating event.

![RT](figs/fig_fujimoto_kita_93_fig3_stn.png?raw=true "Fujimoto & Kita, 1993, figure 3")

![RT](figs/fig_mirzaei_et_al_2017_stn.png?raw=true "STN recordings from Mizraei et al, 2017")

Data from Dodson et al, 2015 and Mirzaei et al, 2017 shows brief increase then dips or increases in activity in GPe prototypical neurons, with also very consistent data about brief burst then shutoff in TA neurons.  So both outer and inner GPe prototypical neuron profiles can be found.

![RT](figs/fig_dodson_et_al_2015_gpe.png?raw=true "GPe recordings around movement from Dodson et al, 2015")

## STNp pause mechanisms: SKCa channels

The small-conductance calcium-activated potassium channel (SKCa) is widely distributed throughout the brain, and in general plays a role in medium-term after-hyper-polarization (mAHP) (Dwivedi & Bhalla, 2021), including most dramatically the full *pausing* of neural firing as observed in the STNp neurons.  The basic mechanism is straightforward: Ca++ influx from VGCC channels, opened via spiking, then activates SKCa channels in a briefly delayed manner (activation time constant of 5-15 msec), and the combined trace of Ca and relatively slow SKCa deactivation results in a window where the increased K+ conductance (leak) can prevent the cell from firing.  If insufficiently intense initial spiking occurs, then the resulting slow accumulation of Ca partially activates SKCa, slowing firing but not fully pausing it.  Thus, the STNp implements a critical switch between opening the BG gating window for strong, novel inputs, vs. keeping it closed for weak, familiar ones.

These SKCa channels are not widely modeled, and existing models from **FujitaFukaiKitano12** (based on **GunayEdgertonJaeger08**), and **GilliesWillshaw06** have different implementations that diverge from some of the basic literature cited below.  Thus, we use a simple model based on the Hill activation curve and separate activation vs. deactivation time constants.

* **XiaFaklerRivardEtAl98**: "Time constants for activation and deactivation, determined from mono-exponential fits, were 5.8, 6.3 and 12.9 ms for activation and 21.7, 29.6 and 38.1ms for deactivation of SK1, SK2 and SK3, respectively."  "... Ca2+ concentrations required for half-maximal activation (K_0.5) of 0.3 uM and a Hill coefficient of ~4 (Fig. 1a)."

* **AdelmanMaylieSah12**: "Fast application of saturating Ca2+ (10 μM) to inside-out patches shows that SK channels have activation time constants of 5–15 ms and deactivation time constants of ∼50 ms (Xia et al, 1998)."  Mediates mAHP, which decays over "several hundred msec".

* **DwivediBhalla21** "SK channels are voltage insensitive and are activated solely by an increase of 0.5–1 μM in intracellular calcium (Ca2+) levels (Blatz and Magleby, 1986; Köhler et al., 1996; Sah, 1996; Hirschberg et al., 1999). An individual channel has a conductance of 10 pS and achieves its half activation at an intracellular calcium level of approximately 0.6 μM (Hirschberg et al., 1999). The time constant of channel activation is 5–15 ms, and the deactivation time is 30 ms (Xia et al., 1998; Oliver et al., 2000)."  in PD: "while the symptoms were aggravated when the channels were blocked in the STN (Mourre et al., 2017)."


# Learning logic

For MSNs, the standard 3-factor matrix learning (with D1/D2 sign reversal) works well here, *without any extra need to propagate the gating signal from GPi back up to Striatum* -- the GPeTA projection inhibits the Matrix neurons naturally.

* dwt = da * rn.Act * sn.Act

However, there still is the perennial problem of temporal delay between gating action and subsequent reward. We use the trace algorithm here, but with one new wrinkle.  The cholinergic interneurons (CINs, aka TANs = tonically active neurons) are ideally suited to provide a "learn now" signal, by firing in proportion to the non-discounted, positive rectified US or CS value (i.e., whenever any kind of reward or punishment signal arrives, or is indicated by a CS). 

Thus, on every trial, a gating trace signal is accumulated:

* Tr += rn.Act * sn.Act

and when the CINs fire, this trace is then applied in proportion to the current DA value, and the trace is reset:

* DWt += Tr * DA
* Tr = 0

# Other models

* **SuryanarayanaHellgrenKotaleskiGrillnerEtAl19** -- focuses mainly on WTA dynamics and doesn't address in conceptual terms the dynamic unfolding.

# References

Bogacz, R., Moraud, E. M., Abdi, A., Magill, P. J., & Baufreton, J. (2016). Properties of neurons in external globus pallidus can support optimal action selection. PLoS computational biology, 12(7).
    
* lots of discussion, good data, but not same model for sure.

Dodson PD, Larvin JT, Duffell JM, Garas FN, Doig NM, Kessaris N, Duguid IC, Bogacz R, Butt SJ, Magill PJ. Distinct Developmental Origins Manifest in the Specialized Encoding of Movement by Adult Neurons of the External Globus Pallidus. Neuron. 2015; 86:501–513. [PubMed: 25843402]

Dunovan, K., Lynch, B., Molesworth, T., & Verstynen, T. (2015). Competing basal ganglia pathways determine the difference between stopping and deciding not to go. Elife, 4, e08723.

* good ref for Hanks -- DDM etc

Hegeman, D. J., Hong, E. S., Hernández, V. M., & Chan, C. S. (2016). The external globus pallidus: progress and perspectives. European Journal of Neuroscience, 43(10), 1239-1265.

Mirzaei, A., Kumar, A., Leventhal, D., Mallet, N., Aertsen, A., Berke, J., & Schmidt, R. (2017). Sensorimotor processing in the basal ganglia leads to transient beta oscillations during behavior. Journal of Neuroscience, 37(46), 11220-11232.

* great data on STN etc

Suryanarayana, S. M., Hellgren Kotaleski, J., Grillner, S., & Gurney, K. N. (2019). Roles for globus pallidus externa revealed in a computational model of action selection in the basal ganglia. Neural Networks, 109, 113–136. https://doi.org/10.1016/j.neunet.2018.10.003

* key modeling paper with lots of refs
    
Wei, W., & Wang, X. J. (2016). Inhibitory control in the cortico-basal ganglia-thalamocortical loop: complex regulation and interplay with memory and decision processes. Neuron, 92(5), 1093-1105.

* simple model of SSRT -- good point of comparison
 
