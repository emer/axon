# Urakubo = Urakubo et al, 2008

This model re-implements the highly biophysically realistic model of spike-timing dependent plasticity from:

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. The Journal of Neuroscience, 28(13), 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598 | [Main PDF](https://github.com/emer/axon/blob/master/examples/urakubo/papers/UrakuboHondaFroemkeEtAl08.pdf) | [Supplemental PDF](https://github.com/emer/axon/blob/master/examples/urakubo/papers/UrakuboHondaFroemkeEtAl08_suppl.pdf) | [Model Details](https://github.com/emer/axon/blob/master/examples/urakubo/papers/UrakuboHondaFroemkeEtAl08_model_sup.pdf) (last one has all the key details but a few typos and the code is needed to get all the details right).

This model captures the complex postsynaptic chemical cascades that result in the changing levels of AMPA receptor numbers in the postsynaptic density (PSD) of a spine, as a function of the changing levels of calcium (Ca2+) entering the spine through NMDA and VGCC (voltage-gated calcium) channels.  The model mechanisms are strongly constrained bottom-up by the known kinetics of these chemical processes, and amazingly it captures many experimental results in the LTP / LTD experimental literature.  Thus, it provides an extremely valuable resource for understanding how plasticity works and how realistic but much simpler models can be constructed, which are actually usable in larger-scale simulations.

The complexity of this model of a single synaptic spine involves 116 different variables updated according to interacting differential equations requiring a time constant of 5e-5 (20 steps per the typical msec step used in axon).  In previous work with the original Genesis implementation of this model (see [CCN textbook](https://CompCogNeuro.org)), we were able to capture the net effect of this model on synaptic changes, at a rate-code level using random poisson spike trains, using the simple "checkmark" linearized version of the BCM learning rule (called "XCAL" for historical reasons), with an `r` value of around .9 (81% of the variance) -- see figure below.  However, with the actual spiking present in axon, there is an important remaining question as to whether a more realistic (yet still simplified) learning mechanism might be able to take advantage of some aspects of the detailed spiking patterns to do "better" somehow than the simplified rate code model.

The original Genesis model is a good object lesson in how a generic modeling framework can obscure what is actually being computed.  The model shows up as a complex soup of crossing lines in the Genesis GUI, and the code is a massive dump of inscrutable numbers, with key bits of actual computation carefully hidden across many different bits of code, with seemingly endless flags and conditions making it very difficult to determine what code path is actually in effect.  In addition, Genesis is ancient software that requires an old gcc compiler to build it, and it only runs under X windows.

By contrast, the present model is purpose-built in compact, well-documented highly readable Go code, using a few very simple types in the [emergent chem](https://github.com/emer/emergent/tree/master/chem) package, providing the basic `React` and `Enz` mechanisms used for simulating the chemical reactions.

# Model Summary

![Urakubo et al., (2008) Model](figs/fig_urakubo_et_al_08_bcm_sum.png?raw=true "Summary of Urakubo et al., 2008 model, and how it simulates many STDP data points, and behavior for more realistic spike trains can be summarized by a simple linearized BCM-like function")

The above figure summarizes the model and some of the STDP data that it simulates, along with the derivation of the XCAL checkmark linearized BCM-like learning function from more realistic spiking inputs.  The detailed CaMKII portion of the model is directly from Dupont et al (2003) with some parameter modifications, while the compartmental spiking model is from Migliore et al. (1999) and Poirazi et al. (2003).  Other signaling cascades were adapted from Bhalla and Iyengar (1999) and earlier cerebellar LTD models by this group: Kuroda et al. (2001); Doi et al. (2005).

In our version, we omitted the full compartmental model in favor of the basic Axon AdEx-style model which just summarizes the Vm curve resulting from a spike.  However, we did have to adjust a few parameters to match the Vm curve from the Genesis model (though these do not appear to make much difference in the end).

We also added VGCC channels (not otherwise used in axon -- verified that they do make a difference for the basic STDP results), and used the allosteric NMDA dynamics from the original Urakubo model instead of the simpler NMDA channel used in axon.  See the `NMDAAxon` option.

# Basic Usage

Toolbar at the top does actions (mouse over and hold to see tooltips), e.g.:
* `Init` re-initializes with any params updates
* `Run` runs current `Stim` program

Params list at left side control lots of things (see tooltips), e.g.:
* `Stim` determines the stimulation program -- see `stims.go` for code -- figures below show some examples.
* `ISISec` = spacing between individual stimulation patterns
* `NReps` = number of repetitions of patterns -- more = bigger effects on weights
* `FinalSecs` = number of seconds to allow AMPAR trafficking to settle out -- has significant effects on final results, particularly on showing LTD effects which emerge over time.  50 secs is long enough to be close to asymptotic -- 20 vs. 50 still shows some diffs but 50 vs. 100 is pretty close.
* Other params used in different cases -- see examples below -- and below that are overall parameters of the neuron etc.

Tab bar selects different plots:
* `DWtPlot` shows final delta-weights (change in synaptic weight), i.e., `Trp_AMPAR` = trapped AMPA receptors, as a proportion of `InitWt` which is baseline value -- only updated for the `Sweep` Stims.
* `PhaseDWtPlot` shows DWt for `ThetaErr` case, which manipulates minus and plus phase activity states according to the Contrastive Hebbian Learning (CHL) algorithm.
* `Msec*Plot` shows all the model state values evolving over time at different temporal scales (100 = 1 point for every 100 msec, etc)

See https://github.com/emer/axon/blob/master/examples/urakubo/results for plots and tab-separated-value (TSV) data for various cases, some of which are summarized below.

# Classical STDP Replication

The classic Bi & Poo STDP curve emerges reliably with e.g., 100 reps with 50 secs final "burn in" of the weights -- maximum burn-in occurs after about 100 secs and after then things can start to decay. The curve gets stronger with more reps as well.  The X axis is Tpost - Tpre, so the "causal" pre-post ordering is positive numbers on the right.

![STDPSweep](results/fig_urakubo22_stdp_100rep_final50.png?raw=true "Classic STDP with standard params -- X axis is Tpost - Tpre")

* `Stim = STDPSweep`
* `NReps = 100`
* `FinalSecs = 50`

# STDP in Packets

What happens in a slightly more "realistic" scenario where pre-post spikes occur at more naturalistic overall frequencies? This was tested with the STDPPacketSweep inputs: 200 msec of 25 or 50 hz spiking, with fixed pre-post time offsets.  This case represents a kind of "best case" for whether there might be some kind of STDP-like signal left in more realistic spike trains -- it is still highly unlikely that in the noisy environment of the brain, you would ever see such reliable pre-post timing across a sequence of spikes like this.

Interestingly, this shows the same qualitative effects but in a much weaker form -- this means that there may be *some* residual STDP effect in the overall learning rule -- can't quite conclude that yet because this model does not capture CHL.

![STDPPacketSweep](results/fig_urakubo22_stdpacket_25hz_isi1_10rep_final50.png?raw=true "STDP in theta-window packets of pre-post activity")

* `Stim = STDPPacketSweep`
* `NReps = 10`
* `FinalSecs = 50`
* `SendHz = 25 or 50`
* `DurMsec = 200`

# XCAL / BCM checkmark learning rule

The data used to derive the XCAL check-mark linearized version of the BCM curve manipulated send and recv Hz and also overall duration -- the main plots are for a given sending Hz, varying recv Hz (X axis) and duration.  These did NOT use poisson random spiking but rather ensured that pre and post fired 180 degrees out phase, so as to effectively negate any timing issues (equally pre or post).  Here are some plots for different levels of sending activity:

![XCAL Original](figs/fig_urakubo_xcal_3d_fits.png?raw=true "Original XCAL data from Genesis model (black underlying grid), with fitted checkmark function overlaid in red on top")

Here is the replication in current model -- viewed as stacked slices through the Z depth axis of the central panel of the above 3D plots -- the X axis is Recv Hz, different lines as shown in legend are  different durations of overall firing.

Sender = 100Hz:

![OppPhaseDur 100 SendHz](results/fig_urakubo22_opphasedur_100shz_final50_1rep.png?raw=true "Opposite phase firing (180 degrees out of phase at start) as function of Recv Hz on X axis and different durations (lines)")

Sender = 50Hz:

![OppPhaseDur 50 SendHz](results/fig_urakubo22_opphasedur_50shz_final50_1rep.png?raw=true "Opposite phase firing (180 degrees out of phase at start) as function of Recv Hz on X axis and different durations (lines)")

* `Stim = OpPhaseDurSweep`
* `NReps = 1` 
* `FinalSecs = 50`
* `SendHz = 50 or 100`

# Random Poisson Firing

This case shows what bad memory had consolidated as the basis for the original XCAL checkmark learning rule -- random poisson firing of send and recv neurons over a packet of a given duration. This produces a more complex curve, e.g., for 200msec activity packets (theta cycle), as a function of recv firing rate (X axis) and sending firing rate (legend / different lines):

![Poisson 200msec](results/fig_urakubo22_poisson_200msec_isi1_final50_10rep.png?raw=true "Poisson random 200 msec packets of pre - post activity at different rates -- X is Recv Hz, Send Hz are different lines as shown in legend")

* `Stim = PoissonHzSweep`
* `NReps = 10` 
* `FinalSecs = 50`
* `DurMsec = 200`

# CHL Error-driven Learning

The `ThetaErr` Stim runs the current Zito lab empirical experiment on the model, testing for plus-phase vs. minus phase differences in firing rate over a 200msec window (theta frequency), with the minus phase the first 100 msec, plus phase the second 200 msec.

![CHL Minus-Plus](figs/fig_chl_minus_plus_dwt_on_off.png?raw=true "Qualitative picture of the CHL error-driven learning function over time, as a minus-phase prediction followed by a plus-phase outcome.  On the Off-Off and On-On diagonal, the weights do not change, and on the off-diagonal terms you get either positive weight change (LTP, for a miss as Off->On) or negative weight change (LTD for an error as On -> Off)")

The X axis on the plot is CHL error signal for each condition: 

* CHL = X+Y+ - X-Y- 

With X = normalized frequency for sender (Hz/100), Y = norm recv frequency.  First term is hebbian co-product in the plus phase (correct answer, outcome, target), second is for the minus phase (guess, prediction, actual model output).

The Y axis is the DWt result of the model. If the model was perfectly capturing the CHL learning rule, the points would all be along a positive diagonal line (i.e., the r = 1 line).  Clearly, the original Urakubo model does not accurately capture the CHL function, although some subset of points are not too far off.  It is particularly inaccurate on the negative error cases, which require strong LTD, despite relatively high levels of activity.

![ThetaErr RSPk 200ms](results/old/fig_urakubo22_thetaerr_rspk_200ms_final100_10rep.png?raw=true "Theta (200 msec) packets of pre - post activity at different rates -- X is Recv Hz, Send Hz are different lines as shown in legend -- recv unit is driven by punctate spikes")

* `Stim = ThetaErr`
* `ISISec = 1`
* `NReps = 10` 
* `FinalSecs = 50`
* `DurMsec = 200`
* `RGClamp = false or true`

The next plot shows the case for `RGClamp = true`, where the recv unit is current-clamped to drive its spiking, instead of phasically driving individual spikes at a prescribed regular interval according to the target Hz.

![ThetaErr RGe 200ms](results/fig_urakubo22_thetaerr_rge_200ms_final50_10rep.png?raw=true "Theta (200 msec) packets of pre - post activity at different rates -- X is Recv Hz, Send Hz are different lines as shown in legend -- recv unit is driven by current clamp")

# Basic model behavior

You can see single stimulus events in the `MsecPlot` at full temporal resolution by running the `STDP` or `Poisson` Stim cases with `Nreps = 1, ISISec = 0, FinalSecs = 0`, so you just see the one trace.  This can be used for exploring parameter differences, seeing the Ca-driven dynamics etc.

# CaMKII drives everything?

![STDP Chem](results/fig_urakubo22_stdp_100rep_final1_chem.png?raw=true "Signaling state at end of std STDP induction")

The above figure suggests that the final dWt is strongly related to CaMKII levels. 

TODO: measure peak Ca too and plot that!  have to grab it within each spike window.


# LFS 900 @ 1Hz = LTD?

One standard way of generating LTD is low-frequency stimulation (LFS) for a large number of repetitions, e.g., 900 @ 1Hz (Dudek & Bear, 1992).  However, this does not produce the appropriate results in the basic Urakubo model:

![LFS 900 1hz](results/fig_urakubo22_lfs900_1hz_final20_ge.1.png?raw=true "LFS 900 reps at 1Hz, should cause LTD but does not")

* `Stim = STDP`
* `NReps = 900` 
* `FinalSecs = 1`
* `ISISec = 1`
* `DeltaT = 0` // key!
* `GeStim = .1 or .2` // .1 produces 2-5 mV EPSP, .2 = 10 mV

# DAPK1 and Competitive NMDA GluN2B binding of CaMKII for LTD

There is recent evidence for a competitive interaction between DAPK1 and CaMKII binding at the GluN2B subunit of the NMDA receptor (Goodell et al., 2017), which is important for the LFS LTD, that the model does not currently capture.  Thus, it is important to add this dynamic, tuning it to capture the LFS LTD effects, and then testing whether this is sufficient to capture the CHL error-driven learning dynamic of primary interest.

## CaMKII binding to GluN2B

![CaMKII GluNR2B Binding](figs/fig_urakubo_camkii_nums.png?raw=true "CaMKII Ca/CaM binding and autophosphorylation, showing the NR2B binding modulation at reactions 6 and 9, in terms of reduced backward rate constant 0.02 instead of 400, 0.001 and 1, along with 8 going forward at 9.  Adapted from SI3 in Urakubo modeling suppl.")

The Urakubo model does not directly include the role of CaMKII binding to the GluN2B subunit of the NMDAR (Bayer et al, 2001; Coultrap & Bayer, 2012; Barcomb et al, 2016), but does have indirect accounting for it.  There are two primary effects:

* Concentrating CaMKII at the PSD on the NMDA receptors depends on GluN2B binding.  Only the PSD-localized CaMKII can activate and stabilize AMPAR in the PSD, so this is the primary determinant of LTP effects of CaMKII.  One site of GluN2B binding is dependent on the CaMKII T286 autophosphorylation (CaMKIIP in the model), which is in turn a complex function of Ca-CaM binding, driven by Ca influx, and another is dependent on Ca-CaM binding (Bayer et al, 2001).  This is reflected in the model by the asymmetric diffusion constants for CaMKIIP, which is 10 times larger for Cyt -> PSD than the PSD -> Cyt direction (6 vs. 0.6).

* Altering the CaM binding constants: "modified the binding constants between bare CaM and the inactive subunit in consideration of experimentally observed basal activity (Fukunaga et al., 1993; Kawaguchi and Hirano, 2002). This modification also implicitly represents CaMKII by binding to NMDARs (Bayer et al., 2001; Leonard et al., 2002)".  In the adapted Figure SI3 shown above, it is reactions 4 and 9: 4 is CaM + plain CaMKII (0.0004 Kf, 1 Kb), 9 is CaM + CaMKIIP (8 Kf, 0.001 Kb).

Thus, to incorporate DAPK1, we first need to simulate the GluN2B binding dynamics explicitly.

Shen & Meyer (1999), Bayer et al (2001) and Leonard et al (2002) have the relevant details on GluN2B binding.  The key points are:

* There are two N2B binding sites: a -C and -P, and the -C is determined by Ca/CaM activation of CaMKII (only), while -P depends on T286 autophosphorylation.  In general, the dynamics are that Ca/CaM activation causes the initial binding, and T286 keeps it around longer after Ca/CaM is gone. 

* >In fact, this NR2B domain contains two sites with different modes of regulated CaMKII binding (Fig. 1), a Ca2+/CaM-regulated site within residues 1,120 -- 1,482 of NR2B (NR2B-C, later determined to be 1303) and a phosphorylation-regulated site within residues 839 -- 1,120 (NR2B-P). 

* >During washes with Ca2+ but without CaM, however, CaM remained bound to the complex. Thus, interaction with NR2B-C seems to increase the CaM affinity of CaMKII, similar to CaM trapping by the autophosphorylated kinase, as CaM dissociates from unphosphorylated kinase that is not complexed with NR2B-C with an off rate of approximately 2 s-1 (ref. 11)

* >Immunoblotting with an antibody sensitive to phosphorylated T286 showed that the autonomously active NR2B-C-bound enzyme is not phosphorylated at T286.

Also, the Urakubo model centrally features Ca-CaM binding to NMDA receptors to mediate the allosteric effects -- this is not on GluN2B, but rather on the NR1 C0 region -- which is on the 2A subunit, not 2B -- so an entirely different mechanism (e.g., Akyol et al, 04).  Interestingly, the NR2A subtype is only expressed later in development, so the allosteric effect may have a developmental timecourse (Umemiya et al, 2001).

In preparation for understanding DAPK1, here's a summary of the key AutoP dynamic on CaMKII, as summarized in above figure.  First, T286 P and Ca/CaM binding are *separate processes* that nevertheless interact strongly.  There are separate vars for for P, noP, and various levels of Ca/CaM binding.  Further there is a "total activity" factor that integrates across P and Ca/CaM states, and yet another GluN2B binding function, so there are 4 primary dynamics:

* **CaM** -- CaM is attracted to T286P and binds strongly to bare CaMKIIP (eq 9, Kf=8, Kb=0.001), but otherwise it takes Ca+ activation to bind to CaMKII (noP) (eq 5, Kf=8 -- only for N2B bound, otherwise inactive binding rates).

only binds to noP CaMKII, *except* for the very first binding to CaMKIIP (eq 9), which has a high Kf (8) and slow Kb (0.001).  P *or* N2B is responsible for this high CaM binding, so eq 9 should apply to either case.  Any N2B bound CaMKII (noP) should have same as 9 affinity.  Also, T286P prevents further CaM binding (presumably the CaM is actually bound to T286?), and P also reduces magnitude of Ca binding to CaM (eq 8) but balances the Kf, Kb (both 1).  Question: should N2B also induce same 8 dynamics in CaMKII plain case?

* **Auto.K** -- determines rate of further AutoP, and it is a complicated function of current P and Ca/CaM binding.  Critically, noP but Ca/CaM binding contributes positively to rate of AutoP -- not only P itself.  And there is some "mass action" effect that is a function of any Ca/CaM or P state.  More P does accelerate this function, but not massively so: if completely left alone and driven by constant Ca input, it accelerates, but PP1 quickly turns off.  Also Ca/CaM unbinds with decent speed.

* **Auto.Act** -- total activation: to first order, any CaMKII with *either* P or Ca/CaM binding contributes, with weak diffs (e.g., 3CaCaMP is the most active, contributes with a factor of 1, but other ones contribute with factors between .75 - .8, so not that big of a diff).  This total act is what drives the further kinase activity for AMPAR trafficking.

* **GluN2B** binding -- There is a *distinct* function for GluN2B binding.  Per above, `GluN2BCaCaM` (10 Kf, 0.001 Kb) drives N2B binding as function of any CaM (does not accelerate for simplicity).  `GluN2BP` for CaMKIIP, no CaM (0.01 Kf, 0.0001 Kb) keeps P bound (low Kb), but does not drive much forward binding.  `GluN2NoP` for CaMKII (noP) (0.00001 Kf, 1000 Kb) unbinds noP (high Kb) quickly, and does not do much forward binding (todo just set to 0?)

## DAPK1 dynamics

![DAPK1 GluNR2B Binding](figs/fig_urakubo_dapk1.png?raw=true "DAPK1 Ca/CaM binding and autophosphorylation, showing the NR2B binding modulation at reactions 6 and 9, in terms of reduced backward rate constant 0.02 instead of 400, 0.001 and 1, along with 8 going forward at 9.  Adapted from SI3 in Urakubo modeling suppl.")

Goodell et al. (2017) provide a detailed investigation of DAPK1 dynamics.

* DAPK1/GluN2B binding is homologous to the well-studied CaMKII/GluN2B binding to the same region.

* However, it has the opposite dependence on Ca-CaM binding: inversely proportional -- key question is exactly how.

* DAPK1 autophosphorylates at the Ser308 site (instead of Thr286, which is CaMKII spot).  de Diego et al., (2010) suggest that the S308 site is similar to T286 in having multiple additional binding steps, and remains phosphorylated after CaM dissociation (autophosphorylation).  Thus, it is reasonable to apply the same complicated T286 logic here it seems, but there are some twists.

* Per Shani et al. (2001), "It was found that in the absence of Ca2+/CaM, DRP-1 displayed unexpectedly high basal levels of autophosphorylation. The latter was strongly inhibited by Ca2+/CaM concentrations that fully activate the enzyme, suggesting inverse relationships between autophosphorylation and substrate phosphorylation."

    DAPK1P (S380 AutoP) = inactive form for binding to GluN2B (akin to its substrate P. effects) -- this is default state.  Data: mutant S308A cannot autoP, and shows increased substrate binding / apoptosis.  mutant S308D is always autoP (aspartic acid), and prevented apoptosis.

* Per Shohat et al (2001), "The autophosphorylation of Ser308 was found to be Ca2/CaM-independent; moreover, it was strongly inhibited by the addition of Ca2/CaM."  Thus, Ca/CaM inhibits P, and P inhibits Ca/CaM reciprocally.

* In summary, it starts out fully P, then with some initial deP from CaN, further Ca/CaM binding further inhibits additional autoP pressure -- so it integrates Ca/CaM into levels of deP and its overall kinetic binding activity is a function of Ca/CaM + deP (this can be a mirror of CaMKII where it is Ca/CaM + P).  Assuming strong binding is due to active Ca3CaM form of CaM -- no indication that raw CaM is bound?

* There does not appear to be a strong interaction between CaM binding and GluN2B binding?  Most of the interaction is with AutoP causing P, and CaM inhibiting that.  But GluN2B binding is just more-or-less an outcome of these CaM, deP vs. AutoP dynamics, without changing them significantly?  TODO: verify this!

* Goodell et al (2017) show that deP is due to CaN, although other lit shows PP2A deP activity, but that is likely in non-neuronal context (Bialik & Kimchi, 2014), and PP2A is not in PSD.

* To implement, we need Act to have reversed roles of P and noP, but AutoK is still driven by P but inhibited by Ca/CaM levels (invert that for example).

* Thus, overall, DAPK1 is default baseline autophosphorylated.  And it is relatively insensitive to Ca/CaM in this state, because of the inhibited binding constants.  However, when you get some initial CaN dephosphorylation activity, that then allows more Ca-CaM binding, which further inhibits autoP, and you move further away from full autoP.  It is this noP DAPK1 that is actually "active" catalytically: this is when it binds to GluN2B most strongly.

* So, key point: CaN is "pulling the trigger" for a learning event, after which time DAPK1 effectively measures / Camintegrates the Ca/CaM through a kind of inverse-autophosphorylation process (the opposite form of positive feedback loop present in CaMKII).  This is then the "opponent process" to the CaMKII, both of which are competing for GluN2B.  If DAPK1 wins out, then LTD happens.  If CaMKII wins, then LTP happens.  But critically, both are directly integrating Ca signals through a similar integrative, rate-sensitive dynamic!  However, *if* DAPK1 preferentially reflects the earlier minus phase time frame of activity (i.e., it decays more quickly), then it directly represents this minus phase co-product, while CaMKII gets to be the plus phase.  It could really be that simple and direct!

* CaN is the primary source of dephosphorylation at S308, such that CaN is primary initiator of DAPK1 binding to GluN2B.  CaN is directly activated by Ca/CaM, so it is in a good position to detect the onset of a learning window, opening up the DAPK1 to be sensitive to further Ca levels.  It is unclear (yet) if PP1 and PP2A, which dephosphorylate the T286 site on CaMKII, might also play a similar role on DAPK1, and if not, why not?  PP1 does de-activate CaN however, so indirectly it is having that effect, and it may thus set the overall time window for a given plasticity event.

* In summary, at the most abstract, computational level, CaMKII represents the plus phase $<xy>_s$ signal, and DAPK1 represents the minus phase $<xy>_m$ signal, in terms of time-integrated Ca++ signals across these phases. These two kinases share many properties, but differ critically in their signs (at multiple levels), and this makes them work just like a classic opponent-process: Go - NoGo, excitation vs. inhibition, etc. They both compete at the NMDA GluN2B site, so that is the final locus of the opponent competitive dynamic. If DAPK1 wins, LTD happens.
If CaMKII wins, LTP happens. If there is a tie, nothing happens! Perfect error-driven learning! Furthermore, both rely on a similar auto-phosphorylation (autoP) dynamic to integrate over recent synaptic activity.

# References

* Akyol, Z., Bartos, J. A., Merrill, M. A., Faga, L. A., Jaren, O. R., Shea, M. A., & Hell, J. W. (2004). Apo-Calmodulin Binds with its C-terminal Domain to the N-Methyl-d-aspartate Receptor NR1 C0 Region *. Journal of Biological Chemistry, 279(3), 2166–2175. https://doi.org/10.1074/jbc.M302542200

* Barcomb, K., Hell, J. W., Benke, T. A., & Bayer, K. U. (2016). The CaMKII/GluN2B Protein Interaction Maintains Synaptic Strength. *Journal of Biological Chemistry, 291(31),* 16082–16089. https://doi.org/10.1074/jbc.M116.734822

* Bayer, K.-U., De Koninck, P., Leonard, A. S., Hell, J. W., & Schulman, H. (2001). Interaction with the NMDA receptor locks CaMKII in an active conformation. *Nature, 411(6839),* 801–805. https://doi.org/10.1038/35081080

* Bhalla, U. S., & Iyengar, R. (1999). Emergent properties of networks of biological signaling pathways. *Science.* https://doi.org/10.1126/science.283.5400.381

* Coultrap, S. J., & Bayer, K. U. (2012). CaMKII regulation in information processing and storage. *Trends in Neurosciences, 35(10),* 607–618. https://doi.org/10.1016/j.tins.2012.05.003

* de Diego, I., Kuper, J., Bakalova, N., Kursula, P., & Wilmanns, M. (2010). Molecular basis of the death-associated protein kinase-calcium/calmodulin regulator complex. Science Signaling, 3(106), ra6. https://doi.org/10.1126/scisignal.2000552

* Dudek, S. M., & Bear, M. F. (1992). Homosynaptic long-term depression in area CA1 of hippocampus and effects of N-methyl-D-aspartate receptor blockade. *Proceedings of the National Academy of Sciences U. S. A., 89(10),* 4363–4367. http://www.ncbi.nlm.nih.gov/pubmed/1350090

* Doi, T., Kuroda, S., Michikawa, T., & Kawato, M. (2005). Inositol 1,4,5-trisphosphate-dependent Ca2+ threshold dynamics detect spike timing in cerebellar purkinje cells. *Journal of Neuroscience, 25(4),* 950–961. https://doi.org/10.1523/JNEUROSCI.2727-04.2005

* Dupont, G., Houart, G., & De Koninck, P. (2003). Sensitivity of CaM kinase II to the frequency of Ca2+ oscillations: A simple model. *Cell Calcium, 34(6),* 485–497. https://doi.org/10.1016/S0143-4160(03)00152-0

* Goodell, D. J., Zaegel, V., Coultrap, S. J., Hell, J. W., & Bayer, K. U. (2017). DAPK1 mediates LTD by making CaMKII/GluN2B binding LTP specific. *Cell Reports, 19(11),* 2231–2243. https://doi.org/10.1016/j.celrep.2017.05.068

* Kuroda, S., Schweighofer, N., & Kawato, M. (2001). Exploration of signal transduction pathways in cerebellar long-term depression by kinetic simulation. *Journal of Neuroscience, 21(15),* 5693–5702. https://doi.org/10.1523/JNEUROSCI.21-15-05693.2001

* Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-Type K+ Conductance in the Back-Propagation of Action Potentials in the Dendrites of Hippocampal Pyramidal Neurons. *Journal of Computational Neuroscience, 7(1),* 5–15. https://doi.org/10.1023/A:1008906225285

* Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic of Subthreshold Synaptic Summation in a Model CA1 Pyramidal Cell. *Neuron, 37(6),* 977–987. https://doi.org/10.1016/S0896-6273(03)00148-X

* Shani, G., Henis-Korenblit, S., Jona, G., Gileadi, O., Eisenstein, M., Ziv, T., Admon, A., & Kimchi, A. (2001). Autophosphorylation restrains the apoptotic activity of DRP-1 kinase by controlling dimerization and calmodulin binding. The EMBO Journal, 20(5), 1099–1113. https://doi.org/10.1093/emboj/20.5.1099

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. *The Journal of Neuroscience, 28(13),* 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598

* Umemiya, M., Chen, N., Raymond, L. A., & Murphy, T. H. (2001). A Calcium-Dependent Feedback Mechanism Participates in Shaping Single NMDA Miniature EPSCs. Journal of Neuroscience, 21(1), 1–9. https://doi.org/10.1523/JNEUROSCI.21-01-00001.2001
