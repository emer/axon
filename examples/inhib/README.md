# Introduction

This is adapted from [CCN Sims](https://github.com/CompCogNeuro/sims).

This simulation explores how inhibitory interneurons can dynamically control overall activity levels within the network, by providing both feedforward and feedback inhibition to excitatory pyramidal neurons.  This inhibition is critical when neurons have bidirectional excitatory connections, as otherwise the positive feedback loops will result in the equivalent of epileptic seizures -- runaway excitatory activity.

In the `axon` framework, it is also critical for exploring basic properties of spiking dynamics and especially the importance of GABA-B and NMDA channels for establishing a clearer, more stable separation between the most strongly activated neurons and the rest, which in turn is essential for enabling credit assignment during learning, causing neurons to become relatively specialized and differentiated from each other.

The network is organized with a configurable number of layers, connected sequentially and bidirectionally, starting with the input layer named `Layer0`.  Each such layer of excitatory pyramidal neurons projects to the next layer of excitatory units, and a layer of 20 inhibitory neurons (`Inihib`). These inhibitory neurons regulate the activation level of the hidden layer units by sending inhibition proportional to incoming excitation. The ratio of 20 inhibitory units to 120 total hidden units (17 percent) is similar to that found in the cortex, which is commonly cited as roughly 15 percent (White, 1989a; Zilles, 1990). The inhibitory neurons are similar to the excitatory neurons (but with several important parameter differences, causing them to fire more regularly and reliably).

# Exploration

Let's begin as usual by viewing the weights of the network.

* Select `r.Wt` in the `Net` netview and then click on some of the `Layer` excitatory and `Inhib` layer units.

Most of the weights are random, except for those from the inhibitory units, which are fixed at a constant value of .5. Notice also that the hidden layer excitatory units receive from the input and inhibitory units, while the inhibitory units receive feedforward connections from the input layer, and feedback connections from the excitatory hidden units, as well as inhibitory connections from themselves.

Now, we will run the network. Note the graph view above the network, which will record the overall levels of activation (average activation) in the hidden and inhibitory units.

* Select `Act` to view activations in the network window, and press `Init` and `Test Trial` in the toolbar.

You will see the input units activated by a random activity pattern, and after several cycles of activation updating, the hidden and inhibitory units will become active. The activation appears quite controlled, as the inhibition counterbalances the excitation from the input layer.

* Select the `TstCycPlot` tab to view a plot of average activity in the layers -- you should see that the first excitatory layer (black line) has around 10 percent activation once it stabilizes after some initial oscillations, while the next layer up is just a bit less active.

* Select the `Spike Rasters` tab to view a plot of unit spiking over time (horizontal axis), which gives a better sense of the relative spike timing within and between layers.

In the next sections, we manipulate some of the parameters in the control panel to get a better sense of the principles underlying the inhibitory dynamics in the network -- you can just stay with this plot view to see the results more quickly than watching the network view update.

# GABA-B and NMDA channels for stabilizing patterns

An essential step for enabling spiking neurons to form suitably stable, selective representations for learning was the inclusion of both NMDA and GABA-B channels, which are voltage dependent in a complementary manner as captured in the Sanders et al, 2013 model (which provided the basis for the implementation here).  These channels have long time constants and the voltage dependence causes them to promote a bistable activation state, with a smaller subset of neurons that have extra excitatory drive from the NMDA and avoid extra inhibition from GABA-B, while a majority of neurons have the opposite profile: extra inhibition from GABA-B and no additional excitation from NMDA.

With stronger conductance levels, these channels can produce robust active maintenance dynamics characteristic of layer 3 in the prefrontal cortex (PFC), but for posterior cortex, we use lower values that produce a weaker, but still essential, form of bistability.  Without these channels, neurons all just "take turns" firing at different points in time, and there is no sense in which a small subset are engaged to represent a specific input pattern -- that had been a blocking failure in all prior attempts to use spiking in Leabra models.

To see this in action:

* Set `GbarGABAB` and `GbarNMDA` both to 0, then continue to `Test Trial` (parameters are updated when this button is pressed), while looking at the `Spike Rasters` plot.

You should see that the `Layer2` spiking becomes highly synchronized and every neuron in the layer fires in each wave of activity, which appears as a solid vertical bar across the plot (with a bit of jitter over a few msec one way or the other).  This also happens in `Layer1` but somewhat less dramatically.  It is possible to tweak other parameters to reduce the strength of this oscillation, but inevitably the network ends up with each neuron in the layer firing at roughly the same overall frequency as the others, despite relatively distinct weight patterns.  This undifferentiated, non-selective pattern of activity prevents the network from learning to allocate different neurons to represent different things.

* Click on `Spike Correls` which shows the spike correlations across all spikes in a layer, both the auto-correlation within a layer (e.g., `Layer2:Layer2`) and between the inhibitory and excitatory layers (e.g., `Layer2:Inhib2`).  Select those items to plot in turn.  

You should see a strongly oscillating wavy pattern, reflecting the periodic nature of the spiking.  

* Hit the `Defaults` button and then `Test Trial` again a couple of times, which restores the GABA-B and NMDA conductances.   Note that now the correlograms look much smoother, indicating that the strong oscillations are no longer present.

* You can also see these effects in the `Net` view, where, with the default parameters, a subset of neurons has significant activity according to the `Act` variable (which integrates over spiking to show the running-average rate of spiking).  However with the GABAB and NMDA conductances set to 0, the activity patterns are much more diffuse and every neuron is weakly activated.


# Strength of Inhibitory Conductances

Let's start by manipulating the maximal conductance for the inhibitory current into the excitatory units, `HiddenGbarI`, which multiplies the level of inhibition coming into the hidden layer (excitatory) neurons (this sets the `Act.Gbar.I` parameter in the Hidden layer). Clearly, one would predict that this plays an important role.

* Decrease `HiddenGbarI` from .3 to .2 and do `Test Trial`. Then increase it to .5 and test again.

> **Question 3.6:** What effects does decreasing and increasing `HiddenGbarI` have on the average level of excitation of the hidden units and of the inhibitory units, and why does it have these effects (simple one-sentence answer)?

* Set `HiddenGBarI` back to .4 (or just hit the `Defaults` button). 

Now, let's see what happens when we manipulate the corresponding parameter for the inhibition coming into the inhibitory neurons, `InhibGbarI`. You might expect to get results similar to those just obtained for `HiddenGbarI`, but be careful -- inhibition upon inhibitory neurons could have interesting consequences.

* First run with a `InhibGbarI` of .75 for comparison. Then decrease `InhibGbarI` to .6 and Run, and next increase `InhibGbarI` to 1.0 and Run. 

With a `InhibGbarI` of .6, you should see that the excitatory activation drops, but the inhibitory level stays roughly the same! With a value of 1.0, the excitatory activation level increases, but the inhibition again remains the same. This is a difficult phenomenon to understand, but the following provide a few ways of thinking about what is going on.

First, it seems straightforward that reducing the amount of inhibition on the inhibitory neurons should result in more activation of the inhibitory neurons. If you just look at the very first blip of activity for the inhibitory neurons, this is true (as is the converse that increasing the inhibition results in lower activation). However, once the feedback inhibition starts to kick in as the hidden units become active, the inhibitory activity returns to the same level for all runs. This makes sense if the greater activation of the inhibitory units for the `InhibGbarI` = .6 case then inhibits the hidden units more (which it does, causing them to have lower activation), which then would result in *less* activation of the inhibitory units coming from the feedback from the hidden units. This reduced activation of the inhibitory neurons cancels out the increased activation from the lower `InhibGbarI` value, resulting in the same inhibitory activation level. The mystery is why the hidden units remain at their lower activation levels once the inhibition goes back to its original activation level.

One way we can explain this is by noting that this is a *dynamic* system, not a static balance of excitation and inhibition. Every time the excitatory hidden units start to get a little bit more active, they in turn activate the inhibitory units more easily (because they are less apt to inhibit themselves), which in turn provides just enough extra inhibition to offset the advance of the hidden units. This battle is effectively played out at the level of the *derivatives* (changes) in activations in the two pools of units, not their absolute levels, which would explain why we cannot really see much evidence of it by looking at only these absolute levels.

A more intuitive (but somewhat inaccurate in the details) way of understanding the effect of inhibition on inhibitory neurons is in terms of the location of the thermostat relative to the AC output vent -- if you place the thermostat very close to the AC vent (while you are sitting some constant distance away from the vent), you will be warmer than if the thermostat was far away from the AC output. Thus, how strongly the thermostat is driven by the AC output vent is analogous to the `InhibGbarI` parameter -- larger values of `InhibGbarI` are like having the thermostat closer to the vent, and will result in higher levels of activation (greater warmth) in the hidden layer, and the converse for smaller values.

* Set `InhibGbarI` back to .75 before continuing (or hit Defaults). 


# Roles of Feedforward and Feedback Inhibition

Next we assess the importance and properties of the feedforward versus feedback inhibitory projections by manipulating their relative strengths. The control panel has two parameters that determine the relative contribution of the feedforward and feedback inhibitory pathways: `FFinhibWtScale` applies to the feedforward weights from the input to the inhibitory units, and `FBinhibWtScale` applies to the feedback weights from the hidden layer to the inhibitory units. These parameters (specifically the .rel components of them) uniformly scale the strengths of an entire projection of connections from one layer to another, and are the arbitrary `WtScale.Rel` (r_k) relative scaling parameters described in *Net Input Detail* Appendix in [CCN TExtbook](https://github.com/CompCogNeuro/ed4).

* Set `FFInhibWtScale` to 0, effectively eliminating the feedforward excitatory inputs to the inhibitory neurons from the input layer (i.e., eliminating feedforward inhibition). 

> **Question 3.7:** How does eliminating feedforward inhibition affect the behavior of the excitatory and inhibitory average activity levels -- is there a clear qualitative difference in terms of when the two layers start to get active, and in their overall patterns of activity, compared to with the default parameters?

* Next, set `FFinhibWtScale` back to 1 and set `FBinhibWtScale` to 0 to turn off the feedback inhibition, and Run. 

Due to the relative renormalization property of these .rel parameters, you should see that the same overall level of inhibitory activity is achieved, but it now happens quickly in a feedforward way, which then clamps down on the excitatory units from the start -- they rise very slowly, but eventually do approach roughly the same levels as before.

These exercises should help you to see that a combination of both feedforward and feedback inhibition works better than either alone, for clear principled reasons. Feedforward can anticipate incoming activity levels, but it requires a very precise balance that is both slow and brittle. Feedback inhibition can react automatically to different activity levels, and is thus more robust, but it is also purely reactive, and thus can be unstable and oscillatory unless coupled with feedforward inhibition.

## Time Constants and Feedforward Anticipation

We just saw that feedforward inhibition is important for anticipating and offsetting the excitation coming from the inputs to the hidden layer. In addition to this feedforward inhibitory connectivity, the anticipatory effect depends on a difference between excitatory and inhibitory neurons in their rate of updating, which is controlled by the `Dt.GTau` parameters `HiddenGTau` and `InhibGTau` in the control panel (see [CCN Textbook](https://github.com/CompCogNeuro/ed4), Chapter 2). As you can see, the excitatory neurons are updated at tau of 40 (slower), while the inhibitory are at 20 (faster) -- these numbers correspond roughly to how many cycles it takes for a substantial amount of change happen. The faster updating of the inhibitory neurons allows them to more quickly become activated by the feedforward input, and send anticipatory inhibition to the excitatory hidden units before they actually get activated.

* To verify this, click on Defaults, set `InhibGTau` to 40 (instead of the 20 default), and then Run. 

You should see that the inhibition is no longer capable of anticipating the excitation as well, resulting in larger initial oscillations. Also, the faster inhibitory time constant enables inhibition to more rapidly adapt to changes in the overall excitation level. There is ample evidence that cortical inhibitory neurons respond faster to inputs than pyramidal neurons (e.g., Douglas & Martin, 1990).

One other important practical point about these update rate constants will prove to be an important advantage of the simplified inhibitory functions described in the next section. These rate constants must be set to be relatively slow to prevent oscillatory behavior.

* To see this, press `Defaults`, and then set `HiddenGTau` to 5, and `InhibGTau` to  2.5 and Run. 

The major, persistent oscillations that are observed here are largely prevented with slower time scale upgrading, because the excitatory neurons update their activity in smaller steps, to which the inhibitory neurons are better able to smoothly react.

# Effects of Learning

One of the important things that inhibition must do is to compensate adequately for the changes in weight values that accompany learning.  Typically, as units learn, they develop greater levels of variance in the amount of excitatory input received from the input patterns, with some patterns providing strong excitation to a given unit and others producing less. This is a natural result of the specialization of units for representing (detecting) some things and not others. We can test whether the current inhibitory mechanism adequately handles these changes by simulating the effects of learning, by giving units excitatory weight values with a higher level of variance.

* First, press `Defaults` to return to the default parameters. Run this case to get a baseline for comparison. 

In this case, the network's weights are produced by generating random numbers with a mean of .25, and a uniform variance around that mean of .2.

* Next, set the `TrainedWts` button on, do `Init` (this change does not take effect unless you do that, to initialize the network weights), and `Test Trial`.

The weights are then initialized with the same mean but a variance of .7 using Gaussian (normally) distributed values. This produces a much higher variance of excitatory net inputs for units in the hidden layer.  There is also an increase in the total overall weight strength with the increase in variance because there is more room for larger weights above the .25 mean, but not much more below it.

You should observe a greater level of excitation using the trained weights compared to the initial untrained weights.

* You can verify that the system can compensate for this change by increasing the `HiddenGbarI` to .5. 

* Before continuing, set `TrainedWts` back off, and do `Init` again.

# Bidirectional Excitation

To make things simpler at the outset, we have so far been exploring a relatively easy case for inhibition where the network does not have bidirectional excitatory connectivity, which is where inhibition really becomes essential to prevent runaway positive feedback dynamics. Now, let's try running a network with two bidirectionally connected hidden layers.

* First, select Defaults to get back the default parameters, do a Run for comparison, and then click on the `BidirNet`, and click on the `Bidir Net` tab to view this network. 

In extending the network to the bidirectional case, we also have to extend our notions of what feedforward inhibition is. In general, the role of feedforward inhibition is to anticipate and counterbalance the level of excitatory input coming into a layer. Thus, in a network with bidirectional excitatory connectivity, the inhibitory neurons for a given layer also have to receive the top-down excitatory connections, which play the role of "feedforward" inhibition.

⇒ Verify that this network has both bidirectional excitatory connectivity and the "feedforward" inhibition coming back from the second hidden layer by examining the `r.Wt` weights as usual. 

* Now `Init` and `Test Trial` this network.  Then click back over to `TstCycPlot` to see average activity over time.

The plot shows the average activity for only the first hidden and inhibitory layers (as before). Note that the initial part up until the point where the second hidden layer begins to be active is the same as before, but as the second layer activates, it feeds back to the first layer inhibitory neurons, which become more active, as do the excitatory neurons. However, the overall activity level remains quite under control and not substantially different than before.  Thus, the inhibition is able to keep the positive feedback dynamics fully in check.

Next, we will see that inhibition is differentially important for bidirectionally connected networks.

* Set the `HiddenGbarI` parameter to .35, and `Test Trial`.

This reduces the amount of inhibition on the excitatory neurons. Note that this has a relatively small impact on the initial, feedforward portion of the activity curve, but when the second hidden layer becomes active, the network becomes catastrophically over activated -- an epileptic fit!

* Set the `HiddenGbarI` parameter back to .4.

# Exploration of FFFB Inhibition

You should run this section after having read the *FFFB Inhibition Function* section of the [CCN Textbook](https://github.com/CompCogNeuro/ed4).

* Reset the parameters to their default values using the `Defaults` button, click the `BidirNet` on to use that, and then Test to get the initial state of the network. This should reproduce the standard activation graph for the case with actual inhibitory neurons.

* Now, set `FFFBInhib` on to use the FFFB function described in the main text. Also set the `HiddenGbarI` and `InhibGbarI` parameters to 1 (otherwise the computed inhibition will be inaccurate), and the rate constant parameters to their defaults for normal (non unit inhib) operation, which is `HiddenGTau` and `InhibGTau` both to 1.4. Finally, you need to turn off the inhibitory projections (when present, these will contribute in addition to whatever is computed by FFFB - but we want to see how FFFB can do on its own): set `FmInhibWtScaleAbs` to 0 (this sets the absolute scaling factor of the connections from inhibitory neurons to 0, effectively nullifying these connections).  

* Press `Test Trial`.

The activations should be right around the 10-15% activity level. How does this change with trained weights as compared to the default untrained weights?

* Set `TrainedWts` on, do `Init`, and `Test Trial`.

You should see the hidden activities approach the 20% level now -- this shows that FFFB inhibition is relatively flexible and overall activity levels are sensitive to overall input strength. You should also notice that FFFB dynamics allow the network to settle relatively quickly -- this is due to using direct and accurate statistics for the incoming netinput and average activations, as compared to the more approximate sampling available to interneurons. Thus, FFFB is probably still more powerful and effective than the real biological system, but this does allow us to run our models very efficiently -- for a small number of cycles per input.

* To test the set point behavior of the FFFB functions, we can vary the amount of excitatory input by changing the `InputPct` levels, to 10 and 30 instead of the 20% default. After you change `InputPct`, you need to do `ConfigPats` in the toolbar (this makes a new input pattern with this percent of neurons active), and then `Test Trial`. 

> **Question 3.8:** How much does the hidden average activity level vary as a function of the different `InputPct` levels (10, 20, 30). What does this reveal about the set point nature of the FFFB inhibition mechanism (i.e., the extent to which it works like an air conditioner that works to maintain a fixed set-point temperature)?

Finally, you can explore the effects of changing the `*GbarI` and `FFinhibWtScale`, `FBinhibWtScale` parameters, which change the overall amount of inhibition, and amounts of feedforward and feedback inhibition, respectively.


