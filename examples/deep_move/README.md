# deep_move

This example tests the `deep` predictive learning model on predicting the effects of moving within a first-person depth-map view of an agent moving within a simple square-shaped environment.  There are two predictions that are updated: head direction relative to some kind of "north" compass direction, which changes with rotations, and a depth map to the walls, encoded using a population code for each of N angles within an overall field of view (FOV).  This uses a simplified version of the [Flat World](https://github.com/emer/envs/tree/master/fworld) environment, and this project serves as an testing and optimization platform for the larger [Emery Map Nav](https://github.com/ccnlab/map-nav/tree/master/sims/emery2) model.

The parameters that work well for `deep_music` don't work here at all, for reasons that make sense: https://github.com/emer/axon/discussions/61  This model requires short time scale integration (i.e., Markovian, one-step history) and local, topographic connectivity to process different depth ranges and angle positions independently.  A fully connected hidden layer does not work at all: it tries to process the depth map wholistically, and that just doesn't work.

# Connectivity and Computational Logic

The essential computational solution in the Depth prediction network is that the Hidden layer encodes the current pattern of depths using topographically organized connectivity, so each neuron is constrained to process a limited range of depths and angles, with the `Action` input modulating this activity to indicate which direction it will move on the next timestep.  Note that the action reflects what will happen *next*, not what happened to create the current visual input.

This information is captured in the CT layer and maintained in an essentially static, one-to-one copy that then drives projections into the Pulvinar layer that decode the Action x Depth signal into the predicted next activity pattern.  The error signals coming from the pulvinar back to the superficial layer are sufficient to shape the learning to properly encode the relevant action and depth information.

* Interestingly, the predictive error at time t is about the encoding that took place at time t-1 in the hidden layer.  Without resorting to backprop through time, it is not possible to push these errors backward in time.  Thus, the primary locus of error-driven learning is at the TRC (Pulvinar) synapse.  In this case the CT representation just needs to contain all the information, and the TRC decodes it --- similar to a reservoir computing model where all the learning is on the decoder, and the reservoir itself does not learn, and is just configured to provide a sufficiently pattern-separated, time-evolving representation.

* The presence of the trace mechanism, which uses prior time activity for the credit assignment factor, provides an important additional way for errors at time t to reshape learning on states from t-1.  However, the receiving activations are still from the wrong time.

# Performance

The HeadDir changes can be predicted at roughly .96 correlation value, which is essentially at ceiling for a spiking network.  The trial-to-trial correlation is around .4, so if the model was merely copying the previous trial's values for its prediction, that's what it would get.  There is some increased tendency for the predicted values to correlate with the previous trial, at around .45.

The Depth map can be predicted at around .83 for 180 FOV, AngInc = 45 (i.e. 5 depths).  When you go up to AngInc = 15, the self-correlations in the depth map end up pushing the baseline trial-wise correlations up, so it is harder to see the difference.  Thus, 180 / 45 is a good test case, and likely captures most of the relevant info.


