# Constraint satisfaction

The Hopfield & Tank (1985) (HT85) model relies on single units with stereotyped wiring.

An actual Axon model requires distributed representations and collaborative interactions among sparsely-active populations of neurons. It is nothing like the HT85 paradigm. In particular, the normal inhibitory dynamics tend to restrict the impact of lateral connections gaining strength in a specific area and getting over threshold. In the current version, noisy weights are determining the entire dynamic of what gets active. This needs to be trained weights somehow. But how?

One alternative is training a network with an input of city-dots and output of some kind of ordering representation, but then it turns the problem into one of pattern recognition instead of CS. However, given all the interdependencies among the elements, and the high dimensional combinatorial space, perhaps this is still a good test. It seems like the right way to go next. Can pre-generate larger space if necessary.


