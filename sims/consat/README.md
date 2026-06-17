# Constraint satisfaction

The Hopfield & Tank (1985) (HT85) model relies on single units with stereotyped wiring.

An actual Axon model requires distributed representations and collaborative interactions among sparsely-active populations of neurons. It is nothing like the HT85 paradigm. In particular, the normal inhibitory dynamics tend to restrict the impact of lateral connections gaining strength in a specific area and getting over threshold. In the current version, noisy weights are determining the entire dynamic of what gets active. This needs to be trained weights somehow. But how?

One alternative is training a network with an input of city-dots and output of some kind of ordering representation, but then it turns the problem into one of pattern recognition instead of CS. However, given all the interdependencies among the elements, and the high dimensional combinatorial space, perhaps this is still a good test. It seems like the right way to go next. Can pre-generate larger space if necessary.

Key points:

* standard visual representation of TSP does not encode the relevant data. distance needs to be explicitly represented. This was done by fiat in the flat TSP model, but the visual dots one simply does not have it.

* metric information like distance is not really of the essence for constraint satisfaction -- requires either "more is more" activation, pop codes, or well-tuned weights, etc. Usual issues. 

* simplest case would be binary input features, where the key property is that the overall *configuration* makes a difference -- it is about the *relationship* among the different input features. For example, figure-ground illusions, where suspicious configuration of elements "adds up" to a perceived figure.

* key challenge: need some automatic, algorithmic way of generating a sufficient number of these problems! what is the actual algorithmic recipe for how this works? some way of translating TSP into this? some kind of maze with different gates? but then it becomes a spatial problem again! need to have everything be simple and explicit.

What about the CNF problem?? Binary, seems good!

3SAT is case with 3 elements OR'd together, combined by ANDs. 

Learn this by mapping from input patterns to binary outputs: satisfiable or not

depends on the configuration.  worth a try!

problem: one key factor is the identity of the variable across clauses. This requires encoding this identity in some way -- but then there is a "matching" problem of identifying where the variables are re-used across problems. that is going to require additional encodings. It might just learn these, but we'll have to see. Need to start of course with a small number of variables. maybe just fix it at 3, so it is always re-used across problems? and keep the ordered list of vars the same? then it is basically just the permutation of the negation signs is the only actual variable. don't need to represent the variables explicitly -- because they are constant, it is implicit. can look at what that surface looks like for different numbers of clauses.

basically, it is like the parity problem. but with more interesting surface?

## TODO



