# Constraint satisfaction

The Hopfield & Tank (1985) (HT85) model relies on single units with stereotyped wiring.

An actual Axon model requires distributed representations and collaborative interactions among sparsely-active populations of neurons. It is nothing like the HT85 paradigm. In particular, the normal inhibitory dynamics tend to restrict the impact of lateral connections gaining strength in a specific area and getting over threshold. In the current version, noisy weights are determining the entire dynamic of what gets active. This needs to be trained weights somehow. But how?

One alternative is training a network with an input of city-dots and output of some kind of ordering representation, but then it turns the problem into one of pattern recognition instead of CS. However, given all the interdependencies among the elements, and the high dimensional combinatorial space, perhaps this is still a good test. It seems like the right way to go next. Can pre-generate larger space if necessary.

Key points:

* standard visual representation of TSP does not encode the relevant data. distance needs to be explicitly represented. This was done by fiat in the flat TSP model, but the visual dots one simply does not have it.

* metric information like distance is not really of the essence for constraint satisfaction -- requires either "more is more" activation, pop codes, or well-tuned weights, etc. Usual issues. 

* simplest case would be binary input features, where the key property is that the overall *configuration* makes a difference -- it is about the *relationship* among the different input features. For example, figure-ground illusions, where suspicious configuration of elements "adds up" to a perceived figure.

* key challenge: need some automatic, algorithmic way of generating a sufficient number of these problems! what is the actual algorithmic recipe for how this works? some way of translating TSP into this? some kind of maze with different gates? but then it becomes a spatial problem again! need to have everything be simple and explicit.

What about the 3SAT CNF satisfiability problem?? Binary, seems good! Except: binary is terrible for the sparse activation constraint. And indeed, this seems fatal in the implemented network. Note: implemented 3SAT using 3 variables that were repeated across clauses, so minimal number of vars, and thus the full CNF condition could be expressed only in terms of the negation operators, which avoids binding issues etc: the variables are constant and implicit. Interestingly, it requires 8 clauses before unsatisfiable cases emerge, and initially they are a tiny fraction. The summary on SO (https://cstheory.stackexchange.com/questions/2168/how-many-instances-of-3-sat-are-satisfiable) suggests that the ratio of clauses / variables defines a critical point (at a value of 4.2) where the space transitions from predominantly satisfiable to predominantly unsatisfiable. This accords well: 8/3 = 2.6667 is first emergence of any unsat, and by 10/3 (3.333) it is growing as a proportion steadily. The 4.2 ratio suggests that between 12 and 13 (12.6) is the transition point, but the combinatorial explosion at that point is already pushing the limits.

Next step is NAry CNF: OR = MAX, AND = MIN. Really the key thing about all these problems is that it needs to depend on the configuration of things, and the MIN and MAX clearly have that character. So, start with just computing MIN and MAX operators and see how that goes.

CNF is too easy: the MIN and MAX operators are easily computed with weights that generalize very well. This is too much pattern recognition and not enough constraint satisfaction.

Per https://en.wikipedia.org/wiki/Constraint_satisfaction_problem -- generic version of CSP is just variables with some kind of arbitrary constraint matricies. So, just need to come up with a bunch of generic variables and then some random constraints, and then have the thing learn this. Don't need any actual semantics. Just need to have relationships and sufficient numbers of constraints to balance things out -- randomly generate constraints until finding one that produces balanced results. Relationships can just be > = <.

Key thing: need the output to be somehow more than binary! Ok, so could define different *classes* according to these constraints, and the problem is to classify. So you have N different sets of constraints, and you need to find the one that fits the best -- that seems good!!

## TODO


