# FFFB Inhibition

FFFB is the feedforward (FF) and feedback (FB) inhibition mechanism, originally developed for the Leabra model.  It produces a robust, graded k-Winners-Take-All dynamic of sparse distributed representations having approximately k out of N neurons active at any time, where k is typically 10-20 percent of N.

* FF is a simple linear function of the average netinput (Ge) coming into neurons in a layer / pool.  It is critical to have a `FF0` offset for the zero-point for FF inhibition.  Note that by giving FF access to the netinput, it implicitly has access to the strength of synaptic weights in a layer, which makes it automatically more robust over the course of learning.

* FB must be integrated over time to avoid oscillations, but is otherwise also a weighted proportion of average activity in a layer.

