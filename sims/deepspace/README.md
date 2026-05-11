# deepspace


## TODO

* Current params looking good! test and fix IOspike logic under GPU, also CNeUpLearn.

* separate VM for VOR vs. not, with differential VOR inhib control -- the case with VOR active should have 0 net error -- to extent it does have non-zero deviation, then it could modulate the error signal / learning rate on VOR?  This is in addition to the basic VS driver, which doesn't ever change. So in fact this is an essential additional factor!!

* VM is just Up, not Dn -- don't use output, just error, forward model has to be based on VS only. In general need to develop a better understanding of what Up does! Dn is simpler to understand.

* do we need VShvUp at all? I guess in comparison to dn

* inhib has to work in a different way from CNiIO -- getting massive carryover

* test env that presents:
	- sensory without motor: this is the key error signal
	- omit visual -- this is what will happen with VOR -- need separate input that indicates expected absence of visual input.

	
### VOR

## Mossy fiber inputs

* MF should represent the earliest version of the descending motor command, as that is the most predictive. 

* It needs to be time-varying, to allow predictive learning of a time-varying sensory signal. Don't want learning to over-generalize.

* In a realistic version of rotation, there would be ongoing distinct motor signals unfolding over time. These are what we want to capture.

* Simplest version for now is just to have a CSC-like pattern that maps out over time across different neurons. How to encode directionality and magnitude within this?

-> Pop version that just ripples through over time, with different time-specific neurons. Instead of current NUnitsPer redundancy, time is the Y axis.

Then cortical pattern is the time integration of this, whereas input is just the wave.

