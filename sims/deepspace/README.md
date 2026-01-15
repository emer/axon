# deepspace

TODO: do VOR

## Mossy fiber inputs

* MF should represent the earliest version of the descending motor command, as that is the most predictive. 

* It needs to be time-varying, to allow predictive learning of a time-varying sensory signal. Don't want learning to over-generalize.

* In a realistic version of rotation, there would be ongoing distinct motor signals unfolding over time. These are what we want to capture.

* Simplest version for now is just to have a CSC-like pattern that maps out over time across different neurons. How to encode directionality and magnitude within this?

-> Pop version that just ripples through over time, with different time-specific neurons. Instead of current NUnitsPer redundancy, time is the Y axis.

Then cortical pattern is the time integration of this, whereas input is just the wave.

