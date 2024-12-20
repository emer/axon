# Arm Maze (armaze) Environment

The Arm Maze environment represents an N-armed maze ("bandit") with each Arm having a distinctive CS stimulus at the start (could be one of multiple possibilities) and (some probability of) a US outcome at the end of the maze (could be either positive or negative, with (variable) magnitude and probability.

It has a full 3D GUI showing the current state.

The controlling Sim is responsible for calling the NewStart() method when the agent should be placed back at the starting point.  Action() records the action and updates the State rendering of the action, but doesn't update the environment until Step, so the environment at the end of a trial always reflects the current trial state.


