# Bench Objrec

bench_objrec is supposed to be an easy-to-understand, easy-to-run network that nevertheless has the same performance characteristics as the big LVis network.
As compared to examples/bench_lvis, bench_objrec is more difficult to understand (uses Looper, lots of logging code), but actually converges and has a working GUI.

This network is similar to [lvis/sims/objrec](https://github.com/ccnlab/lvis/tree/main/sims/objrec).
See the [README](https://github.com/ccnlab/lvis/blob/main/sims/objrec/README.md) over there for more detailed explanations.

Changes: 
- Works with most recent Axon version.
- Prints a few basic stats to stdout

TODOs:
- [ ] Delete lots of the unnecessary code (like everything in logging, except the most basic stats e.g. `PhaseDiff`)
- [ ] Instead of being `package main`, turn the `main()` into a `BenchmarkObjrec(b *testing.B)` to make it easier to run & profile. Currently, this prohibits the use of a GUI (the window never gets created, just blocks [here](https://github.com/go-gl/glfw/blob/main/v3.3/glfw/window.go#L348) forever).
