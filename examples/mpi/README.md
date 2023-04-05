# MPI Message Passing Interface Example

This is a version of the ra25 example that uses MPI to distributed computation across multiple processors (*procs*).  See [Wiki MPI](https://github.com/emer/emergent/wiki/MPI) for more info.

N completely separate instances of the same simulation program are run in parallel, and they communicate weight changes and trial-level log data amongst themselves.  Each proc thus trains on a subset of the total set of training patterns for each epoch.  Thus, dividing the patterns across procs is the most difficult aspect of making this work.  The mechanics of synchronizing the weight changes and etable data are just a few simple method calls.

Speedups approach linear, because the synchronization is relatively infrequent, especially for larger networks which have more computation per trial.  The biggest cost in MPI is the latency: sending a huge list of weight changes infrequently is much faster overall than sending smaller amounts of data more frequently.

You can only use MPI for running in nogui mode, using command-line args -- otherwise you'd get multiple copies of the GUI running..

# Building and running

To build with actual mpi support, you must do:

```bash
$ go build -tags mpi
```

which is in the provided `Makefile`, so you can also do:
```bash
$ make
```

otherwise it builds with a dummy version of mpi that doesn't actually do anything (convenient for enabling both MPI and non-MPI support in one codebase).  Always ensure that your code does something reasonable when mpi.WorldSize() == 1 -- that is what the dummy code returns.

Also you should check the `ss.Args.Bool("mpi")`, set by the `-mpi` command line arg, to do different things depending -- e.g., don't try to aggregate DWts if not using MPI, as it will waste a lot of time and accomplish nothing.

To run, do something like this:

```bash
$ mpirun -np 2 ./mpi -mpi
```

The number of processors must divide into 24 for this example (number of patterns used in this version of ra25) evenly (2, 3, 4, 6, 8).

# General tips for MPI usage

* **MOST IMPORTANT:** all procs *must* remain *completely* synchronized in terms of when they call MPI functions -- these functions will block until all procs have called the same function.  The default behavior of setting a saved random number seed for all procs should ensure this.  But you also need to make sure that the same random permutation of item lists, etc takes place across all nodes.  The `empi.FixedTable` environment does this for the case of a table with a set of patterns.

* Any logs recording below the Epoch level need to be sync'd across nodes before aggregating data at the Epoch level, using `ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)`.


# Key Diffs from ra25

Here are the main diffs that transform the ra25.go example into this mpi version:

* Search for `mpi` in the code (case insensitive) -- most of the changes have that in or near them.

* Most of the changes are the bottom of the file.

## main() Config() call

At the top of the file, it can be important to configure `TheSim` *after* mpi has been initialized, if there are things that are done differently there -- thus, you should move the `TheSim.Config()` call into `CmdArgs`:

```go
func main() {
	TheSim.New() // note: not running Config here -- done in CmdArgs for mpi / nogui
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		TheSim.Config()      // for GUI case, config then run..
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}
```

## Sim struct

There are some other things added but they are just more of what is already there -- these are the uniquely MPI parts, at end of Sim struct type:

```go
	Comm    *mpi.Comm `view:"-" desc:"mpi communicator"`
	AllDWts []float32 `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts []float32 `view:"-" desc:"buffer of MPI summed dwt weight changes"`
```

## Allocating Patterns Across Nodes

In `ConfigEnv`, non-overlapping subsets of input patterns are allocated to different nodes, so that each epoch has the same full set of input patterns as with one processor.

```go
	ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	if ss.UseMPI {
		st, ed, _ := empi.AllocN(ss.Pats.Rows)
		ss.TrainEnv.Table.Idxs = ss.TrainEnv.Table.Idxs[st:ed]
	}
```

## Logging

The `elog` system has support for gathering all of the rows of Trial-level logs from each of the different processors into a combined table, which is then used for aggregating stats at the Epoch level.  To enable all the standard infrastructure to work in the same way as in the non-MPI case, the aggregated table is set as the Trial log.  This means that after we do the aggregation of the Trial data (at the Epoch level), we need to reset the number of rows back to the original number present per each processor, otherwise the table grows exponentially!  If the Trial data is always accumulated by adding rows and resetting back to 0 at the end of the epoch, then you would just do that as usual.

Here's the relevant code in the `Log()` method:

```go
	var saveRow int
	if ss.Args.Bool("mpi") && time == etime.Epoch { // gather data for trial level at epoch
		saveRow = ss.Logs.Table(mode, etime.Trial).Rows
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}
    ...
	if ss.Args.Bool("mpi") && time == etime.Epoch { // reset rows back to original pre-gather
		dt := ss.Logs.Table(mode, etime.Trial)
		dt.SetNumRows(saveRow)
	}
```

## CmdArgs

At the end, CmdArgs has quite a bit of MPI-specific logic in it, which we don't reproduce here -- see `ra25.go` code and look for mpi.

We use `mpi.Printf` instead of `fmt.Printf` to have it only print on the root node, so you don't get a bunch of duplicated messages.

## MPI Code

The main MPI-specific code is at the end, reproduced here for easy reference.  NOTE: please always use the code in ra25.go as a copy-paste source as there might be a few small changes, which will be more closely tracked there than here.


```go
// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Args.Bool("mpi") {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
// includes all other long adapting factors too: DTrgAvg, ActAvg, etc
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.Args.Bool("mpi") {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
    // note: if using GPU, add: ss.Net.SyncAllToGPU() here!
	ss.Net.WtFmDWt(&ss.Context)
}
```

