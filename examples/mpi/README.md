# MPI Message Passing Interface Example

This is a version of the ra25 example that uses MPI to distributed computation across multiple processors (*procs*).  See [Wiki MPI](https://github.com/emer/emergent/wiki/MPI) for more info.

N completely separate instances of the same simulation program are run in parallel, and they communicate weight changes and trial-level log data amongst themselves.  Each proc thus trains on a subset of the total set of training patterns for each epoch.  Thus, dividing the patterns across procs is the most difficult aspect of making this work.  The mechanics of synchronizing the weight changes and table data are just a few simple method calls.

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

## Sim struct

There are some other things added but they are just more of what is already there -- these are the uniquely MPI parts, at end of Sim struct type:

```go
	Comm    *mpi.Comm `view:"-" desc:"mpi communicator"`
	AllDWts []float32 `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
```

## Allocating Patterns Across Nodes

In `ConfigEnv`, non-overlapping subsets of input patterns are allocated to different nodes, so that each epoch has the same full set of input patterns as with one processor.

```go
	ss.TrainEnv.Table = table.NewIndexView(ss.Pats)
	if ss.Config.Run.MPI {
		st, ed, _ := empi.AllocN(ss.Pats.Rows)
		ss.TrainEnv.Table.Indexes = ss.TrainEnv.Table.Indexes[st:ed]
	}
```

In other sims with more complex or interactive environments, it is best to give each environment its own random seed using the `erand.SysRand` which implements the `erand.Rand` interface, and can be passed to any of the `erand` methods (which wrap and extend the go standard `rand` package functions).  See the `boa` model for example.

```go
	Rand        erand.SysRand `view:"-" desc:"random number generator for the env -- all random calls must use this"`
	RndSeed     int64         `inactive:"+" desc:"random seed"`
```

The built-in data parallel processing in v1.8 requires coordinating the allocation of inputs across both.  In this example project, there are 24 input patterns, and mpi takes the first cut by allocating subsets of the patterns that each node processes.  Then, the remaining patterns can be learned using data parallel within each node.  It may be important to ensure that these divide the total equally, for example:

* `mpi -np 2` leaves 12 patterns for within-node data parallel, so -ndata=4, 6, or 12 work there.
* `mpi -np 4` only leaves 6 patterns, so -ndata=2, 3, 6 are options.

In this particular case, it would just use more trials per epoch and the environment wraps around, so it isn't that big of a deal, but it might matter for other sims.

## Logging

The `elog` system has support for gathering all of the rows of Trial-level logs from each of the different processors into a combined table, which is then used for aggregating stats at the Epoch level.  To enable all the standard infrastructure to work in the same way as in the non-MPI case, the aggregated table is set as the Trial log. 

In most cases, the trial log is reset at the start of the new epoch, so the aggregated data will be reset.  However, if the logging logic uses the trial number, you need to reset the number of rows back to the original number present per each processor, otherwise the table grows exponentially!

Here's the relevant code in the `Log()` method:

```go
	if ss.Config.Run.MPI {
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}
```

To record the trial log data for each MPI processor, you need to set log files for each (by default log files are only saved for the 0 rank):

```go
	if ss.Config.Log.Trial {
		fnm := elog.LogFilename(fmt.Sprintf("trl_%d", mpi.WorldRank()), ss.Net.Name(), ss.Params.RunName(ss.Config.Run.Run))
		ss.Logs.SetLogFile(etime.Train, etime.Trial, fnm)
	}
```


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
		ss.Config.Run.MPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Config.Run.MPI {
		mpi.Finalize()
	}
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	ctx := &ss.Context
	if ss.Config.Run.MPI {
		ss.Net.CollectDWts(ctx, &ss.AllDWts)
		ss.Comm.AllReduceF32(mpi.OpSum, ss.AllDWts, nil) // in place
		ss.Net.SetDWts(ctx, ss.AllDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(ctx)
}
```

