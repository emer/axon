# simscripts

These are [Cogent Shell](https://github.com/cogentcore/core/tree/main/shell) (`cosh`) scripts for the [Cogent Numbers](https://github.com/cogentcore/cogent/tree/main/numbers) `DataBrowser` to manage the process of running simulations on a remote cluster.  This framework replaces [grunt](https://github.com/emer/grunt) and incorporates some of its key features, while eliminating the need for any python code, and the extra complexity of the server-side daemon.

A key feature of `cosh` is the ability to transparently run shell commands on a remote host, connected through `ssh`, including using the `scp` protocol to copy files, in addition to the standard capturing of command output to a variable.  _This eliminates the need for any code to be installed on the remote host:_ everything runs from the local client (laptop), greatly simplifying the overall programming logic.

The `databrowser.Browser` GUI widget provides a tabbed file-browser for editing files, viewing tabular data in `table.Table` spreadsheet-like tables, and interactively plotting data.  In addition, custom tables can be created, as done in these scripts, that summarize directories of data (from different simulation runs) with various meta data displayed.  By selecting rows in such tables and running scripts installed on the toolbar, the user can manage the process of submitting and comparing simulation runs.

# Workflow

The general workflow is as follows, assuming a standard [install](#install) has been performed, with a `simdata` symbolic link in the main simulation code directory pointing to the simulation data with these scripts installed.

* Run numbers _from the main sim code directory_ with a new browser:

```sh
> numbers -e databrowser.NewBrowserWindow("simdata")
``` 

* `Jobs` shows current jobs in a Jobs tab

* `Submit` runs a new sim job

* `Status` gets status of any running jobs.  Anything done running gets status of `Finalized` and is no longer updated by Status.  All job metadata is downloaded from host, _but not the Results_ output data, which is `Fetch`ed separately because it may be large and often needs to be consolidated in a particular way, because multiple runs of a job are executed in parallel.

* `Fetch` gets result `.tsv` files from server, consolidating parallel runs into `_allepc.tsv` and `_avgepc.tsv` etc files.  It can be run on running or Finalized jobs.  When run on Finalized, then the status is set to `Fetched` and it is automatically skipped in any future Fetch actions.

* `Results` grabs specific result data files into a `Results` tab, from which further examination and plotting occurs.  This step is necessary because there are typically multiple different types of results files, so you need to select which type you want view.

* `Plot` plots combined data across any selected files in `Results` tab, allowing you to compare them, using the `JobID` as a legend so each job has its own line color.

* `Diff` shows a diff browser for any two selected Jobs, or one selected job vs. the current sim working directory.

# Install



# Example `cosh` code

The following annotated example code demonstrates the key features of the `cosh` language, from the `Status` script. `cosh` automatically detects shell exec lines vs. Go code in an intuitive way, based on various syntactic indicators.  Within a Go context, exec code is explicitly indicated with backticks, and within exec, Go code is surrounded by `{ }` braces.

`@1` specifies the current remote host -- there can be any number of maintained host connections, with unique names -- and `@0` is the local host:

```go
	sj := `@1 cat job.job`  // get the job id, by running cat on remote host @1
	if sstat != "Done" && !force {  // standard Go control logic
		[@1 squeue -j {sj} -o %T >& job.squeue] // [ ] = don't stop on failure
		stat := `@1 cat job.squeue` // get results
   ...
```

The current working directory is maintained and updated on each host (local and remote).  Here is another example of the combination of go and shell exec, including the `scp` command to retrieve files from the remote host:

```go
	jfiles := `@1 /bin/ls -1 job.*` // get all job files
	for _, jf := range cosh.SplitLines(jfiles) { // cosh package has helper functions
		rfn := "@1:" + jf // prefix filename with @1: for remote host, otherwise local
		scp {rfn} {jf}    // { } indicates go expressions within shell exec context
	}
	@0 // switch context back to local host for further processing
```
