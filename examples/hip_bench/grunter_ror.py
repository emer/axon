#!/usr/local/bin/python3

# this is the grunt git-based run tool user-script: https://github.com/emer/grunt
# it must be checked into your local source repository and handles any user commands
# including mandatory submit and results commands.
#
# script is run in the jobs path:
# ~/grunt/wc/server/username/projname/jobs/active/jobid/projname
#
# this sample version includes slurm status and cancel commands and is used for
# launching multiple jobs in parallel (array jobs)

import sys
import os
import subprocess
from subprocess import Popen, PIPE
import glob
from datetime import datetime, timezone
import shutil
import re
import getpass

##############################################################
# key job parameters here, used in writing the job.sbatch

# max number of hours -- slurm will terminate if longer, so be generous
# 2d = 48, 3d = 72, 4d = 96, 5d = 120, 6d = 144, 7d = 168
# full run taking about 60 hrs, so use 72
hours = 2

# memory per CPU, which is only way to allocate on hpc2 (otherwise per node and doesn't fit)
# to tune, look at AveRSS from salloc report
# on a prior job, and divide that by number of tasks
# 7 is max per node x 16 nodes x 2 cpus
# full orig requires 7G @ 2 CPU
mem = "1G" # 3G @ 2 thr is minimum, 5G reserves a node and is sig faster

# number of mpi "tasks" (procs in MPI terminology)
tasks = 1

# number of cpu cores (threads) per task
cpus_per_task = 1

# how to allocate tasks within compute nodes
# cpus_per_task * tasks_per_node <= total cores per node
tasks_per_node = 1

# qos is the queue name
qos = "oreillylab"

# qos short is short name for queue if name is cutoff
qos_short = "oreillyl"

# in other cases, you might have to specify a partition
# partition = "low"

# default array settings, settings of runniing the same job multiple times in parallel. E.g. 10 runs in parallel
array = "0-9"

# number of emergent runs per arrray job 
runs = 1

##############################################################
# main vars

# grunt_jobpath is current full job path:
# ~/grunt/wc/server/username/projname/jobs/active/jobid/projname
grunt_jobpath = os.getcwd()

# grunt_user is user name (note: this is user *on the server*)
grunt_user = getpass.getuser()
# print("grunt_user: " + grunt_user)

# grunt_proj is the project name
grunt_proj = os.path.split(grunt_jobpath)[1]

# grunt_jobid is the jobid assigned to this job in grunt
grunt_jobid = os.path.split(os.path.split(grunt_jobpath)[0])[1]


##############################################################
# utility functions

def write_string(fnm, stval):
    with open(fnm,"w") as f:
        f.write(stval + "\n")

def write_string_apend(fnm, stval):
    with open(fnm,"a") as f:
        f.write(stval + "\n")

def read_string(fnm):
    # reads a single string from file and strips any newlines -- returns "" if no file
    if not os.path.isfile(fnm):
        return ""
    with open(fnm, "r") as f:
        val = str(f.readline()).rstrip()
    return val

def read_strings(fnm):
    # reads multiple strings from file, result is list and strings still have \n at end
    if not os.path.isfile(fnm):
        return []
    with open(fnm, "r") as f:
        val = f.readlines()
    return val

def read_strings_strip(fnm):
    # reads multiple strings from file, result is list of strings with no \n at end
    if not os.path.isfile(fnm):
        return []
    with open(fnm, "r") as f:
        val = f.readlines()
        for i, v in enumerate(val):
            val[i] = v.rstrip()
    return val

def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)    

def timestamp_local(dt):
    # returns a string of datetime object in local time -- for printing
    return utc_to_local(dt).strftime("%Y-%m-%d %H:%M:%S %Z")

def timestamp_fmt(dt):
    # returns a string of datetime object formatted in standard timestamp format
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def parse_timestamp(dtstr):
    # returns a datetime object from timestamp-formatted string, None if not properly formatted
    try:
        dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S %Z")
    except ValueError as ve:
        # print(str(ve))
        return None
    return dt

def timestamp():
    return timestamp_fmt(datetime.now(timezone.utc))

def read_timestamp(fnm):
    # read timestamp from file -- returns None if file does not exist or timestamp format is invalid
    if not os.path.isfile(fnm):
        return None
    return parse_timestamp(read_string(fnm))

def read_timestamp_to_local(fnm):
    # read timestamp from file -- if can be converted to local time, then do that, else return string
    if not os.path.isfile(fnm):
        return ""
    dstr = read_string(fnm)
    dt = parse_timestamp(dstr)
    if dt == None:
        return dstr
    return timestamp_local(dt)
    
# write_sbatch_headers defines the sbatch parameters shared accross startup,
# cleanup and main array jobs
def write_sbatch_header(f):
    f.write("#SBATCH --job-name=" + grunt_proj + "_" + grunt_jobid + "\n")
    f.write("#SBATCH --mem-per-cpu=" + mem + "\n")
    f.write("#SBATCH --time=" + str(hours) + ":00:00\n") 
    f.write("#SBATCH --ntasks=" + str(tasks) + "\n")
    f.write("#SBATCH --cpus-per-task=" + str(cpus_per_task) + "\n")
    f.write("#SBATCH --ntasks-per-node=" + str(tasks_per_node) + "\n")
    # f.write("#SBATCH --qos=" + qos + "\n")
    # f.write("#SBATCH --partition=" + partition + "\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write("#SBATCH --mail-user=" + grunt_user + "\n")
    # these might be needed depending on environment in head node vs. compute nodes
    # f.write("#SBATCH --export=NONE\n")
    # f.write("unset SLURM_EXPORT_ENV\n")
    
# This job is a singelton job before launching the main arrray job used to setup the array job
# such as compiling the code, or other setup work that should only run once per array job
def write_sbatch_setup():
    args = " ".join(read_strings_strip("job.args"))
    f = open('job.setup.sbatch', 'w')
    f.write("#!/bin/bash -l\n")  # -l = login session, sources your .bash_profile
    f.write("#SBATCH --output=job.%A.setup.out\n")
    # f.write("#SBATCH --error=job.%A.setup.err\n")
    write_sbatch_header(f)

    ####################################################################
    ####### This is the code that sets up the job such as compiling.
    ####### This is custom to the job you are running

    f.write("date -u '+%Y-%m-%d %T %Z' > job.start\n")
    f.write("go build -mod=mod\n")  # add anything here needed to prepare code.

    f.flush()
    f.close()

# write_sbatch_array writes out the job.sbatch containing the actual main array job.
def write_sbatch_array(setup_id):
    args = " ".join(read_strings_strip("job.args"))
    f = open('job.sbatch', 'w')
    f.write("#!/bin/bash -l\n")  # -l = login session, sources your .bash_profile
    f.write("#SBATCH --array=" + array + "\n")
    f.write("#SBATCH --output=job.%A_%a.out\n")
    # f.write("#SBATCH --error=job.%A_%a.err\n")
    # Only run the main array job once the setup job has completed
    f.write("#SBATCH --dependency=afterany:" + str(setup_id) + "\n")
    write_sbatch_header(f)

    ####################################################################
    ####### This is the code that lauches the actual main work job.
    ####### This is custom to the job you are running

    f.write("echo $SLURM_ARRAY_JOB_ID\n")
    f.write("\n\n")
    f.write("srun ./" + grunt_proj + " --nogui --run $SLURM_ARRAY_TASK_ID --runs $((SLURM_ARRAY_TASK_ID + " +  str(runs) +")) " +  args + "\n")
    f.flush()
    f.close()

# write_sbatch_cleanup runs for cleanup once all jobs of the main arrray job have completed
def write_sbatch_cleanup(array_id):
    args = " ".join(read_strings_strip("job.args"))
    f = open('job.cleanup.sbatch', 'w')
    f.write("#!/bin/bash -l\n")  # -l = login session, sources your .bash_profile
    f.write("#SBATCH --output=job.%A.cleanup.out\n")
    # f.write("#SBATCH --error=job.%A.cleanup.err\n")
    f.write("#SBATCH --dependency=afterany:" + str(array_id) + "\n")
    write_sbatch_header(f)

    ####################################################################
    ####### This is the code that runs as a singleton once all array jobs have completed.
    ####### This can be used to collate results or used for other cleanup work
    ####### This is custom to the job you are running

    ### Make sure to write the job.end file to let grunt know that the job has completd
    f.write("date -u '+%Y-%m-%d %T %Z' > job.end\n")

    f.flush()
    f.close()

def sbatch_submit(sbatch_fn):
    try:
        result = subprocess.check_output(["sbatch",sbatch_fn])
    except subprocess.CalledProcessError:
        print("Failed to submit " + sbatch_fn + " script")
        return
    prog = re.compile('.*Submitted batch job (\d+).*')
    result = prog.match(str(result))
    slurmid = result.group(1)
    if os.path.isfile('job.slurmid'):
        write_string_apend("job.slurmid", slurmid)
    else:
        write_string("job.slurmid", slurmid)
    return slurmid


def submit():
    if os.path.isfile('job.sbatch'):
        print("Error: job.sbatch exists -- attempt to submit job twice!")
        return
    write_sbatch_setup()
    slurmid_setup = sbatch_submit("job.setup.sbatch")

    write_sbatch_array(slurmid_setup)
    slurmid_array = sbatch_submit("job.sbatch")

    write_sbatch_cleanup(slurmid_array)
    slurmid_cleanup = sbatch_submit("job.cleanup.sbatch")
    
    print("submitted successfully -- slurm job id: " + slurmid_array)
                                                            
    print("status: " + stat)
    write_string("job.status", stat)
    
def results():
    # important: update this to include any results you want to add to results repo
    print("\n".join(glob.glob('*.tsv')))
    print("\n".join(glob.glob('*.csv')))

def status():
    slids = read_strings_strip("job.slurmid")
    stat = "NOSLURMID"
    if len(slids) == 0 or slids == None:
        print("No slurm id found -- maybe didn't submit properly?")
    else:
        if len(slids) == 1:
            print("slurm id to stat: ", slid)
            result = ""
            try:
                result = subprocess.check_output(["squeue","-j",slid,"-o","%T"], universal_newlines=True)
            except subprocess.CalledProcessError:
                print("Failed to stat job")
            res = result.splitlines()
            if len(res) == 2:
                stat = res[1].rstrip()
            elif len(res) > 2: 
                stati = {} 
                for status in res: 
                    if status == "STATE": 
                        continue 
                    if status in stati.keys(): 
                        stati[status] = stati[status] + 1 
                    else: 
                        stati[status] = 1 
                    stat = "" 
                    for status in stati: 
                        stat = stat + status + str(stati[status]) + " " 
            else: 
                stat = "NOTFOUND"
        else:
            if len(slids) == 3:
                try:
                    result = subprocess.check_output(["squeue","-j",slids[0],"-o","%T"], universal_newlines=True)
                except subprocess.CalledProcessError:
                    print("Failed to stat job")
                res = result.splitlines()
                if len(res) == 2:
                    if res[1].rstrip() == "RUNNING":
                        stat = "COMPILING"
                    else:
                        stat = res[1].rstrip()
                else:
                    try:
                        result = subprocess.check_output(["squeue","-j",slids[1],"-o","%T"], universal_newlines=True)
                    except subprocess.CalledProcessError:
                        print("Failed to stat job")
                    res = result.splitlines()
                    if len(res) == 2:
                        stat = res[1].rstrip()
                    elif len(res) > 2: 
                        stati = {} 
                        for status in res: 
                            if status == "STATE": 
                                continue 
                            if status in stati.keys(): 
                                stati[status] = stati[status] + 1 
                            else: 
                                stati[status] = 1 
                                stat = "" 
                        for status in stati: 
                            stat = stat + status + str(stati[status]) + " "
                    else:
                        try:
                            result = subprocess.check_output(["squeue","-j",slids[2],"-o","%T"], universal_newlines=True)
                        except subprocess.CalledProcessError:
                            print("Failed to stat job")
                        res = result.splitlines()
                        if len(res) == 2:
                            if res[1].rstrip() == "RUNNING":
                                stat = "CLEANUP"
                            else:
                                stat = res[1].rstrip()
                        else:
                            stat = ""
        
    print("status: " + stat)
    write_string("job.status", stat)
    
def cancel():
    write_string("job.canceled", timestamp())
    slids = read_strings_strip("job.slurmid")
    for slid in slids:
        if slid == "" or slid == None:
            print("No slurm id found -- maybe didn't submit properly?")
            return
        print("canceling slurm id: ", slid)
        try:
            result = subprocess.check_output(["scancel",slid])
        except subprocess.CalledProcessError:
            print("Failed to cancel job")
            return

# custom command to get info on cluster        
def queue():
    result = ""
    try:
        result = subprocess.check_output(["sinfo"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run sinfo")
    res = result.splitlines()
    ts = timestamp_local(datetime.now(timezone.utc))
    qout = ["queue at: " + ts + "\n", "sinfo on: " + qos + "\n"]
    for r in res:
        # if qos in r:   # filter by qos
        qout.append(r)
            
    # qout.append("\nsqueue -q " + qos + "\n")
    # try:
    #     result = subprocess.check_output(["squeue", "-q", qos], universal_newlines=True)
    # except subprocess.CalledProcessError:
    #     print("Failed to run squeue")
    # res = result.splitlines()
    # for r in res:
    #     if qos_short in r:  # doesn't fit full qos
    #         qout.append(r)

    qout.append("\nsqueue -u " + grunt_user + "\n")
    try:
        result = subprocess.check_output(["squeue", "-u", grunt_user], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run squeue")
    res = result.splitlines()
    for r in res:
        # if qos_short in r:  # doesn't fit full qos
        qout.append(r)
    
    write_string("job.queue", "\n".join(qout))

# etcat runs etcat on given file type e.g., "_epc"
def etcat(ftype):
    grpath = os.path.join("gresults", sys.argv[2])
    fls = os.path.join(grpath, "*" + ftype + ".tsv")
    fl = glob.glob(fls)
    if len(fl) == 0:
        print("no epoch files in path: " + grpath)
        exit(0)
    allout = "_all" + ftype[1:]
    avgout = "_avg" + ftype[1:]
    alloutf = ""
    avgoutf = ""
    for f in fl:
        if not "_00" in f: # find base one without run number
            alloutf = f.replace(ftype, allout)
            avgoutf = f.replace(ftype, avgout)
    try:
        result = subprocess.check_output(["etcat", "-avg", "-o", avgoutf] + fl, universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run etcat")
    try:
        result = subprocess.check_output(["etcat", "-d", "-o", alloutf] + fl, universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run etcat")
    os.chdir(grpath)
    try:
        result = subprocess.check_output(["git", "add"] + glob.glob("*" + allout) + glob.glob("*" + avgout), universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run git add")
    try:
        result = subprocess.check_output(["git", "commit", "-am", "etcat data"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run git commit")
    try:
        result = subprocess.check_output(["git", "push"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run git push")
    
if len(sys.argv) < 2 or sys.argv[1] == "help":
    print("\ngrunter.py is the git-based run tool extended run script\n")
    print("supports the following commands:\n")
    print("submit\t submit job to slurm")
    print("results\t list job results")
    print("status\t get current slurm status")
    print("cancel\t tell slurm to cancel job")
    print()
    exit(0)
    
cmd = sys.argv[1]

if cmd == "submit":
    submit()
elif cmd == "results":
    results()
elif cmd == "status":
    status()
elif cmd == "cancel":
    cancel()
elif cmd == "queue":
    queue()
elif cmd == "etcat":
    # etcat("_epc")
    etcat("_run")
else:
    print("grunter.py: error: cmd not recognized: " + cmd)
    exit(1)
exit(0)


