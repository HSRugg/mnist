This demo will teach you how to run Tensorflow models on TensorPort. We will go
from a simple mnist demo to a more complex self-steering car demo.

1.1) Simple mnist (google Tensorflow demo) - no dataset setup
1.2) Deep dive into dataset setup
2.1) CNN mnist - single-node
2.2) Make that CNN distributed
3) [Self - steering car here](https://github.com/malomarrec/tensorport-self-driving-demo)



# Setup

You need an account on https://www.tensorport.com/

Start off by getting the TensorPort CLI:

```bash
pip install tensorport
```

You can now login through `tport login`.

Then clone this repo on your computer:

``` bash
$ git clone https://github.com/malomarrec/tensorport-self-driving-demo.git
```

`cd` into that repo.

# 1.1 Simple mnist

For this step we will run `mnist.py`. This is a simple one layer network.
The data will be downloaded automatically in what you specify in the code in
`<ROOT_PATH_TO_LOCAL_DATA>/mnist-dataset` and the outputs and logs in
`PATH_TO_LOCAL_LOGS`.

## Run locally

You can run that code in your local environment with:

```bash
$ python mnist.py
```
You should see loss values starting to appear.
Note that you need to have Tensorflow 1.0.0 or later installed in that environment.

## Simulate a distributed environment locally

Before running on TensorPort, we want to make sure that the code runs in a
distributed fashion. To speed up that process, the TensorPort CLI can
simulate a remote and distributed TensorPort environment on your computer.

### Single-node

Let's first run a single-node TensorPort environment. This will create and empty
virtualenv, and optionally install the requirements specified in a requirements
file.

```bash
$ tport run --local --single-node
-> Specify the python module to run [main]:
$ mnist
-> Requirements file not specified. You can
   0- Specify a requirements file
   1- Run without installing any requirments
   2- Use your current environemt (not recommended).
$ ...
```

Here we don't need any specific library, so we can just use a basic virtualenv
with Tensorflow only. Select 1. The utility will install a basic virtualenv and
run your code.

```
Running single-node locally
Using real prefix '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7'
New python executable in /Users/malo/.tensorport/env/test/bin/python2.7
Also creating executable in /Users/malo/.tensorport/env/test/bin/python
...
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.080547988]
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.15762429]
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.097241819]
```
That step can be useful to debug a script and make sure all requirements will
be installed remotely.

### Distributed

To simulate a distributed environment, run:


```bash
$ tport run --local --distributed
-> Specify the python module to run [main]:
$ mnist
-> Requirements file not specified. You can
   0- Specify a requirements file
   1- Run without installing any requirments
   2- Use your current environemt (not recommended).
$ ...
```

```
-> Select number of workers [1]:
```

Enter the number of workers you want to emulate, or press enter to get the
default (1). Don't go too crazy on the number of workers, this is running on
your local machine!

```
-> Select number of paramter servers [1]:
```
The parameter server will handle storing your model parameters and handle
I/O to workers. 1 should be enough.


```
 -> Specify the python module to run [main]:
 $ mnist
 -> Requirements file not specified. You can
    0- Specify a requirements file
    1- Run without installing any requirments
    2- Use your current environemt (not recommended).
 $ 1
```

Note: This code will work only on Tensorflow 1.0.0 (ot will fail on more recent
versions in distributed mode).

Note that all of this can be passed in one single command:

```bash
tport run --local --distributed --worker-replicas 2 --ps-replicas 1 --requirements 1 --module mnist
```


## Run on TensorPort

Our code is ready to run on tensorport.

### Create a TensorPort Project

`cd` into the repo then run

``` bash
$ tport create project
 -> Choose a valid project name:
$ mnist-demo
Creating project mnist-demo .......
Project malo/mnist-demo is created.
Matrix: https://tensorport.com/matrix/malo/mnist-demo

Push the code to tensorport using 'git push tensorport master'
```

The project has been create on TensorPort's git. Let's push the code:

``` bash
$ git push tensorport master
```

You should see:
```
Counting objects: 30, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (29/29), done.
Writing objects: 100% (30/30), 180.55 KiB | 0 bytes/s, done.
Total 30 (delta 14), reused 0 (delta 0)
To https://git.tensorport.com/malo/mnist-demo.git
 * [new branch]      master -> master
 ```

Our code is uploaded, we're ready to run.

```bash
$ tport run
-> Job name [cool-shape-270]:
```
Select the project:
```
-> Select the project that you want to use or specify --project parameter.
0 | malo/mnist-demo | Commits: 8
1 | malo/self-driving-car-1 | Commits: 20
$ 0
```

Now select the most recent commit (0):
``` bash
-> Specify the commit for the job, select 0 for latest commit:
|   | Commit # | Commit message                                        | date                |
|---|----------|-------------------------------------------------------|---------------------|
| 0 | d7d1ad23 | Minor fixes on paths                                  | 2017-07-14T11:42:38 |
| 1 | 0aa4592e | Start work on doc                                     | 2017-07-14T11:26:39 |
| 2 | 166c44ee | Fix issue with stringy task index (move to flag to do | 2017-07-13T18:33:11 |
|   |          | the casting) + add accuracy out                       |                     |
...
Select commit from [0 - 7] [0]: 0
```

Now pick the python module you want to run, and leave the package path empty as
the module is in the root.
```bash
Specify the python module to run [main]: mnist
-> Specify the package path []:
-> Do you want to add a dataset to the job? Y/n [y]: n
-> Specify the requirements file [requirements.txt]:
-> Select a TensorFlow version from
0 | 1.0.0
1 | 1.1.0
2 | 1.2.0
$ 0
-> Select number of workers [1]: 2
-> Select instance type of workers from:
| # | Name       | CPU | GPU  | Memory(GiB) |
|---|------------|-----|------|-------------|
| 0 | c4.2xlarge | 8   | None | 15          |
| 1 | p2.xlarge  | 4   | 1    | 61          |
| 2 | t2.small   | 1   | None | 2           |
-> Select instance type of workers from [0 - 2] [0]: 1
p2.xlarge selected for workers' instance type
-> Select number of paramter servers [1]: 1
-> Select instance type of parameter servers from:
| # | Name       | CPU | GPU  | Memory(GiB) |
|---|------------|-----|------|-------------|
| 0 | c4.2xlarge | 8   | None | 15          |
| 1 | p2.xlarge  | 4   | None | 61          |
| 2 | t2.small   | 1   | None | 2           |
-> Select instance type of parameter servers from [0 - 2] [0]: 0
c4.2xlarge selected for parameter servers' instance type

-> Select the time limit for the job [48h0m]:
Job will run for 48 hours and 0 minutes

-> Job description [optional] []:
Job crimson-fog-711 created successfully.

You can see your job in matrix : https://tensorport.com/matrix/malo/mnist-demo.

Job 'crimson-fog-711' started.
```







Your job is created and running. You can monitor the status of your jobs with:





You can watch the progress of the resources
requirements and data cloning by running:


Log on to Matrix to see the outputs and add the project to TensorBoard.







# 1.2 Deep dive into dataset setup

In the previous part, we've left aside how to setup a dataset, as the data was
small enough and could be automatically downloaded by the script. In most cases
however the data is large enough that we don't want to have to move it around
too much.

In that part we'll take the same example and add dataset handling.

If you've run the previous script, data should have been downloaded in
`<ROOT_PATH_TO_LOCAL_DATA>/mnist-dataset`, in my case `~/Documents/data`


Copy that data somewhere else:, for example in ~/Documents/data/`.
Right now the download utility creates an `mnist` folder, so the local tree
looks like this:
~/Documents
  | data
    | mnist-dataset
      | mnist
         | t10k-labels-idx1-ubyte.gz
         | t10k-images-idx3-ubyte.gz
         | <files>



#

For this example, we'll pretend that we are creating a dataset from scratch.
Copy the dataset somewhere else, for example in `~/Documents/data2`.
Now the local tree is:

~/Documents
  | data2
    | mnist-dataset
      | mnist
         | t10k-labels-idx1-ubyte.gz
         | t10k-images-idx3-ubyte.gz
         | <files>

Remove the dataset from the old location.

`cd` into the mnist repo, and initialize a git repo there.

```bash
$ git init
$ git add *
$ git commit -m "First commit: add data"
```

Once this is done, we can create the TensorPort dataset:

```bash
$ tport create dataset
 -> Choose a valid dataset name: mnist-dataset
Creating dataset test-comma ......
Dataset malo/test-comma is created.
Matrix: https://tensorport.com/matrix/malo/test-comma
```

Now let's push the code to tensorport:

``` bash
$ git push tensorport master
```

This is not the best way to handle data for large files. We will use Git LFS
to speed things up in part 3) of this tutorial (self-steering car).

In the code, we need to update the location of the data (if you don't do it, the
code will still run but it will download the dataset anew). You need to update the
`data_dir` flag. To make sure we can use the same code on TensorPort and locally
we use a small wrapper `get_data_path(dataset_name, local_root, local_repo, path)`
with:
- `dataset_name`: the dataset name on TensorPort
- `local_root`: the dataset root locally (here `ROOT_PATH_TO_LOCAL_DATA`)
- `local_repo`: the name of the local data repo (can be = `dataset_name`)
- `path`: path to the files within the dataset, if you have a complicated tree

Locally, that wrapper returns:
`local_root/local_repo/path`, here: ``~/Documents/data2/mnist` as `path =''`.
On TensorPort, that function returns:
`/data/dataset_name/path`.

Let's change `ROOT_PATH_TO_LOCAL_DATA` to `~/Documents/data2/mnist` in `mnist.py`.

The `dataset_name` is always `<username_of_owner>/<dataset_name>`. You are going
to create the dataset now, so you will be able to update
`<username_of_owner>/<dataset_name>` accordingly (in my example, `malo/mnist-dataset`).

The tree on tensorport will be:
data
  | malo
    | mnist-dataset
       | t10k-labels-idx1-ubyte.gz
       | t10k-images-idx3-ubyte.gz
       | <files>

but we don't need to worry too much about this as get_data_path will take care of it.

Now we can:
- push the updated code to tensorport
- run a job

Run using `tport run` just as in the previous part, but at the dataset prompt:
```
 -> Do you want to add a dataset to the job? Y/n [y]: y
 -> Select a dataset
0 | malo/mnist
1 | malo/comma-train-only
 -> Select dataset to use from [0 - 1] [0]:
$ 0
 -> Do you want to add another dataset to the job? y/N [n]:
 $ n
 -> Specify the requirements file [requirements.txt]:
-> Select a TensorFlow version from
0 | 1.0.0
1 | 1.1.0
2 | 1.2.0
$ 0
...[same process as in the first part]
```

Now our job started, as we can see with `tport pulse`, and we can monitor the
workflow with `tport watch`
