This tutorial will teach you how to run Tensorflow models on TensorPort. We will go
from a simple mnist example to a more complex self-steering car demonstration.

1.1) Simple mnist (google Tensorflow demo) - no dataset setup

1.2) Deep dive into dataset setup

2.1) CNN mnist - single-node

2.2) Make that CNN distributed

3) [Self - steering car here](https://github.com/malomarrec/tensorport-self-driving-demo)



# Setup

You will need an account on https://www.tensorport.com/, PIP, Git, and GIT LFS

Start off by getting the TensorPort CLI:

```bash
pip install tensorport
```

You can now login the command: `tport login`.

Then clone this repo on your computer:

``` bash
$ git clone https://github.com/malomarrec/tensorport-self-driving-demo.git
```

`cd` into that repo.

``` bash
$ cd tensorport-self-driving-demo/
```

# 1.1 Simple mnist

For this step we will run `mnist.py`. This is a simple one layer network.
The data will be downloaded automatically to the directory specified by the code
`<ROOT_PATH_TO_LOCAL_DATA>/mnist-dataset` and the outputs and logs in
`PATH_TO_LOCAL_LOGS`.

## Run locally

You can run this code in your local environment with:
Note:  you need to have Tensorflow 1.0.0 or later installed in that environment.


```bash
$ python mnist.py
```
You should see loss values starting to appear.

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
run your code, you should see the following output.

```
Running single-node locally
...
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.080547988]
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.15762429]
I tensorflow/core/kernels/logging_ops.cc:79] Loss = [0.097241819]
```

If this worked you can be confident that your scrip is bug free and all 
requirements will be install correctly when it is run remotely.
I almost always try this before running any large project remotely. 

### Distributed

To simulate a distributed environment run:


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
default of 1. Don't go too crazy on the number of workers, this is still running on
your local machine.

```
-> Select number of paramter servers [1]:
```
The parameter server will handle storage of your model parameters and handle
the I/O to workers, 1 should be enough.


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

Note that if you like all of this can be passed as a single command:

```bash
tport run --local --distributed --worker-replicas 2 --ps-replicas 1 --requirements 1 --module mnist
```


## Run on TensorPort

Our code is ready to run on TensorPort.

### Create a TensorPort Project

`cd` into the repo

``` bash
$ tport create project
 -> Choose a valid project name:
$ mnist-demo
Creating project mnist-demo .......
Project malo/mnist-demo is created.
Matrix: https://tensorport.com/matrix/malo/mnist-demo

Push the code to TensorPort using 'git push tensorport master'
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

Now pick the python module (this is just the name of the file) you want to run, and leave the package path empty as
the module is located in the root.


```bash
Specify the python module to run [main]: mnist
-> Specify the package path []:```

Then fillout the following questions just like before.

```
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

In the previous part we've left out how to setup a dataset. The data was
small enough and could be automatically downloaded by the script. In most cases
however the data is large enough that we don't want to do this. 

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

We can now create the TensorPort dataset:

```bash
$ tport create dataset
 -> Choose a valid dataset name: mnist-dataset
Creating dataset test-comma ......
Dataset malo/test-comma is created.
Matrix: https://tensorport.com/matrix/malo/test-comma
```

Now let's push the code to TensorPort:

``` bash
$ git push tensorport master
```

This is not the best way to handle data for large files (over 100MB). We will use Git LFS
to speed things up in part 3 of this tutorial (self-steering car).

In the code (mnist.py) we need to update the location of the data (if you don't do it, the
code will still run but it will download the dataset anew). You need to update the
`data_dir` flag. To make sure we can use the same code on TensorPort and locally
we use a small wrapper `get_data_path(dataset_name, local_root, local_repo, path)`
with:
- `dataset_name`: the dataset name on TensorPort
- `local_root`: the dataset root locally (here `ROOT_PATH_TO_LOCAL_DATA`)
- `local_repo`: the name of the local data repo (can be = `dataset_name`)
- `path`: path to the files within the dataset, if you have a complicated tree

Locally, the wrapper returns: `local_root/local_repo/path`, 
In this case locally it will return sometihng like: ``~/Documents/data2/mnist` as `path =''`.
On TensorPort, the wrapper returns:
`/data/dataset_name/path`.

Let's change `ROOT_PATH_TO_LOCAL_DATA` to `~/Documents/data2/mnist` in `mnist.py`.

The `dataset_name` is always `<username_of_owner>/<dataset_name>`. You are going
to create the dataset now, so you will be able to update
`<username_of_owner>/<dataset_name>` accordingly (in my example, `malo/mnist-dataset`).

The tree on TensorPort will be:
data
  | malo
    | mnist-dataset
       | t10k-labels-idx1-ubyte.gz
       | t10k-images-idx3-ubyte.gz
       | <files>

but we don't need to worry too much about this as our get_data_path function will take care of it.

Now we can:
- Push the updated code to TensorPort
- Run the code as Job on TensorPort

Run Job by using the `tport run` command just as in the previous part, but at the dataset prompt select y to use a dataset and 0 for the most recently created dataset:
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
workflow with `tport watch`.

In the next two steps we will explain more about how to write your script so
that it runs in distributed mode on TensorPort. To make it a little more interesting,
we'll use an CNN model on the mnist dataset.


# 2.1 CNN mnist - single-node
Open mnist_conv_local.py. This code is a CNN version of mnist, that runs only on
a single node. It's there as a point of comparison for the distributed code.
It has nice TensorBoard summaries that you can visualize on the platform.

# 2.1 CNN mnist - distributed

To run a distributed model, we use TensorFLow's [`MonitoredTrainingSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession).
A TensorPort utility take care of all the node set up.

Open mnist_conv_distributed.py.

From line 27, the code is a TensorPort snippet (that you can reuse) that configures
the distributed training. Let's go over it block by block.

```python
try:
  job_name = os.environ['JOB_NAME']
  task_index = os.environ['TASK_INDEX']
  ps_hosts = os.environ['PS_HOSTS']
  worker_hosts = os.environ['WORKER_HOSTS']
except:
  job_name = None
  task_index = 0
  ps_hosts = None
  worker_hosts = None
```

This block recovers the environment of the node the script is running on, or
no environment if it is running locally. That will be passed to the MonitoredTrainingSession
so that only the Master will save checkpoints and parameters servers are configured
properly.

The following FLAGS section was covered in the previous part.

```python
def device_and_target():
  # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
  # Don't set a device.
  if FLAGS.job_name is None:
    print("Running single-machine training")
    return (None, "")

  # Otherwise we're running distributed TensorFlow
  print("Running distributed training")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
    raise ValueError("Must specify an explicit `ps_hosts`")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
    raise ValueError("Must specify an explicit `worker_hosts`")

  cluster_spec = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
  })
  server = tf.train.Server(
      cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  return (
      tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster_spec),
      server.target,
  )
```
In a distributed environment, `device_and_target()`` creates the Cluster config.
`device, target = device_and_target()` gets the node environment. We will use it
when we set up the model graph:

```python
with tf.device(device):
  # define here the computation graph
  # placeholders
  # nodes
  # loss
```

and as arguments to the training session:

```
tf.train.MonitoredTrainingSession(
     master=target, #
     is_chief=(FLAGS.task_index == 0), #only the chief will save checkpoints
     checkpoint_dir=FLAGS.log_dir) as sess:

     #run session
```


Now you should be good to go and run this on TensorPort :). 
