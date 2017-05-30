# How to use surfsara
In this document I will briefly explain how to use cartesius in surfsara.

## SSH client
You will need an SSH client, I assume that everyone is familiar with these tool so I will just mention that if you are using windows you may want to use Cygwin, Bash on Ubuntu on Windows (this is the one I would recommend) or Putty. If you are in linux just use the ssh command directly.

## Get access to the login node
Surfsara has a whitelisted ip list from which users can access to their servers. Big possibility your home ip is not included in this list, but the university one is. I know two solutions for getting access to their servers:
- Open your ssh client and type `ssh -t vgarciacazorla@lilo.science.ru.nl ssh gdemo013@cartesius.surfsara.nl` (with your uni and cartesius usernames)
- Use a VPN to connect to uni and then just type `ssh gdemo013@cartesius.surfsara.nl`

After any of this steps just type your password and you will be already in the login node of cartesius.
## Run stuff
The login node is only used to prepare averything (code, data etc.) but to run the actual experiments we need to get access to a node with gpus. So the way we do that is creating a file (running ` nano batch_job` for example) inside the login node with the following content:
```
#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --time=01:00:00
#SBATCH --job-name=clions

module load cuda/8.0.44 
module load cudnn/8.0-v5.1
module load gcc/5.2.0
module load python/3.5.0

srun python3 main.py
```
The first lines speak for themselves and are related with the resources that we want to alocate, [here is the documentation](https://userinfo.surfsara.nl/systems/cartesius/usage/batch-usage) about its parameters and common usage. The example I wrote only books 1 hour so after that the process will be killed.

The next lines with `module load` indicates the stuff will be needed for the experiment which in our case are the gpu libraries, python and so on.

Once our `batch_job` file is ready we have to execute it like this:
`sbatch batch_job`
This will start the whole process (and start consuming your credits).

#Managing batch jobs
With `sbatch batch_job` we have submitted a job. We can submit more than one if we want. If we type now `squeue -u $(whoami)` we will see all our submitted batches. In the last column we can see the name of the node wherein each batch job is running (`gcnXX`, where `XX` are numbers). Is it possible that although we have submitted our batch job, it hasn't been assigned any node yet (instead of `gcnXX` it says `(None)`). 
```
[gdemo013@int2 src]$ squeue -u gdemo013
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           3176889       gpu   resnet gdemo013 PD       0:00      1 (None)
           3176890       gpu xception gdemo013 PD       0:00      1 (None)
           3176888       gpu    vgg16 gdemo013  R       0:36      1 gcn18

```

This will happen if we want to allocate a lot of hours for example. Sooner or later our batch job will be given a node, so it is recommended to run `watch -n1 squeue -u $(whoami)` so it refreshes automatically and we will know when our job is given a node. For this matter, as we can see in the [documentation](https://userinfo.surfsara.nl/systems/cartesius/usage/batch-usage), if we set `gpu_short` in the `--partition` (as we have done in the `batch_job` example file) we can only "book" one hour. This is because that mode is meant for test runs; if we want more we will have to set it to `gpu` instead.

## Monitoring the experiment
So now we are still in the login node but the experiment is running in a different one. We may want to access that node in order to check how things are going, like monitoring GPU usage via the `watch -n1 nvidia-smi` command. In order to do so we have to run `squeue -u $(whoami)`. I four job has been assigned the node `gcn34` for example,we will have to type `ssh gcn34`, input our password and we will be in the allocated node.
What if we just want to see the printouts of our script? Although the experiment is running you don't see the output directly, even if you are in the node where it is running.  However you see that some files are being generated named like `slurm-3130333.out` for example. These files contains the output of the experiment so to visualize each of them you can simply do `cat slurm-3130333.out`. This should be enough to monitoring it but I also believe that there must be a more confortable way to deal with it, I will update this when I find something.

## That's it
If something is not clear tell me and I will try to explain better or elaborate more. Also if you think that it would be interesting to include more explanations just tell me ;)

