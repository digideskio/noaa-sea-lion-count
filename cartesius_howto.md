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
So now we are still in the login node but the experiment is running in a different one. We want to access that node in order to check how things are going, like visualising the printouts of our script for example. In order to do so we have to run `squeue -u $(whoami)`. The output will give you information about the allocated node. Specifically in the last column you will see the name of the node, which usually is something like `gcn34` for example, so now you do `ssh gcn34` and you will be in the allocated node.

## Monitoring the experiment
One in the allocated node, the experiment should be running but you don't see the output directly. However you see that some files are being generated named like `slurm-3130333.out` for example. These files contains the output of the experiment so to visualize each of them you can simply do `cat slurm-3130333.out`. However if you do something like `watch -n1 slurm-3130333.out` it will automatically show its content each 1 second (the file gets updated while the experiment is running). This should be enough to monitoring it but I also believe that there must be a more confortable way to deal with it, I will update this when I find something.

## That's it
If something is not clear tell me and I will try to explain better or elaborate more. Also if you think that it would be interesting to include more explanations just tell me ;)

