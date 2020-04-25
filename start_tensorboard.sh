#!/bin/bash
echo Using $1 for tensorboard
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
tensorboard --logdir=${1}/logs --port 6012 --bind_all &
