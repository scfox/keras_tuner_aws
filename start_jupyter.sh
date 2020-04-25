#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
pip install -r src/requirements.txt
jupyter notebook --port 6010
