#!/bin/bash
aws s3 sync s3://sagemaker-scf/$1/models/best model/trained
mv model/trained/variables/variables.part-00000-of-00001.data-00000-of-00001 model/trained/variables/variables.data-00000-of-00001