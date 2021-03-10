#!/bin/bash                                                                   

tag=nakfour/pytorch-fdd:1.3

cp ../../creditcard.csv .
docker build -t $tag .
docker push $tag
rm creditcard.csv
