#!/bin/bash                                                                   

tag=nakfour/pytorch-fdd-dist:1.2

cp ../../creditcard.csv .
docker build -t $tag .
docker push $tag
rm creditcard.csv
