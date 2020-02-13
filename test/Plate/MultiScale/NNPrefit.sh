#!/bin/bash

for idx in  1 2 
do
    julia NNPreLSfit.jl $idx & 
done 

wait
