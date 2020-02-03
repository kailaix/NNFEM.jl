#!/bin/bash
force_scale=5.0

for tid in 100 101 102 103 104 200 201 202 203 204 300
do 
    julia Data_NNPlatePull.jl $tid $force_scale 2 2 &
done

wait