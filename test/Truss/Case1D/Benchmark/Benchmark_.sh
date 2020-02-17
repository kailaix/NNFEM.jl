#!/bin/bash
for exp_id in 1 2 3 4 5 6 7 8 9 10
do 
julia NNLearn.jl $1 $2 $3 $4 $5 $exp_id &
done 
wait 