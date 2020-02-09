#!/bin/bash


nntype=piecewise 
for idx in 1 2 3 4
do
for nn_init_id in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do 
julia NNLearn.jl $nntype $idx $nn_init_id &
done
done 

wait


nntype=ae_scaled
for idx in 1 2 3 4
do
for nn_init_id in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do 
julia NNLearn.jl $nntype $idx $nn_init_id &
done
done 

wait


nntype=piecewise2 
for idx in 1 2 3 4
do
for nn_init_id in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do 
julia NNLearn.jl $nntype $idx $nn_init_id &
done
done 

wait



