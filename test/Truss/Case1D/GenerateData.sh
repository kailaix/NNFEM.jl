#!/bin/bash
for tid in 1 2 3 4 5
do 
julia Data_NNTrussPull_Plasticity.jl nn_dummy $tid &
done 

wait
