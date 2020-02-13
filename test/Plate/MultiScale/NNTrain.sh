#!/bin/bash
 
julia NN_Train_NNPlatePull-2.jl 3 & 
julia NN_Train_NNPlatePull-3.jl 3 & 
julia NN_Train_NNPlatePull-4.jl 3 & 
julia NN_Train_NNPlatePull-5.jl 3 & 

wait
