#!/bin/bash

julia Train_NNPlatePull.jl 0 &
julia Train_NNPlatePull.jl 1 &
julia Train_NNPlatePull.jl 2 &

wait
