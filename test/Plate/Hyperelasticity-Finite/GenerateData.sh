#!/bin/bash

force_scale=1.0

julia Data_NNPlatePull.jl 100 $force_scale 2 2 &
julia Data_NNPlatePull.jl 101 $force_scale 2 2 &
julia Data_NNPlatePull.jl 102 $force_scale 2 2 &
julia Data_NNPlatePull.jl 103 $force_scale 2 2 &
julia Data_NNPlatePull.jl 104 $force_scale 2 2 &
julia Data_NNPlatePull.jl 105 $force_scale 2 2 &
julia Data_NNPlatePull.jl 106 $force_scale 2 2 &
julia Data_NNPlatePull.jl 200 $force_scale 2 2 &
julia Data_NNPlatePull.jl 201 $force_scale 2 2 &
julia Data_NNPlatePull.jl 202 $force_scale 2 2 &
julia Data_NNPlatePull.jl 203 $force_scale 2 2 &
julia Data_NNPlatePull.jl 204 $force_scale 2 2 &
julia Data_NNPlatePull.jl 205 $force_scale 2 2 &
julia Data_NNPlatePull.jl 206 $force_scale 2 2 &
julia Data_NNPlatePull.jl 300 $force_scale 2 2 &



wait
