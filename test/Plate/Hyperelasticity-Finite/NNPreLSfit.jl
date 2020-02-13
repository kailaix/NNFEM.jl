#!/bin/bash


julia NNPreLSfit.jl 0 &
julia NNPreLSfit.jl 1 &
julia NNPreLSfit.jl 2 &



wait
