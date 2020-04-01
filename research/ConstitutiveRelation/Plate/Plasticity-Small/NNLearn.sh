#!/bin/bash

julia NNLearn.jl 1 &
julia NNLearn.jl 2 &

wait
