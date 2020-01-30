#!/bin/bash



julia NNLearn.jl 0 spd_Chol_Orth 0 &
julia NNLearn.jl 1 spd_Chol_Orth 0 &
julia NNLearn.jl 2 spd_Chol_Orth 0 &


wait
