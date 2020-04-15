#!/bin/bash



# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &


#julia NNPreLSfit.jl 0 spd_Chol_Orth 0 &
#julia NNPreLSfit.jl 1 spd_Chol_Orth 0 &


julia Train_NNPlatePull.jl 0 &
julia Train_NNPlatePull.jl 1 &
julia Train_NNPlatePull.jl 2 &

# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &

wait
