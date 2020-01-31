#!/bin/bash



# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &

julia NNPreLSfit.jl 0 spd_Chol_Orth 0 &
julia NNPreLSfit.jl 1 spd_Chol_Orth 0 &
julia NNPreLSfit.jl 2 spd_Chol_Orth 0 &


#julia NNLearn.jl 0 spd_Chol_Orth 0 &
#julia NNLearn.jl 1 spd_Chol_Orth 0 &


# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &

wait
