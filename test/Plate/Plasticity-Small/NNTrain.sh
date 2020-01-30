<<<<<<< HEAD
#!/bin/bash



# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &




julia NN_Train_NNPlatePull.jl  1 spd_Chol_Orth 0 &
julia NN_Train_NNPlatePull.jl  2 spd_Chol_Orth 0 &

#julia NNPreLSfit.jl 0 spd_Chol_Orth 0 &
#julia NNPreLSfit.jl 1 spd_Chol_Orth 0 &



# Execute jobs in parallel
#julia NN_Train_NNPlatePull.jl 0 spd_Chol_Orth 0 &
#julia NN_Train_NNPlatePull.jl 1 spd_Chol_Orth 0 &

wait
=======
for idx in 0 1 2 
do 
julia NN_Train_NNPlatePull.jl $idx &
done 
>>>>>>> a03560c8aa597b42ed229ef85fd91a9c6f9aa95f
