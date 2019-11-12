
for idx in 0 1
do 
for H in spd_zero_to_H
do
for use_reg in 0 1
do
 julia NN_Train_NNPlatePull.jl $idx $H $use_reg &
#julia NNPreLSfit.jl $idx $H $use_reg &
done
done
done
