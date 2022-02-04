for force_scale in 5.0 6.0 8.0 10.0 20.0
do 
for tid in 106 206 300
do 
julia NN_Test_NNPlatePull.jl $force_scale $tid &
done 
done 