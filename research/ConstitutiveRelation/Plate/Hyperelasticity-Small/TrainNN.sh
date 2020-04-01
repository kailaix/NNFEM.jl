for idx in 0 1 2
do 
julia Train_NNPlatePull.jl $idx &
done 