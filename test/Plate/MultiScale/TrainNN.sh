for idx in 0 1 2 3
do
    julia NN_Train_NNPlatePull.jl $idx & 
done 