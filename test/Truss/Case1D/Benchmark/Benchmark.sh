for nn_depth in 5 10 20 40 80
do 
for nn_width in 3 10 20
do 
for sigmaY in 0.01e6 0.1e6 1e6 10e6
do 
for dsigma in 0.1 1.0 10.0 100.0
do 
for activation in tanh relu leaky_relu selu elu 
do 
for exp_id in 1 2 3 4 5
do 
srun -n 1 -N 1 julia NNLearn $nn_depth $nn_width $sigmaY $dsigma $activation $exp_id &
done
done 
done
done 
done
done  
