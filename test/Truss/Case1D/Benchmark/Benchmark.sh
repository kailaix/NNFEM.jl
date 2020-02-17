# thresholding 
for nn_depth in 3 
do 
for nn_width in 20
do 
for sigmaY in 0.001e6 0.01e6 0.1e6 1e6
do 
for dsigma in 0.1 1.0 10.0 100.0
do 
for activation in tanh
do 
for exp_id in 1 2 3 4 5
do 
srun -n 1 -N 1 ./Benchmark_.sh $nn_depth $nn_width $sigmaY $dsigma $activation &
done
done 
done
done 
done
done  

# nn architecture 
for nn_depth in 3 5 10 20 
do 
for nn_width in 5 20 40 
do 
for sigmaY in 0.01e6
do 
for dsigma in 1.0
do 
for activation in tanh relu leaky_relu selu elu 
do 
for exp_id in 1 2 3 4 5
do 
srun -n 1 -N 1 ./Benchmark_.sh $nn_depth $nn_width $sigmaY $dsigma $activation &
done
done 
done
done 
done
done  

