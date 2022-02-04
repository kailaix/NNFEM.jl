# # thresholding 
for nn_depth in 3 
do 
for nn_width in 20
do 
for sigmaY in 1.0 5.0 10.0 50.0 100.0 500.0 1000.0 10000.0
do 
for dsigma in 0.001 0.01 0.1 1.0 10.0 
do 
for activation in tanh
do 
srun -n 1 -N 1 ./Benchmark_.sh $nn_width $nn_depth $sigmaY $dsigma $activation &
done
done 
done
done 
done  

# # nn architecture 
for nn_depth in 1 3 8 20
do 
for nn_width in 2 10 20 40 
do 
for sigmaY in 100.0
do 
for dsigma in 0.1
do 
for activation in tanh relu leaky_relu selu elu 
do 
srun -n 1 -N 1 ./Benchmark_.sh $nn_width $nn_depth $sigmaY $dsigma $activation &
done
done 
done
done 
done  

