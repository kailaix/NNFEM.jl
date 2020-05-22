for sigma in 1.0 0.1 0.01 0.001
do 
for ndata in 10 20 50 100 200 289
do 
julia poisson17.jl & 
done 
wait %1 %2 %3 %4 %5 %6
done 
