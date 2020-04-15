for idx in 0 1 2
do 
for tid in 200 300
do 
julia Test_NNPlatePull.jl $idx $tid & 
done
done
