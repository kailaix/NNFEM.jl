for tid in 100 101 102 103 104 200 201 202 203 204 300
do 
    julia Data_NNPlatePull.jl $tid 5.0 2 2 &
done
