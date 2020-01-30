<<<<<<< HEAD
#!/bin/bash

force_scale=50.0

julia Data_NNPlatePull.jl 100 $force_scale 2 2 &
julia Data_NNPlatePull.jl 101 $force_scale 2 2 &
julia Data_NNPlatePull.jl 102 $force_scale 2 2 &
julia Data_NNPlatePull.jl 103 $force_scale 2 2 &
julia Data_NNPlatePull.jl 104 $force_scale 2 2 &
julia Data_NNPlatePull.jl 105 $force_scale 2 2 &
julia Data_NNPlatePull.jl 106 $force_scale 2 2 &
julia Data_NNPlatePull.jl 200 $force_scale 2 2 &
julia Data_NNPlatePull.jl 201 $force_scale 2 2 &
julia Data_NNPlatePull.jl 202 $force_scale 2 2 &
julia Data_NNPlatePull.jl 203 $force_scale 2 2 &
julia Data_NNPlatePull.jl 204 $force_scale 2 2 &
julia Data_NNPlatePull.jl 205 $force_scale 2 2 &
julia Data_NNPlatePull.jl 206 $force_scale 2 2 &
julia Data_NNPlatePull.jl 300 $force_scale 2 2 &



wait
=======
for tid in 100 101 102 103 104 200 201 202 203 204 300
do 
    julia Data_NNPlatePull.jl $tid 5.0 2 2 &
done
>>>>>>> a03560c8aa597b42ed229ef85fd91a9c6f9aa95f
