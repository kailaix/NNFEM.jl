
if !isdir("Debug"); mkdir("Debug"); end
if !isdir("Data"); mkdir("Data"); end


for i = 1:5
    @spawnat i include("Data_NNPlatePull.jl")
    # println(tid)
end