function buildops(dirname)
    PWD = pwd()
    cd(joinpath("$(@__DIR__)/../", dirname))
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd(PWD)
end

buildops("deps/CustomOp/FintComp")
buildops("deps/CustomOp/SymOp")
buildops("deps/CustomOp/OrthotropicOp")
buildops("deps/CustomOp/SPDOp")
buildops("deps/CustomOp/CholOp")
buildops("deps/CustomOp/CholOrthOp")
