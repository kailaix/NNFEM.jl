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


buildops("test/Plate/MultiScale/Ops/Sym")
buildops("test/Plate/Hyperelasticity/Ops/Sym")
buildops("test/Plate/Plasticity/Ops/Sym")
buildops("deps/CustomOp/FintComp")
