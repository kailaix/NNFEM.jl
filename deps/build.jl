using ADCME
function buildops(dirname)
    PWD = pwd()
    cd(joinpath("$(@__DIR__)/../", dirname))
    if !isdir("build")
        mkdir("build")
        cd("build")
        ADCME.cmake()
        run(`$(ADCME.MAKE) -j`)
    end
    cd(PWD)
end

buildops("deps/CustomOp/FintComp")
buildops("deps/CustomOp/SymOp")
buildops("deps/CustomOp/OrthotropicOp")
buildops("deps/CustomOp/SPDOp")
buildops("deps/CustomOp/CholOp")
buildops("deps/CustomOp/CholOrthOp")

# # build ADLaw
# PWD = pwd()
# cd(joinpath("$(@__DIR__)/../", "deps/CustomOp/ADLaw"))
# if !isdir("Adept-2")
#     run(`sh download_lib.sh`)
# end
# if !isdir("build")
#     mkdir("build")
#     cd("build")
#     ADCME.cmake()
#     run(`$(ADCME.MAKE) -j`)
# end
# cd(PWD)
