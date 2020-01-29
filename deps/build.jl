using ADCME
function buildops(dirname)
    PWD = pwd()
    cd(joinpath("$(@__DIR__)/CustomOp/", dirname))
    if !isdir("build")
        mkdir("build")
        cd("build")
        ADCME.cmake()
        run(`$(ADCME.MAKE) -j`)
    end
    cd(PWD)
end

buildops("FintComp")
buildops("SymOp")
buildops("OrthotropicOp")
buildops("SPDOp")
buildops("CholOp")
buildops("CholOrthOp")

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
