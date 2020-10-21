function install_mfem()
    PWD = pwd()
    change_directory()
    http_file("https://bit.ly/mfem-4-1", "mfem-4-1.tgz")
    uncompress("mfem-4-1.tgz", "mfem-4.1")
    str = String(read("mfem-4.1/CMakeLists.txt"))
    str = replace(str, "add_library(mfem \${SOURCES} \${HEADERS} \${MASTER_HEADERS})"=>"""add_library(mfem SHARED \${SOURCES} \${HEADERS} \${MASTER_HEADERS})
set_property(TARGET mfem PROPERTY POSITION_INDEPENDENT_CODE ON)""")
    open("mfem-4.1/CMakeLists.txt", "w") do io 
        write(io, str)
    end
    change_directory("mfem-4.1/build")
    require_file("build.ninja") do
        ADCME.cmake(CMAKE_ARGS = ["SHARED=YES", "STATIC=NO"])
    end
    require_library("mfem") do 
        ADCME.make()
    end
    mv("libmfem.so.4.1.0", joinpath(ADCME.LIBDIR, "libmfem.so"), force=true)
    cd(PWD)
end

