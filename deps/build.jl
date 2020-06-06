using ADCME
using Libdl

#--------------------------------------------------------------
# install gmsh 
const prefix = joinpath(ADCME.LIBDIR,"..")


download_info = Dict(
    "Linux64" => ("https://gmsh.info/bin/Linux/gmsh-4.5.6-Linux64-sdk.tgz", "gmsh.tgz"),
    "Windows64" => ("https://gmsh.info/bin/Windows/gmsh-4.5.6-Windows64-sdk.zip", "gmsh.zip"),
    "Windows32" => ("https://gmsh.info/bin/Windows/gmsh-4.5.6-Windows32-sdk.zip", "gmsh.zip"),
    "MacOSX" => ("https://gmsh.info/bin/MacOSX/gmsh-4.5.6-MacOSX-sdk.tgz", "gmsh.tgz"),
    "Linux32" => ("https://gmsh.info/bin/Linux/gmsh-4.5.6-Linux32-sdk.tgz", "gmsh.tgz")
)

key = ""
if Sys.iswindows()
    global key = "Windows$(Sys.WORD_SIZE)"
elseif Sys.isapple()
    global key = "MacOSX"
else
    global key = "Linux$(Sys.WORD_SIZE)"
end

if !isfile(joinpath(prefix, download_info[key][2]))
    download(download_info[key][1], joinpath(prefix, download_info[key][2]))
    if !isdir(joinpath(prefix, "gmsh"))
        mkdir(joinpath(prefix, "gmsh"))
    end 
    if Sys.iswindows()
        run(`cmd /c unzip $(joinpath(prefix, download_info[key][2])) -d $(joinpath(prefix, "gmsh"))`)
    else
        run(`tar zxvf $(joinpath(prefix, download_info[key][2])) -C $(joinpath(prefix, "gmsh"))`)
    end
end

cp(joinpath(prefix, "gmsh", "gmsh-4.5.6-$key-sdk", "lib"), joinpath(@__DIR__, "Gmsh"), force=true)
cp(joinpath(@__DIR__, "gmsh.jl"), joinpath(@__DIR__, "Gmsh", "gmsh.jl"), force=true)


install_adept()

files = readdir("CustomOp")
all_files = []
for f in files 
    if occursin("CMake", f) || occursin("build", f)
        continue
    end
    push!(all_files, f)
end

srcs = "add_library(DataLib SHARED "
for r in all_files
    fs = readdir("CustomOp/"*r)
    fs = filter(x->endswith(x, ".cpp"), fs)[1]
    global srcs *= "\${CMAKE_SOURCE_DIR}/$r/$fs "
end
srcs *= ")"

cnt = String(read("CustomOp/CMakeListsTemplate.txt"))
cnt = replace(cnt, "[REPLACE]"=>srcs)
open("CustomOp/CMakeLists.txt", "w") do io 
    write(io, cnt)
end

if !isdir("$(@__DIR__)/CustomOp/build")
    mkdir("$(@__DIR__)/CustomOp/build")
end
cd("$(@__DIR__)/CustomOp/build")
ADCME.cmake()
ADCME.make()
cd("../../")


