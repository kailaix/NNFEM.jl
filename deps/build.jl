using ADCME
using Libdl

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

rm("$(@__DIR__)/CustomOp/build", force=true, recursive=true)
mkdir("$(@__DIR__)/CustomOp/build")
cd("$(@__DIR__)/CustomOp/build")
ADCME.cmake()
ADCME.make()
cd("../../")


