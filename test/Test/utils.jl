@testset "GmeshReader" begin
    file = @__DIR__*"/../deps/plate.msh"
    elements, nodes, boundaries = readMesh(file)
end