using Revise
using NNFEM
using Statistics 

nodes, elements = meshread("crack.msh")

visualize_mesh(nodes, elements)

ids = Int64[]
for i = 1:size(nodes, 1)
    x, y = nodes[i,:]
    if  abs(y-0.5*(x-0.5))<0.01 && (0.5 < x <= 1.55)
        push!(ids, i)
        @info x, y
        plot([x], [y], ".g")
    end
end

new_nodes = []
adic = Dict{Int64, Int64}()
for i = 1:size(elements, 1)
    e = elements[i,:]
    if !any([j in ids for j in e])
        continue
    end

    @info i 
    x, y = mean(nodes[e,:], dims=1)[:]
    if y-0.5*(x-0.5)<0
        plot([x], [y], ".r")
    else
        continue
    end

    for j = 1:4
        if e[j] in ids 
            if haskey(adic,e[j])
                elements[i,j] = adic[e[j]]
            else
                push!(new_nodes, nodes[e[j],:] + 0.02*rand(2))
                elements[i,j] = size(new_nodes,1) + size(nodes,1)
                adic[e[j]] = size(new_nodes,1) + size(nodes,1)
            end
        end
    end
        
end

new_nodes = hcat(new_nodes...)
nodes = [nodes;new_nodes']

close("all")
visualize_mesh(nodes, elements)
