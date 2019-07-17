export readMesh, save, load, read_data, write_data
function readMesh(gmshFile::String)
    fp = open(gmshFile);
    boundaries = Dict{String, Array}()
    physicalnames = Dict{Int64, String}()
    nodes = nothing
    elements = nothing
    
    line = readline(fp)
    if (line == "\$MeshFormat")
        format = readline(fp)
        @assert(readline(fp) == "\$EndMeshFormat")
    end

    line = readline(fp)
    if (line == "\$PhysicalNames")
        nphys = parse(Int64,readline(fp))
        for i = 1:nphys
            line = readline(fp)
            physicalid = parse(Int64, split(line)[2])
            physicalnames[physicalid] = split(line)[3]
            boundaries[split(line)[3]] = []
        end
        @assert(readline(fp) == "\$EndPhysicalNames")
    end

    line = readline(fp)
    if (line == "\$Nodes")
        nnodes = parse(Int64, readline(fp))
        nodes = zeros(nnodes,2)
        for i = 1:nnodes
            l = readline(fp)|>split
            nodes[i,1] = parse(Float64, l[2])
            nodes[i,2] = parse(Float64, l[3])
        end
        @assert(readline(fp) == "\$EndNodes")
    end
    # println(physicalnames)
    # error()
    line = readline(fp)
    if (line == "\$Elements")
        nelems = readline(fp)
        nelems = parse(Int64,nelems)
        elements = []
        for i = 1:nelems
            l = readline(fp)|>split 
            l4 = parse(Int64, l[4])
            physicalname = physicalnames[l4]
            if startswith(physicalname,"\"Dirichlet")
                k1,k2 = parse(Int64, l[6]),parse(Int64, l[7])
                push!(boundaries[physicalname],[k1;k2])
            elseif startswith(physicalname,"\"Neumann")
                k1,k2 = parse(Int64, l[6]),parse(Int64, l[7])
                push!(boundaries[physicalname],[k1;k2])
            else 
                # println(physicalnames[l[4]], startswith(physicalnames[l[4]],"Dirichlet"))
                # println(l)
                k = [parse(Int64, l[i]) for i = 6:9]
                push!(elements, k)
            end
        end
        # elem = zeros(length(elements),4)
        # for i = 1:length(elements)
        #     elem[i,:] = elements[i]
        # end
        @assert(readline(fp) == "\$EndElements")
    end
    @warn("Jobs are not finished!!! Users need to compute EBC, g, NBC, f.")
    return elements, nodes, boundaries
end

function save(file::String, domain::Domain, globaldata::GlobalData)
    @save file domain globaldata
end

function load(file::String)
    @load file domain globaldata
    return domain, globaldata
end


# domain.state, domain.fint, fext
function write_data(file::String, domain::Domain)
end

# state, fext
function read_data(file::String)
end