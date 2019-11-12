export readMesh, save, load, read_data, write_data, convert_mat
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
    file = matopen(file, "w")
    write(file, "state", domain.history["state"])
    write(file, "fext", domain.history["fext"])
    write(file, "fint", domain.history["fint"])
    close(file)
end

# state, fext
function read_data(file::String)
    vars = matread(file)
    #use fint for debugging purpose
    vars["state"], vars["fext"]
end


# type == nn2array: read mat convert to 1D array
# type == array2nn: read 1D array convert to mat
function convert_mat(type::String, nnlayers::Array{Int64}, file1::String)
    if type == "nn2array" 
       nn = matread(file1)
       nnlayerNum = length(nnlayers) - 1
       theta_indptr = ones(Int64, 2*nnlayerNum+1)
       for i = 1:nnlayerNum
       	   theta_indptr[2*i]   = theta_indptr[2*i-1] + nnlayers[i]*nnlayers[i+1]
	       theta_indptr[2*i+1] = theta_indptr[2*i] + nnlayers[i+1]
       end

       
       theta = zeros(Float64, theta_indptr[end]-1)
       for i = 1:nnlayerNum
           for name in nn
               if occursin("connected_" * string(i) * "backslashweights", name.first)
                   theta[theta_indptr[2*(i+1)-1]:theta_indptr[2*(i+1)]-1] = transpose(name.second)[:]
               end
  	       if occursin("connected_" * string(i) * "backslashbiases", name.first)
                   theta[theta_indptr[2*(i+1)]:theta_indptr[2*(i+1)+1]-1] = transpose(name.second)[:]
               end
  	       if occursin("connectedbackslashweights", name.first)
                   theta[theta_indptr[1]:theta_indptr[2]-1] = transpose(name.second)[:]
               end
  	       if occursin("connectedbackslashbiases", name.first)
                   theta[theta_indptr[2]:theta_indptr[3]-1] = transpose(name.second)[:]
               end
           end
       end    

       return theta
   end
end