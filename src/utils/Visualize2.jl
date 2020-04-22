export visualize_displacement, visualize_von_mises_stress, visualize_mesh, visualize_boundary


"""
    visualize_von_mises_stress(domain::Domain)

Animation of von Mises stress tensors. 
"""
function visualize_von_mises_stress(domain::Domain)
    stress = domain.history["stress"]
    S = zeros(length(stress), length(domain.elements))
    x = zeros(length(domain.elements))
    y = zeros(length(domain.elements))
    for t = 1:length(stress)
        cnt = 1
        for (k,e) in enumerate(domain.elements)
            ct = mean(domain.elements[k].coords, dims=1)
            x[k], y[k] = ct[1,1], ct[1,2]
            
                ss = Float64[]
                nstress = length(e.mat)
                for p = 1:nstress
                    push!(ss, postprocess_stress(stress[t][cnt, :] ,"vonMises"))
                    cnt += 1
                end
                S[t, k] = mean(ss)
            
        end
    end   
    
    # function update(i)
    # c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    close("all")
    
    xlabel("x")
    ylabel("y")
    tricontour(x, y, S[1,:], 15, linewidths=0.5, colors="k")
    tricontourf(x, y, S[1,:], 15)
    axis("scaled")
    cb = colorbar()
    gca().invert_yaxis()
    function update(i)
        gca().clear()
        tricontour(x, y, S[i,:], 15, linewidths=0.5, colors="k")
        tricontourf(x, y, S[i,:], 15)
        xlabel("x")
        ylabel("y")
    end

    animate(update, Int64.(round.(LinRange(2, size(S,1),20))))
end



"""
    visualize_von_mises_stress(domain::Domain, t_step::Int64)

Plot of von Mises stress tensors at time step `t_step`.
"""
function visualize_von_mises_stress(domain::Domain, t_step::Int64)
    stress = domain.history["stress"]
    S = zeros(length(stress), length(domain.elements))
    x = zeros(length(domain.elements))
    y = zeros(length(domain.elements))
    for t = 1:length(stress)
        cnt = 1
        for (k,e) in enumerate(domain.elements)
            ct = mean(domain.elements[k].coords, dims=1)
            x[k], y[k] = ct[1,1], ct[1,2]
            
                ss = Float64[]
                nstress = length(e.mat)
                for p = 1:nstress
                    push!(ss, postprocess_stress(stress[t][cnt, :] ,"vonMises"))
                    cnt += 1
                end
                S[t, k] = mean(ss)
            
        end
    end   
    
    # function update(i)
    # c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    close("all")
    
    xlabel("x")
    ylabel("y")
    tricontour(x, y, S[t_step,:], 15, linewidths=0.5, colors="k")
    tricontourf(x, y, S[t_step,:], 15)
    axis("scaled")
    colorbar()
    gca().invert_yaxis()
end

"""
    visualize_displacement(domain::Domain)

Animation of displacements. 
"""
function visualize_displacement(domain::Domain)
    u = hcat(domain.history["state"]...)
    visualize_displacement(u, domain)
end

function visualize_displacement(u::Array{Float64, 2}, domain::Domain)
    if size(u,2)==2domain.nnodes 
        u = Array(u')
    end
    X0, Y0 = domain.nodes[1:domain.nnodes], domain.nodes[domain.nnodes+1:end]
    NT = size(u, 2)
    U0 = zeros(domain.nnodes, NT)
    V0 = zeros(domain.nnodes, NT)
    for i = 1:NT
        U0[:,i] = X0 + u[1:domain.nnodes, i]
        V0[:,i] = Y0 + u[domain.nnodes+1:end, i]
    end
    xmin, xmax = minimum(U0), maximum(U0)
    ymin, ymax = minimum(V0), maximum(V0)
    l1 = (xmax-xmin)/10
    l2 = (ymax-ymin)/10
    xmin -= l1 
    xmax += l1 
    ymin -= l2 
    ymax += l2
    close("all")
    p, = plot([], [], ".", markersize=5)
    t = title("Snapshot = 0")
    xlim(xmin, xmax)
    ylim(ymin, ymax)
    axis("scaled")
    function update(frame)
        p.set_data(U0[:,frame], V0[:,frame])
        t.set_text("Snapshot = $frame")
    end
    out = animate(update, Int64.(round.(LinRange(1, NT, 20))))
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    out
end

function visualize_mesh(nodes::Array{Float64,2}, elems::Array{Int64, 2})
    patches = PyObject[]
    for i = 1:size(elems,1)
        e = elems[i,:]
        p = plt.Polygon(nodes[e,:],edgecolor="k",lw=1,fc=nothing,fill=false)
        push!(patches, p)
    end
    p = matplotlib.collections.PatchCollection(patches, match_original=true)
    gca().add_collection(p)
    axis("scaled")
    xlabel("x")
    ylabel("y")
end

function visualize_mesh(domain::Domain) 
    elements = zeros(Int64, length(domain.elements), 4)
    for (k,e) in enumerate(domain.elements)
        elements[k,:] = e.elnodes
    end
    visualize_mesh(domain.nodes, elements)
end



function visualize_boundary(domain::Domain, direction::String="x")
    visualize_mesh(domain)
    direction = direction == "x" ? 1 : 2;

    function _helper(idx, marker, Label)
        if length(idx)==0
            return 
        end
        x, y = domain.nodes[idx,1], domain.nodes[idx,2]
        plot(x, y, marker, markersize=10, label="$Label")
    end
    _helper(findall(domain.EBC[:,direction] .== -1), ".", "Fixed Dirichlet")
    _helper(findall(domain.EBC[:,direction] .== -2), ".", "Time-dependent Dirichlet")
    _helper(findall(domain.FBC[:,direction] .== -1), "+", "Fixed Neumann")
    _helper(findall(domain.FBC[:,direction] .== -2), "+", "Time-dependent Neumann")

    if size(domain.edge_traction_data,1)>0
        ids = unique(domain.edge_traction_data[:,3])
        for (k, ids_) in enumerate(ids)
            data = domain.edge_traction_data[domain.edge_traction_data[:,3].==ids_,1:2]
                        
            for i = 1:size(data, 1)
                elem = domain.elements[data[i,1]]
                if data[i,2]==1
                    idx = [1;2]
                elseif data[i,2]==2
                    idx = [2;3]
                elseif data[i,2]==3
                    idx = [3;4]
                elseif data[i,2]==4
                    idx = [4;1]
                end
                x = elem.coords[idx,1]
                y = elem.coords[idx,2]
                if i==1
                    plot(x, y, "--",linewidth=2, color="C$(k+3)", label="Edge Traction (ID=$ids_)")
                else
                    plot(x, y, "--",linewidth=2, color="C$(k+3)")
                end
            end
        end 

    end
    legend()
    gca().invert_yaxis()
    
end