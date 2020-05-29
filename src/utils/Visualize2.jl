export visualize_displacement, visualize_von_mises_stress, visualize_mesh, visualize_boundary,
visualize_scalar_on_scoped_body, visualize_total_deformation_on_scoped_body, visualize_von_mises_stress_on_scoped_body


"""
    visualize_von_mises_stress(domain::Domain; frames::Int64 = 20, kwargs...)

Animation of von Mises stress tensors. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_von_mises_stress.gif?raw=true)
"""
function visualize_von_mises_stress(domain::Domain; frames::Int64 = 20, kwargs...)
    NT = length(domain.history["stress"])
    if NT == 0 
        error(ArgumentError("history[\"stress\"] is empty.")) 
    end
    visualize_von_mises_stress_on_scoped_body(zeros(NT+1, domain.nnodes*2), domain; frames = frames, kwargs...)
end

"""
    visualize_von_mises_stress(domain::Domain, t_step::Int64; kwargs...)

Plot of von Mises stress tensors at time step `t_step`.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_von_mises_stress_50.png?raw=true)
"""
function visualize_von_mises_stress(domain::Domain, t_step::Int64; kwargs...)
    domain2 = copy(domain)
    domain2.history["stress"] = [domain.history["stress"][t_step]]
    visualize_von_mises_stress_on_scoped_body(zeros(1, domain2.nnodes*2), domain2; frames=1, kwargs...)
end

"""
    visualize_displacement(domain::Domain)
    visualize_displacement(u::Array{Float64, 2}, domain::Domain)

Animation of displacements using points. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_displacement.gif?raw=true)
"""
function visualize_displacement(domain::Domain; scale_factor::Float64 = 1.0)
    u = hcat(domain.history["state"]...) * scale_factor
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


"""
    visualize_displacement(domain::Domain)
    visualize_displacement(nodes::Array{Float64,2}, elems::Array{Int64, 2})

Visualizes the mesh.
![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_mesh.png?raw=true)
"""
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
    gca().invert_yaxis()
end

function visualize_mesh(domain::Domain) 
    elements = zeros(Int64, length(domain.elements), 4)
    for (k,e) in enumerate(domain.elements)
        elements[k,:] = e.elnodes
    end
    visualize_mesh(domain.nodes, elements)
end


"""
    visualize_boundary(domain::Domain, direction::String="x")

Visualizes the boundary conditions. The boundary configuration is shown in the direction `direction`.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_boundary.png?raw=true)
"""
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
    # gca().invert_yaxis()
    
end

"""
    visualize_scalar_on_scoped_body(d::Array{Float64}, domain::Domain)
    visualize_scalar_on_scoped_body(s::Array{Float64, 1}, d::Array{Float64,1}, domain::Domain;
        scale_factor::Float64 = 1.0)

Plot the scalar on scoped body. For example, `s` can be the von Mises stress tensor. 
"""
function visualize_scalar_on_scoped_body(s::Array{Float64, 1}, d::Array{Float64,1}, domain::Domain;
        scale_factor::Real = 1.0)
    n = matplotlib.colors.Normalize(vmin=minimum(s), vmax=maximum(s))
    cm = matplotlib.cm
    cmap = cm.jet
    m = cm.ScalarMappable(norm=n, cmap=cmap)
    elements = domain.elements
    patches = []
    for i = 1:size(elements,1)
        e = elements[i].elnodes
        sv = length(s)==domain.nnodes ? mean(s[e]) : s[i]
        dv = [d[e] d[e .+ domain.nnodes]]
        p = plt.Polygon(elements[i].coords + scale_factor * dv,edgecolor="k",lw=1,fill=true,
            fc=m.to_rgba(sv))
        push!(patches, p)
    end
    p = matplotlib.collections.PatchCollection(patches, match_original=true)
    gca().add_collection(p)
    colorbar(m)
    gca().invert_yaxis()
    axis("scaled")
    xlabel("x")
    ylabel("y")
end 

function visualize_scalar_on_scoped_body(s_all::Array{Float64, 2}, d_all::Array{Float64,2}, domain::Domain;
    scale_factor::Real = 1.0, frames::Int64 = 20)
    if frames==1
         visualize_scalar_on_scoped_body(s_all[1,:], d_all[1,:], domain; scale_factor = scale_factor)
         return nothing
    end
    n = matplotlib.colors.Normalize(vmin=minimum(s_all), vmax=maximum(s_all))
    cm = matplotlib.cm
    cmap = cm.jet
    m = cm.ScalarMappable(norm=n, cmap=cmap)
    elements = domain.elements
    close("all")
    # Find the bounding box: the domain takes up 80% 
    D = zeros(size(d_all,1), domain.nnodes)
    for i = 1:size(d_all,1)
        D[i,:] = domain.nodes[:,1] + scale_factor*d_all[i,1:domain.nnodes]
    end
    a1, b1 = minimum(D), maximum(D)
    h1 = (b1-a1)*0.1

    D = zeros(size(d_all,1), domain.nnodes)
    for i = 1:size(d_all,1)
        D[i,:] = domain.nodes[:,2] + scale_factor*d_all[i,domain.nnodes+1:end]
    end
    a2, b2 = minimum(D), maximum(D)
    h2 = (b2-a2)*0.1
    
    
    function update(frame)
        s = s_all[frame,:]
        d = d_all[frame,:]
        gca().clear()
        patches = []
        for i = 1:size(elements,1)
            e = elements[i].elnodes
            sv = length(s)==domain.nnodes ? mean(s[e]) : s[i]
            dv = [d[e] d[e .+ domain.nnodes]]
            p = plt.Polygon(elements[i].coords + scale_factor * dv,edgecolor="k",lw=1,fill=true,
                fc=m.to_rgba(sv))
            push!(patches, p)
        end
        p = matplotlib.collections.PatchCollection(patches, match_original=true)
        gca().add_collection(p)
        xlabel("x")
        ylabel("y")
        title("Snapshot = $frame")
        axis("scaled")
        xlim(a1-h1, b1+h1)
        ylim(a2-h2, b2+h2)
        gca().invert_yaxis()
    end
    update(1)
    t = title("Snapshot = 1")
    cb = colorbar(m)
    animate(update, Int64.(round.(LinRange(div(size(s_all,1), frames), size(s_all,1),frames))))
end


@doc raw"""
    visualize_total_deformation_on_scoped_body(d_all::Array{Float64,2}, domain::Domain;
    scale_factor::Float64 = 1.0, frames::Int64 = 20)

Visualizes the total deformation

$$\sqrt{u_x^2 + u_y^2}$$


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_total_deformation_on_scoped_body.gif?raw=true)
"""
function visualize_total_deformation_on_scoped_body(d_all::Array{Float64,2}, domain::Domain;
    scale_factor::Real = 1.0, frames::Int64 = 20)
    s_all = d_all
    S = zeros(size(s_all,1), domain.nnodes)
    for i = 1:size(s_all,1)
        S[i,:] = @. sqrt( s_all[i,1:domain.nnodes]^2 + s_all[i,domain.nnodes+1:end]^2 ) 
    end
    visualize_scalar_on_scoped_body(S, d_all, domain, scale_factor = scale_factor, frames=frames)
end

"""
    visualize_von_mises_stress_on_scoped_body(d_all::Array{Float64,2}, domain::Domain;
    scale_factor::Real = 1.0, frames::Int64 = 20)

Similar to [`visualize_von_mises_stress`](@ref), but the domain can be a deformed body. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/visualize_von_mises_stress_on_scoped_body.gif?raw=true)
"""
function visualize_von_mises_stress_on_scoped_body(d_all::Array{Float64,2}, domain::Domain;
    scale_factor::Real = 1.0, frames::Int64 = 20)
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

    visualize_scalar_on_scoped_body(S, d_all, domain, scale_factor = scale_factor, frames=frames)
end


