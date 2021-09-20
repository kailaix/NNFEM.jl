export visstatic, visdynamic, show_strain_stress, prepare_strain_stress_data1D, prepare_strain_stress_data2D,
prepare_sequence_strain_stress_data2D, VisualizeStress2D,visσ, VisualizeStrainStressSurface, visstate, postprocess_stress
import PyPlot:scatter3D
function visstatic(domain::Domain, vmin=nothing, vmax=nothing; scaling = 1.0)
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    fig,ax = subplots()
    temp = nodes + [u v]
    x1, x2 = minimum(temp[:,1]), maximum(temp[:,1])
    y1, y2 = minimum(temp[:,2]), maximum(temp[:,2])
    σ =[]
    for e in domain.elements
        σs = e.stress
        if σs[1]==undef
            σs = ones(length(e.stress))
        end
        push!(σ,mean([sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2) for σ in σs]))
    end
    vmin = vmin==nothing ? minimum(σ) : vmin 
    vmax = vmax==nothing ? maximum(σ) : vmax
    #@show vmin, vmax
    cNorm  = matplotlib.colors.Normalize(
            vmin=vmin,
            vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("jet"))
    for (k,e) in enumerate(domain.elements)
        n_ = nodes[getNodes(e),:] + scaling*[u[getNodes(e),:] v[getNodes(e),:]]
        p = plt.Polygon(n_, facecolor = scalarMap.to_rgba(σ[k]), fill=true, alpha=0.5)
        ax.add_patch(p)
    end
    xlim(x1 .-0.1,x2 .+0.1)
    ylim(y1 .-0.1,y2 .+0.1)
end

function show_strain_stress(domain::Domain)
    strain = domain.history["strain"]
    stress = domain.history["stress"]
    strain = hcat(strain...)
    stress = hcat(stress...)
    close("all")
    for i = 1:size(strain,1)
        plot(strain[i,:], stress[i,:])
    end
    xlabel("strain")
    ylabel("stress")
end

function prepare_strain_stress_data1D(domain::Domain)
    strain = domain.history["strain"]
    stress = domain.history["stress"]
    strain = hcat(strain...)
    stress = hcat(stress...)
    ngp = size(strain,1)
    nt = size(strain,2)
    X = zeros(nt*ngp,3)
    y = zeros(nt*ngp)
    strain = [zeros(ngp, 1) strain]
    stress = [zeros(ngp,1) stress]
    nt += 1
    k = 1
    for i = 1:ngp
        for j = 2:nt
            X[k,:] = [strain[i,j] strain[i,j-1] stress[i,j-1]] #ε, ε0, σ0
            y[k] = stress[i,j]
            k = k + 1
        end
    end
    X,y
end


function prepare_strain_stress_data1D(strain::Array{Any, 1}, stress::Array{Any, 1},)
    strain = hcat(strain...)
    stress = hcat(stress...)
    ngp = size(strain,1)
    nt = size(strain,2)
    X = zeros(nt*ngp,3)
    y = zeros(nt*ngp)
    strain = [zeros(ngp, 1) strain]
    stress = [zeros(ngp,1) stress]
    nt += 1
    k = 1
    for i = 1:ngp
        for j = 2:nt
            X[k,:] = [strain[i,j] strain[i,j-1] stress[i,j-1]] #ε, ε0, σ0
            y[k] = stress[i,j]
            k = k + 1
        end
    end
    X,y
end


function prepare_strain_stress_data2D(domain::Domain, scale::Float64=1.0)
    strain = domain.history["strain"]
    stress = domain.history["stress"]
    ngp = size(strain[1],1)
    nt = length(strain)
    X = zeros(nt*ngp,9)
    y = zeros(nt*ngp,3)
    pushfirst!(strain, zero(strain[1]))
    pushfirst!(stress, zero(stress[1]))
    
    k = 1
    nt += 1
    for j = 2:nt
        for i = 1:ngp
            X[k,:] = [strain[j][i,:]; strain[j-1][i,:]; stress[j-1][i,:]/scale]#ε, ε0, σ0
            y[k,:] = stress[j][i,:]/scale
            k = k + 1
        end
    end
    X,y
end


function prepare_strain_stress_data2D(strain::Array{Any, 1}, stress::Array{Any, 1}, scale::Float64=1.0)
    ngp = size(strain[1],1)
    nt = length(strain)
    X = zeros(nt*ngp,9)
    y = zeros(nt*ngp,3)
    pushfirst!(strain, zero(strain[1]))
    pushfirst!(stress, zero(stress[1]))
    
    k = 1
    nt += 1
    for j = 2:nt
        for i = 1:ngp
            X[k,:] = [strain[j][i,:]; strain[j-1][i,:]; stress[j-1][i,:]/scale]#ε, ε0, σ0
            y[k,:] = stress[j][i,:]/scale
            k = k + 1
        end
    end
    X,y
end


# Return strain_seq[ngp, NT+1, 3], stress_seq[ngp, NT+1, 3]
function prepare_sequence_strain_stress_data2D(domain::Domain, scale::Float64=1.0)
    strain = domain.history["strain"]
    stress = domain.history["stress"]
    ngp = size(strain[1],1)
    nt = length(strain)
    strain_seq = zeros(nt+1, ngp, 3)
    stress_seq = zeros(nt+1, ngp, 3)
    
    
    for j = 2:nt+1
            #@show nt, ngp, size(strain[j-1])
            strain_seq[j,:,:] = strain[j-1]
            stress_seq[j,:,:] = stress[j-1]
    end
    strain_seq, stress_seq
end


# https://risa.com/risahelp/risa3d/Content/3D_2D_Only_Topics/Plates%20-%20Results.htm
function postprocess_stress(stress::Array{Float64}, name::String)
    if name == "vonMises"
        σx = stress[1]; σy = stress[2]; τxy = stress[3]
        σ1 = (σx+σy)/2+sqrt((σx-σy)^2/4 + τxy^2)
        σ2 = (σx+σy)/2-sqrt((σx-σy)^2/4 + τxy^2)
        sqrt(σ1^2-σ1*σ2+σ2^2)

        #σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2
    elseif name == "principal"
        σx = stress[1]; σy = stress[2]; τxy = stress[3]
        σ1 = (σx+σy)/2+sqrt((σx-σy)^2/4 + τxy^2)
        σ2 = (σx+σy)/2-sqrt((σx-σy)^2/4 + τxy^2)
        [σ1, σ2]
    else
        @error("postprocess_stress does not recognize ", name) 
    end
end

function postprocess_strain(strain::Array{Float64}, name::String)
    εx = strain[1]; εy = strain[2]; γxy = strain[3]
    if name == "vonMises"
        sqrt(
            3/2*(εx^2+εy^2) + 3/4*γxy^2
        )
    elseif name == "principal"
        s1 = (εx + εy)/2 + sqrt(
            ((εx-εy)/2)^2 + (γxy/2)^2
        )
        s2 = (εx + εy)/2 - sqrt(
            ((εx-εy)/2)^2 + (γxy/2)^2
        )
        [s1;s2]
    else
        @error("postprocess_strain does not recognize ", name) 
    end
end

function VisualizeStress2D(domain::Domain)
    strain = domain.history["strain"]
    stress = domain.history["stress"]
    NT = length(stress)
    ngp = size(stress[1], 1)
    V = zeros(NT, ngp, 2)
    for i = 1:NT
        for j = 1:ngp
            V[i,j,:] = postprocess_stress(stress[i][j,:], "principal")
        end
    end
    close("all")
    for i = rand(1:ngp, 10)
        x = V[:,i,1][:]/1e8
        y = V[:,i,2][:]/1e8
        plot(x, y, ".-")
    end
    V
end

function VisualizeStress2D(σ_ref::Array{Float64}, σ_comp::Array{Float64}, NT::Int64, Nlines::Int64=1)
    ngp = Int64(size(σ_ref,1)/NT)
    V_ref = zeros(NT, ngp, 3)
    V_comp = zeros(NT, ngp, 3)
    # 2 
    for i = 1:NT
        for j = 1:ngp
            V_ref[i,j,1] = postprocess_stress(σ_ref[(i-1)*ngp + j,:], "vonMises")
            V_ref[i,j,2:3] = postprocess_stress(σ_ref[(i-1)*ngp + j,:], "principal")
            V_comp[i,j,1] = postprocess_stress(σ_comp[(i-1)*ngp + j,:], "vonMises")
            V_comp[i,j,2:3] = postprocess_stress(σ_comp[(i-1)*ngp + j,:], "principal")
        end
    end
    close("all")
    col = "r"
    col2 = "g"
    
    k = 0
    for i = rand(1:ngp, Nlines)
        k += 1
        x = V_ref[:,i,1][:]
        y = V_ref[:,i,2][:]
        plot(x, y, ".-"*col[1])

        x = V_comp[:,i,1][:]
        y = V_comp[:,i,2][:]
        plot(x, y, ".--"*col2[1])
    end
end

function VisualizeStrainStressSurface(X::Array{Float64}, Y::Array{Float64}, seed::Int64=233)
    n = size(X,1)
    Random.seed!(seed)
    idx = rand(1:n, 2000)
    scatter3D(X[idx,1], X[idx,2], Y[idx,1], marker=".")
end

function visσ(domain::Domain, vmin=nothing, vmax=nothing; scaling = 1.0)
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    fig,ax = subplots()
    temp = nodes + [u v]
    x1, x2 = minimum(temp[:,1]), maximum(temp[:,1])
    y1, y2 = minimum(temp[:,2]), maximum(temp[:,2])
    σ =[]
    for e in domain.elements
        σs = e.stress
        # σs = [ones(3) for i = 1:length(e.stress)] # ! remove me
        push!(σ,mean([postprocess_stress(s, "vonMises")[1] for s in σs]))
    end
    vmin = vmin==nothing ? minimum(σ) : vmin 
    vmax = vmax==nothing ? maximum(σ) : vmax
    #@show vmin, vmax
    cNorm  = matplotlib.colors.Normalize(
            vmin=vmin,
            vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("jet"))
    for (k,e) in enumerate(domain.elements)
        N = getNodes(e)
        if length(N)==9
            N = N[[1;5;2;6;3;7;4;8]]
        end
        n_ = nodes[N,:] + scaling*[u[N,:] v[N,:]]
        p = plt.Polygon(n_, facecolor = scalarMap.to_rgba(σ[k]), fill=true)
        ax.add_patch(p)
    end
    scalarMap.set_array(σ)
    colorbar(scalarMap)
    xlim(x1 .-0.1,x2 .+0.1)
    ylim(y1 .-0.1,y2 .+0.1)
end

# visualize each quadrature point
function visσ(domain::Domain, ngp::Int64, vmin=nothing, vmax=nothing; σ=nothing, scaling = 1.0)
    
    
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    fig,ax = subplots()
    temp = nodes + [u v]
    x1, x2 = minimum(temp[:,1]), maximum(temp[:,1])
    y1, y2 = minimum(temp[:,2]), maximum(temp[:,2])
    
    if σ===nothing
        σ =[]
        for e in domain.elements
            σs = e.stress
            # σs = [ones(3) for i = 1:length(e.stress)] # ! remove me
            append!(σ, [postprocess_stress(s, "vonMises")[1] for s in σs])
        end
    end
    vmin = vmin==nothing ? minimum(σ) : vmin 
    vmax = vmax==nothing ? maximum(σ) : vmax
    #@show vmin, vmax
    cNorm  = matplotlib.colors.Normalize(
            vmin=vmin,
            vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("jet"))
    
    Gp_order_1D = zeros(Int64, ngp)
    if ngp == 2
        Gp_order_1D = [1, 2]
    elseif ngp == 3
        Gp_order_1D = [2, 1, 3]
    elseif ngp == 4
        Gp_order_1D = [4, 2, 1, 3]
    else
        error(" ngp == ", ngp, " has not implemented")
    end

    Gp_order = zeros(Int64, ngp, ngp)
    for gy = 1:ngp
        for gx = 1:ngp
        Gp_order[gx, gy] = (Gp_order_1D[gx] - 1)*ngp  +  Gp_order_1D[gy]
        end
    end 

    XYs = zeros(ngp+1, ngp+1, 2)
    xy = zeros(4, 2)
    for (k,e) in enumerate(domain.elements)
        N = getNodes(e)[[1;2;3;4]]

        XY = nodes[N,:] + scaling*[u[N,:] v[N,:]]

        for gy = 0:ngp
            for gx = 0:ngp
                w = [(1.0 - gx/ngp) * (1.0 - gy/ngp), gx/ngp * (1.0 - gy/ngp), 
                      gx/ngp * gy/ngp, (1.0 - gx/ngp) * gy/ngp]
       
                XYs[gx+1, gy+1, :] .= w[1]*XY[1,1] +w[2]*XY[2,1] +w[3]*XY[3,1] +w[4]*XY[4,1], w[1]*XY[1,2] +w[2]*XY[2,2] +w[3]*XY[3,2] +w[4]*XY[4,2]  
            end
        end

        for gy = 1:ngp
            for gx = 1:ngp

                xy[1, :] = XYs[gx, gy, :]
                xy[2, :] = XYs[gx+1, gy, :]
                xy[3, :] = XYs[gx+1, gy+1, :]
                xy[4, :] = XYs[gx, gy+1, :]
                
                p = plt.Polygon(xy, facecolor = scalarMap.to_rgba(σ[(k - 1)*ngp^2 + Gp_order[gx, gy]]), fill=true)
                ax.add_patch(p)
            end
        end
    end
    scalarMap.set_array(σ)
    colorbar(scalarMap)
    # xlim(x1 .-0.1,x2 .+0.1)
    # ylim(y1 .-0.1,y2 .+0.1)
end


function visstate(domain::Domain, state::Array{Float64}; kwargs...)
    u,v = state[1:domain.nnodes], state[domain.nnodes+1:end]
    nodes = domain.nodes
    fig,ax = subplots()
    temp = nodes + [u v]
    x1, x2 = minimum(temp[:,1]), maximum(temp[:,1])
    y1, y2 = minimum(temp[:,2]), maximum(temp[:,2])
    for (k,e) in enumerate(domain.elements)
        N = getNodes(e)
        if length(N)==9
            N = N[[1;5;2;6;3;7;4;8]]
        end
        n_ = nodes[N,:] + [u[N,:] v[N,:]]
        p = plt.Polygon(n_; kwargs...)
        ax.add_patch(p)
    end
    xlim(x1 .-0.1,x2 .+0.1)
    ylim(y1 .-0.1,y2 .+0.1)
end

function visstate(domain::Domain; kwargs...)
    visstate(domain, domain.state; kwargs...)
end


# need 1)empty domain for data structure 
#      2)data in stresses and disps
function visσ(domain::Domain, nx::Int64, ny::Int64,  stress::Array{Float64, 2}, disps::Array{Float64, 1}, 
    vmin=nothing, vmax=nothing; scaling = [1.0, 1.0])

    L_scale, sigma_scale = scaling

    u,v = disps[1:domain.nnodes], disps[domain.nnodes+1:end]
    nodes = domain.nodes
    ngpt = length(domain.elements[1].weights)
    ngp = Int64(sqrt(ngpt))

    ### Compute X[nx*ngp+1, ny*ngp+1], Y[nx*ngp+1, ny*ngp+1], C[nx*ngp, ny*ngp] for pcolormesh
    X = zeros(Float64, nx*ngp+1, ny*ngp+1)
    Y = zeros(Float64, nx*ngp+1, ny*ngp+1)
    C = zeros(Float64, nx*ngp, ny*ngp)
    
    Gp_order_1D = zeros(Int64, ngp)
    if ngp == 2
        Gp_order_1D = [1, 2]
    elseif ngp == 3
        Gp_order_1D = [2, 1, 3]
    elseif ngp == 4
        Gp_order_1D = [4, 2, 1, 3]
    else
        error(" ngp == ", ngp, " has not implemented")
    end

    Gp_order = zeros(Int64, ngp, ngp)
    for gy = 1:ngp
        for gx = 1:ngp
        Gp_order[gx, gy] = (Gp_order_1D[gx] - 1)*ngp  +  Gp_order_1D[gy]
        end
    end 


    for iy = 1:ny
        for ix = 1:nx
            eid = ix + (iy-1)*nx
            ele = domain.elements[eid]
            # (iy-1)*ngp + ngp
            # (iy-1)*ngp + 1
            # (iy-1)*ngp + 0       (ix-1)*ngp + 0, (ix-1)*ngp + 1, (ix-1)*ngp + ngp

            N = getNodes(ele)
            N = N[[1;2;3;4]] #take these nodes on the four corners
            xy = L_scale*(nodes[N,:] + [u[N,:] v[N,:]])
     
            for gy = 0:ngp
                for gx = 0:ngp
                    w = [(1.0 - gx/ngp) * (1.0 - gy/ngp), gx/ngp * (1.0 - gy/ngp), 
                          gx/ngp * gy/ngp, (1.0 - gx/ngp) * gy/ngp]
           
                    X[1 + (ix-1)*ngp + gx, 1 + (iy-1)*ngp + gy] = w[1]*xy[1,1] +w[2]*xy[2,1] +w[3]*xy[3,1] +w[4]*xy[4,1] 
                    Y[1 + (ix-1)*ngp + gx, 1 + (iy-1)*ngp + gy] = w[1]*xy[1,2] +w[2]*xy[2,2] +w[3]*xy[3,2] +w[4]*xy[4,2]  
                end
            end


            for gy = 1:ngp
                for gx = 1:ngp
                    σ = stress[(eid-1)*ngpt + Gp_order[gx ,gy], :]
                    σvm = sigma_scale * postprocess_stress(σ, "vonMises")
                    C[(ix-1)*ngp + gx, (iy-1)*ngp + gy] = σvm
                    
                end
            end

        end
    end



    vmin = vmin==nothing ? minimum(C) : vmin 
    vmax = vmax==nothing ? maximum(C) : vmax

    @show minimum(C), maximum(C) , "!!!!!!"

    fig,ax = subplots()
    x1, x2 = minimum(X), maximum(X)
    y1, y2 = minimum(Y), maximum(Y)
    

    p = plt.pcolormesh(X, Y, C, cmap="rainbow")
    
    clim(vmin,vmax)
    # cNorm  = colors.Normalize(
    #         vmin=0,
    #         vmax=1.0)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    
    # #scalarMap.set_array(σ)
    # colorbar(scalarMap)
    colorbar()
    dx, dy = (x2 - x1)*0.1, (y2 - y1)*0.1
    xlim(x1 .- dx,x2 .+ dx)
    ylim(y1 .- dy,y2 .+ dy)
    axis("equal")

    return vmin, vmax
end
