export visstatic, visdynamic, show_strain_stress, prepare_strain_stress_data1D, prepare_strain_stress_data2D,
prepare_sequence_strain_stress_data2D, VisualizeStress2D,visσ, VisualizeStrainStressSurface, visstate
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
    cNorm  = colors.Normalize(
            vmin=vmin,
            vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for (k,e) in enumerate(domain.elements)
        n_ = nodes[getNodes(e),:] + scaling*[u[getNodes(e),:] v[getNodes(e),:]]
        p = plt.Polygon(n_, facecolor = scalarMap.to_rgba(σ[k]), fill=true, alpha=0.5)
        ax.add_patch(p)
    end
    xlim(x1 .-0.1,x2 .+0.1)
    ylim(y1 .-0.1,y2 .+0.1)
end

function visdynamic(domain::Domain, name::String)
    
    # Set up formatting for the movie files
    Writer = animation.writers.avail["html"]
    writer = Writer(fps=15, bitrate=1800)

    close("all")
    fig = figure()
    # visualization
    scat0 = scatter(domain.nodes[:,1], domain.nodes[:,2], color="grey")
    grid(true)
    ims = Any[(scat0,)]

    N = size(domain.nodes,1)
    for k = 1:length(domain.history["state"])
        u1 = domain.history["state"][k][1:N] + domain.nodes[:,1]
        u2 = domain.history["state"][k][N+1:end] + domain.nodes[:,2]

        scat = scatter(u1, u2, color="orange")
        grid(true)
        tt = gca().text(.5, 1.05,"$k")
        # s2 = scatter(nodes[div(n+1,2)*n,1], nodes[div(n+1,2)*n,2], marker="x", color="red")
        # s3 = scatter(u1[div(n+1,2)*n], u2[div(n+1,2)*n], marker="*", color="red")
        push!(ims, (scat0,scat,tt))
    end

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=true)
    im_ani.save("$name.html", writer=writer)

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
    cNorm  = colors.Normalize(
            vmin=vmin,
            vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
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