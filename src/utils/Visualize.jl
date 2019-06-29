export visstatic, visdynamic
function visstatic(domain::Domain, vmin=nothing, vmax=nothing; scaling = 1.0)
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    fig,ax = subplots()
    temp = nodes + [u v]
    x1, x2 = minimum(temp[:,1]), maximum(temp[:,1])
    y1, y2 = minimum(temp[:,2]), maximum(temp[:,2])
    σ =[]
    for e in domain.elements
        σs = e.strain
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
    for k = 1:length(domain.state_history)
        u1 = domain.state_history[k][1:N] + domain.nodes[:,1]
        u2 = domain.state_history[k][N+1:end] + domain.nodes[:,2]

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