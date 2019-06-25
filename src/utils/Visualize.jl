using PyPlot
using PyCall
animation = pyimport("matplotlib.animation")

export visstatic, visdynamic
function visstatic(domain::Domain)
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    u = nodes[:,1] + u; v = nodes[:,2] + v
    scatter(u, v)
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