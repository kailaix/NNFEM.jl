export visualize_displacement, visualize_von_mises_stress, visualize
function visualize(domain::Domain)
end

function visualize_von_mises_stress(Se::Array{Float64}, domain::Domain)
end

function visualize_displacement(u::Array{Float64, 2}, domain::Domain)
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
    function update(frame)
        p.set_data(U0[:,frame], V0[:,frame])
        t.set_text("Snapshot = $frame")
    end
    animate(update, Int64.(round.(LinRange(1, NT, 20))))
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
end