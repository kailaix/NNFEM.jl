export visualize_displacement, visualize_von_mises_stress, visualize
function visualize(domain::Domain)
end

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
    visualize_displacement(domain::Domain)

Animation of displacements. 
"""
function visualize_displacement(domain::Domain)
    u = hcat(domain.history["state"]...)
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
    out = animate(update, Int64.(round.(LinRange(1, NT, 20))))
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    out
end