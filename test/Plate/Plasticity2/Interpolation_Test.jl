include("nnutil.jl")




function get_state_history(tid, force_scale, fiber_size, nx = 10, ny = 5)
    nodes, _, _, _, _, _, _ = BoundaryCondition(tid, nx, ny, porder)
    elements = []
    for j = 1:ny
        for i = 1:nx 
            n = (nx*porder+1)*(j-1)*porder + (i-1)porder+1
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2

                elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                elnodes = [n, n + 2, n + 2 + 2*(2*nx+1),  n + 2*(2*nx+1), n+1, n + 2 + (2*nx+1), n + 1 + 2*(2*nx+1), n + (2*nx+1), n+1+(2*nx+1)]
            else
                error("polynomial order error, porder= ", porder)
            end
            coords = nodes[elnodes,:]
            push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
        end
    end


    nx_f, ny_f = nx*fiber_size, ny*fiber_size
    
    # number of subelements in one element in each directions
    sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

    ndofs = 2
    fine_to_coarse = zeros(Int64, ndofs*(nx*porder+1)*(ny*porder+1))
    for idof = 1:ndofs
        for iy = 1:ny*porder+1
            for ix = 1:nx*porder+1
                fine_to_coarse[ix + (iy - 1)*(nx*porder+1) + (idof-1)*(nx*porder+1)*(ny*porder+1)] = 
                1 + (ix - 1) * sx_f + (iy - 1) * (nx_f*porder + 1) * sy_f + (nx_f*porder + 1)*(ny_f*porder + 1)*(idof - 1)
            end
        end
    end


    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder, force_scale )
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    # full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    
    #update state history and fext_history on the homogenized domain
    state_history = [x[fine_to_coarse] for x in full_state_history]

    fext_history = []
    setNeumannBoundary!(domain, FBC, fext)
    for i = 1:NT
        globdat.time = Δt*i
        updateDomainStateBoundary!(domain, globdat)
        push!(fext_history, domain.fext[:])
    end
    domain.history["state"] = state_history
    return domain, globdat, state_history, fext_history
end


# basis function order
porder = 2
prop = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2,
"sigmaY"=>0.97e+4, "K"=>1e+5)



T = 0.05
NT = 100
Δt = T/NT


tid = 200
force_scale = 5.0

domain_fs1, globdat_fs1, state_history_fs1, fext_history_fs1 = get_state_history(tid, force_scale, 1)
@show size(hcat(fext_history_fs1...),2)
F_tot_fs1, E_all_fs1, w∂E∂u_all_fs1 = preprocessing(domain_fs1, globdat_fs1, hcat(fext_history_fs1...), Δt)


domain_fs2, globdat_fs2, state_history_fs2, fext_history_fs2 = get_state_history(tid, force_scale, 2)
@show size(hcat(fext_history_fs2...),2)
F_tot_fs2, E_all_fs2, w∂E∂u_all_fs2 = preprocessing(domain_fs2, globdat_fs2, hcat(fext_history_fs2...), Δt)

close("all")
visstate(domain_fs1; fill=false, edgecolor="blue")
savefig("test.png")

close("all")
visstate(domain_fs2; linestyle ="--", edgecolor="red", fill=false)
savefig("test2.png")
