using Revise
using NNFEM
using PyPlot


using Revise
using ADCME
using PyCall
using Test 
using MAT
using NNFEM
using PyPlot
using JLD2
using LinearAlgebra
using Distributions, Random
using ForwardDiff
using DelimitedFiles

np = pyimport("numpy")
"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type: a string of load type, which can be "constant", or "Gaussian"
args: an array, 
      for Constant load, it has p1 and p2, in tangential and normal direction
      for "Gaussian" load, it has p, x0, and σ, the force is in the normal direction
                  
"""
function ComputeLoad(L, ne, porder, ngp, type,  args)
    @assert(ngp <= 4)
    dx = L/ne

    xx = np.linspace(0, L, ne*porder + 1)
    
    Ft, Fn = zeros(Float64, ne*porder + 1), zeros(Float64, ne*porder + 1)

    elem_coords = zeros(Float64, porder + 1, 2)

    # construct pressure load function
    if type == "Constant"
        pt, pn = args
        f_px, f_py = x->pt, x->pn
    elseif type == "Gaussian"
        pn, x0, σ = args
        f_px, f_py = x->0, x-> pn * 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)
    else
        error("Force load type is not recogonized ", type)
    end


    # assemble the force

    @show ne
    for e = 1:ne
        if porder == 1
            loc_id = [e, e+1]
        elseif porder == 2
            loc_id = [2*(e-1)+1, 2*(e-1)+3, 2*(e-1)+2]
        else
            error("porder error porder ==", porder)
        end

        loc_xx = xx[loc_id]         #x-coordinates of element nodes 
        elem_coords[:,1] .= loc_xx
        

        #return list 
        weights, hs = get1DElemShapeData( elem_coords, ngp)  
        
        for igp = 1:ngp
            gp_xx = loc_xx' * hs[igp]   #x-coordinates of element Gaussian points
            Ft[loc_id] += f_px(gp_xx) * hs[igp] * weights[igp]
            Fn[loc_id] += f_py(gp_xx) * hs[igp] * weights[igp]

        end
    end

    return Ft, Fn
end


function BoundaryCondition(tid, nx, ny, porder=2, Lx = 1.0, Ly = 0.5; force_scale=5.0)
    nnodes, neles = (nx*porder + 1)*(ny*porder + 1), nx*ny

    x = np.linspace(0.0, Lx, nx*porder + 1)
    y = np.linspace(0.0, Ly, ny*porder + 1)


    X, Y = np.meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    ndofs = 2

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    gt = nothing

    #todo only change 203

    # setting EBC:
    # tid = 1XX   |==|--> 
    # tid = 2XX   
    if div(tid,100)==1
        EBC[collect(1:nx*porder+1), :] .= -1 # fix bottom
        FBC[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -2 # force on the top
    elseif div(tid,100)==2
        EBC[collect(1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -1 # fix left
        FBC[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -2 # force on the right        
    elseif div(tid,100)==3
        EBC[collect(1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -1 # fix left
        FBC[collect(2:nx*porder+1), :] .= -2 # force on the bottom
    end

    # 
    if tid==100
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [0, P1])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 200
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [P1, 0])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2
    
    elseif tid == 300

        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Gaussian",  [Lx*P2, Lx*5.0/6.0, 0.2*Lx])
        fext[collect(2:nx*porder+1), 1] .= F1[2:end]
        fext[collect(2:nx*porder+1), 2] .= F2[2:end]
    else
        error("tid = $tid is not understood")
    end

    dof_to_active = findall(FBC[:].==-2)
    ft = t->fext[:][dof_to_active]*sin(π*t/(T))


    npoints = (nx+1)*(ny+1)
    node_to_point = zeros(Int64, nnodes) .- 1
    pointid = 1
    for j = 1:2:(ny*porder + 1)
        for i = 1:2:(nx*porder + 1)
            node_to_point[(j-1)*(nx*porder + 1) + i] = pointid
            pointid += 1
        end
    end

    return nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point
end



force_scale = 2.0 #50
tid =200
time = 0.03
fiber_size = 1
porder = 2
T = 0.2  
NT = 200
    

P1 = 44800 /force_scale #gcm/ms^2 compress/pull
P2 =  4480 /force_scale #gcm/ms^2 bend 
P3 = 16800 /force_scale



@show P1, P2, force_scale
ngp = 3

testtype = "PlaneStress"
prop = Dict("name"=> testtype, "rho"=> 0.800, "E"=>2.0e6, "nu"=>0.2)



nxc, nyc = 10,5
nx, ny =  nxc*fiber_size, nyc*fiber_size
Lx, Ly = 0.1, 0.05 #m
nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder, Lx, Ly; force_scale=force_scale)

function Edge_func(x, y, t, tid)
    zero = similar(x) * 0.0
    if (tid == 100)
        f1 = @. zero
        f2 = @. P1*sin(π*t/(T)) + zero
    end
    if (tid == 200)
        f1 = @. P1*sin(π*t/(T)) + zero
        f2 = @. zero
    end
    if (tid == 300)
        pn, x0, σ = [Lx*P2, Lx*5.0/6.0, 0.2*Lx]
        f1 = @. zero
        f2 = @. pn * 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)*sin(π*t/(T)) 
    end

    return [f1 f2] 
end


if tid == 100
    Edge_Traction_Data = zeros(Int64, nx, 3)
    for i = 1:nx
        Edge_Traction_Data[i, :] .= i+(ny-1)*nx, 3, 100 
    end
elseif tid == 200
    Edge_Traction_Data = zeros(Int64, ny, 3)
    for i = 1:ny
        Edge_Traction_Data[i, :] .= nx+(i-1)*nx, 2, 200 
    end 
elseif tid == 300
    Edge_Traction_Data = zeros(Int64, nx, 3)
    for i = 1:nx
        Edge_Traction_Data[i, :] .= i, 1, 300 
    end
end
 
    
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
        push!(elements,FiniteStrainContinuum(coords,elnodes, prop, 3))
    end
end


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext, Edge_Traction_Data)
setGeometryPoints!(domain, npoints, node_to_point)

state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft, nothing, Edge_func)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


fold = zeros(Float64, size(domain.fext))
ID = domain.ID
nodal_force = ft(time) # user defined time-dependent boundary    
dof_id = 0
#update fext for active nodes (length of neqs)
for idof = 1:domain.ndims
    for inode = 1:domain.nnodes
        if (domain.FBC[inode, idof] == -2)
            global dof_id
            dof_id += 1
            fold[ID[inode, idof]] = nodal_force[dof_id]
        end
    end
end

fnew = getEdgeForce(domain, globdat, time)

@show norm(fold), norm(fnew)
@show norm(fold - fnew)
