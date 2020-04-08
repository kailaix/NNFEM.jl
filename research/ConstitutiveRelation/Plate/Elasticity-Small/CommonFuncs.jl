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
    ft = nothing
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
    elseif div(tid,100)==4
        EBC[collect(1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -1 # fix left
        EBC[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -2 #pull right
    end

    # Uniform conservative pressure load 
    # F1 = 2e2 /force_scale #gcm/ms^2 compress/pull
    # F2 = 2e1 /force_scale #gcm/ms^2 bend 

    P1 = 8 /force_scale #gcm/ms^2 compress/pull
    P2 = 0.8 /force_scale #gcm/ms^2 bend 
    P3 = 3/force_scale
    @show P1, P2
    ngp = 3
    #Bending or Pulling
    # 
    if tid == 200
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [P1, 0])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2
    
    elseif tid == 400
        g[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= 0.00508e-3 #m/ms
        g[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= 0
    else
        error("tid = $tid is not understood")
    end

    dof_to_active = findall(FBC[:].==-2)
    if length(dof_to_active) != 0
        ft = t->fext[:][dof_to_active] #*sin(π*t/(T))
    end

    dof_to_active = findall(EBC[:].==-2)
    if length(dof_to_active) != 0
        gt = t->[g[:][dof_to_active]*t, g[:][dof_to_active], g[:][dof_to_active]*0.0]
    end


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
