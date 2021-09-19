using Revise
using NNFEM
using LinearAlgebra
using PyPlot

"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type: a string of load type, which can be "constant", or "Gaussian"
args: an array, 
      for Constant load, it has p1 and p2, in tangential and normal direction
      for "Gaussian" load, it has p, x0, and σ, the force is in the normal direction
                  
"""
function ComputeLoad(coords, porder, ngp, f_px, f_py)
    L = 1.0

    @assert(ngp <= 4)
    # coords is an array of N by 2
    @assert( all(coords[:, 2] .== 1.0))
    @assert(coords[1, 1] == L && coords[end, 1] == 0.0)
    

    npoints = size(coords, 1)
    nelems = div(npoints - 1, porder)
    Ft, Fn = zeros(Float64, npoints), zeros(Float64, npoints)
    elem_coords = zeros(Float64, porder + 1, 2)


    # assemble the force

    @show nelems
    for e = 1:nelems  # element from left to right
        
        if porder == 1
            loc_id = [e, e+1]
        elseif porder == 2
            loc_id = [2*(e-1)+3, 2*(e-1)+1, 2*(e-1)+2]
        else
            error("porder error porder ==", porder)
        end
        loc_xx = coords[loc_id, 1]
        elem_coords[:,1] .= loc_xx
        

        #return list 
        weights, hs = get1DElemShapeData( elem_coords, ngp)  
        
        for igp = 1:ngp
            gp_xx = loc_xx' * hs[igp]   #x-coordinates of element Gaussian points
            Ft[loc_id] += f_px(gp_xx) * hs[igp] * weights[igp]
            Fn[loc_id] += f_py(gp_xx) * hs[igp] * weights[igp]

            # @info " elem: ", e, " igp: ", igp, " gp_xx ", gp_xx
            # @info " fy: ", f_py(gp_xx) * hs[igp] * weights[igp]

        end
    end

    return Ft, Fn
end



porder = 2
ngp = porder + 1
meshfile = "square-circle-coarse-o$(porder)"
nodes, quads, boundaries = readMesh(meshfile*".msh")

"""
Property:
The matrix is made of titanium, which are assumed to be elasto-plastic titanium material,  
time ms
ρ = 4.5 g/cm^3;  E = 100GPa = 	100*10^9 Kg/m/s^2 = 10^6 g/cm/ms^2, K=10e+9Pa =  10^5 g/cm/ms^2
ν =0.2   σY=970 MPa = 9700

Fibers are made of SiC, which are assumed to be isotropic and elastic, with
https://www.azom.com/properties.aspx?ArticleID=42
ρ = 3.2 g/cm^3  E = 400GPa =  4*10^6 g/cm/ms^2  ν = 0.35
length scale cm
"""
# prop1 = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2, "sigmaY"=>0.97e+4, "K"=>1e+5)
prop2 = Dict("name"=> "PlaneStress", "rho"=> 3.2, "E"=>4e6, "nu"=>0.35)
# prop1 = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2)

# testtype = "PlaneStressIncompressibleRivlinSaunders"
#https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/bmk/default.htm?startat=ch01s01ach08.html
prop1 = Dict("name"=> "PlaneStressIncompressibleRivlinSaunders", "rho"=> 0.800, "C1"=>0.1863e6, "C2"=>0.00979e6)
# prop2 = prop1
# using Plots
# ele_type = "\"Square\""
# ele_type = "\"Circle\""
# elnodes = quads[ele_type][12]
# coords = nodes[elnodes,:]
# plot!(coords[:,1], coords[:,2], arrow=true)
# construct elements
elements = []
for ele_type in ["\"Square\"", "\"Circle\""]
    for elnodes in quads[ele_type]
        coords = nodes[elnodes,:]
        # Square=> matrix, Circle=> fiber
        prop = (ele_type == "\"Square\"" ? prop1 : prop2)
        push!(elements,SmallStrainContinuum(coords, elnodes, prop, ngp))
    end
end

ndofs = 2
nnodes = size(nodes, 1)
EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

# fix bottom
bottom_edge = zeros(Int64, length(boundaries["\"DirichletBottom\""])*porder+1)
for i = 1:length(boundaries["\"DirichletBottom\""])

    @assert(i == 1 || bottom_edge[porder*(i-1)+1] == boundaries["\"DirichletBottom\""][i][1])
    if porder == 1
        bottom_edge[i:i+1] = boundaries["\"DirichletBottom\""][i]
    else
        bottom_edge[2i-1:2i+1] = boundaries["\"DirichletBottom\""][i][[1,3,2]]
    end
end
EBC[bottom_edge, :] .= -1 

# using Plots
# coords = nodes[bottom_edge,:]
# plot!(coords[:,1], coords[:,2], arrow=true)



# force on the top
top_edge = zeros(Int64, length(boundaries["\"NeumannTop\""])*porder+1)
for i = 1:length(boundaries["\"NeumannTop\""])
    @assert(i == 1 || top_edge[porder*(i-1)+1] == boundaries["\"NeumannTop\""][i][1])
    if porder == 1
        top_edge[i:i+1] = boundaries["\"NeumannTop\""][i]
    else
        top_edge[2i-1:2i+1] = boundaries["\"NeumannTop\""][i][[1,3,2]]
    end
end
FBC[top_edge, :] .= -1

# using Plots
# coords = nodes[top_edge,:]
# plot!(coords[:,1], coords[:,2], arrow=true)

# construct pressure load function
# if type == "Constant"
#     pt, pn = args
#     f_px, f_py = x->pt, x->pn
# elseif type == "Gaussian"
#     pn, x0, σ = args
#     f_px, f_py = x->0, x-> pn * 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)
# else
#     error("Force load type is not recogonized ", type)
# end
f_px, f_py = x->0, x-> 100.0
Ft, Fn =  ComputeLoad(nodes[top_edge, :], porder, ngp, f_px, f_py)
fext[top_edge, 1] .= Ft
fext[top_edge, 2] .= Fn
        


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
# setGeometryPoints!(domain, npoints, node_to_point)



state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
gt, ft = nothing, nothing
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


@info "fext norm = ", norm(domain.fext)


globdat, domain = ImplicitStaticSolver(globdat, domain; N= 10, ε = 1.e-6, maxiterstep=100)

# # error()
# # todo write data
# if !isdir("$(@__DIR__)/Data/order$porder")
#     mkdir("$(@__DIR__)/Data/order$porder")
# end
# if !isdir("$(@__DIR__)/Debug/order$porder")
#     mkdir("$(@__DIR__)/Debug/order$porder")
# end

# write_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat", domain)
# @save "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain

close("all")
visσ(domain, ngp, 50, 145)
axis("equal")
savefig("Fig/"*meshfile*".png")

# close("all")
# ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
# plot(ux)
# savefig("Debug/order$porder/ux$(tid)_$(force_scale)_$(fiber_size).png")

# close("all")
# uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
# plot(uy)
# savefig("Debug/order$porder/uy$(tid)_$(force_scale)_$(fiber_size).png")

