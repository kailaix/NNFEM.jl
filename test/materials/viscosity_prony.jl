using Revise
using NNFEM 
using ADCME 
using ADCMEKit
using ProgressMeter

m = 40
n = 20
h = 1/n
NT = 100
Δt = 1/NT 
tau = 0.5
c = 1.0

elements = []
prop = Dict("name"=> "PlaneStrainViscoelasticityProny", "rho"=> 2000.0, "E"=> 37844812616.2842, "nu"=> 0.307699122884735, "tau"=>tau, "c"=>c)
coords = zeros((m+1)*(n+1), 2)
for j = 1:n
    for i = 1:m
        idx = (m+1)*(j-1)+i 
        elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
        ngp = 2
        nodes = [
        (i-1)*h (j-1)*h
        i*h (j-1)*h
        i*h j*h
        (i-1)*h j*h
        ]
        coords[elnodes, :] = nodes
        push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
    end
end


# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)

for i = 1:m+1
    idx = n*(m+1)+i 
    EBC[idx,:] .= -1 
end

# for j = 1:n
#     idx = (j-1)*(m+1) + m + 1
#     FBC[idx, :] .= -1
#     f[idx,:] = [-1.0;0.0]
# end


Edge_traction_data = []
IdDict = Dict(
    (1,2)=>1,
    (2,3)=>2,
    (3,4)=>3,
    (1,4)=>4
)
for k = 1:length(elements)
    e = elements[k]
    if sum(e.coords[:,1] .> 2.0 - 1e-3)==2
        i = findall(e.coords[:,1] .> 2.0 - 1e-3)
        push!(Edge_traction_data, [k IdDict[(minimum(i), maximum(i))] 0])
    end
end
Edge_traction_data = Int64.(vcat(Edge_traction_data...))

function Edge_func(x, y, time, id)
    return [-1e6 * ones(length(x)) zeros(length(x))]
end

ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f, Edge_traction_data)
globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, nothing, nothing, nothing, Edge_func)

# LinearGeneralizedAlphaSolverInit!(globaldata, domain, Δt)
for i = 1:NT
    # global globaldata, domain = LinearGeneralizedAlphaSolverStep(globaldata, domain, Δt)
    global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt)
end


# visualize_von_mises_stress_on_scoped_body(getStateHistory(domain), domain; scale_factor=9.0e7, vmin=0.012, vmax=27.992)
figure()
visualize_von_mises_stress_on_scoped_body(getStateHistory(domain)[end:end,:], domain; 
                frames=1, scale_factor=8.3e2, vmin=2339, vmax=3.56e6)
savefig("prony_vm.png")
figure()
visualize_total_deformation_on_scoped_body(getStateHistory(domain)[end:end,:], domain; 
                frames=1)
savefig("prony_td.png")

figure()
visualize_x_deformation_on_scoped_body(getStateHistory(domain)[end:end,:], domain; 
                frames=1, scale_factor=8.3e2)
savefig("prony_x.png")

figure()
visualize_y_deformation_on_scoped_body(getStateHistory(domain)[end:end,:], domain; 
                frames=1, scale_factor=8.3e2)
savefig("prony_y.png")

            