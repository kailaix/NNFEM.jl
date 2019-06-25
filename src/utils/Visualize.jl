using PyPlot
export visualize
function visualize(domain::Domain)
    u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    nodes = domain.nodes
    u = nodes[:,1] + u; v = nodes[:,2] + v
    scatter(u, v)
end