domain = example_domain(10,10,0.1)
globdat = example_global_data(domain)

m = 10
n = 10
EBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
for j = 1:n+1
    EBC[(j-1)*(m+1) + m+1, :] .= -1
    g[(j-1)*(m+1) + m+1, :] = rand(2)
end

domain = Domain(domain.nodes, domain.elements, 2, EBC, g, zeros(Int64, (m+1)*(n+1), 2), zeros((m+1)*(n+1), 2))


domain.state = rand(length(domain.state))
fint0, stiff0 = assembleStiffAndForce( domain, Δt)
bd_contrib = fint0 - stiff0 * domain.state[domain.eq_to_dof]


domain.state[domain.eq_to_dof] = rand(domain.neqs)
fint0, stiff0 = assembleStiffAndForce( domain, Δt)
bd_contrib0 = fint0 - stiff0 * domain.state[domain.eq_to_dof]
@info "bd contrib", norm(bd_contrib-bd_contrib0)



