
# αscheme 
using PoreFlow
using ADCME
using PyPlot


m = 15
n = 15
h = 1/m 
NT = 100
Δt = 1/NT
bd = bcnode("all", m, n, h)
function solver(A, rhs, i)
    A, Abd = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h)
    tf.compat.v1.add_to_collection("Abd", Abd[nbd,:])
    rhs = rhs - Abd * abd[i]
    rhs = scatter_update(rhs, [bd; bd .+ (m+1)*(n+1)], abd[i]) 
    # op = tf.print(i, sum(Abd * abd[i]))
    # rhs = bind(rhs, op)
    return A\rhs
end

M = compute_fem_mass_matrix(m, n, h)
K = compute_fem_stiffness_matrix(H, m, n, h)
F = zeros(NT, 2(m+1)*(n+1))

nbd = ones(Bool, (m+1)*(n+1)*2)
nbd[[bd; bd.+(m+1)*(n+1)]] .= false
ts = αscheme_time(Δt*ones(NT); ρ=0.0)
for i = 1:NT 
    t = ts[i]
    f1 = eval_f_on_gauss_pts((x,y)->(x^2 + y^2)*exp(-t) - 7.90123456790123*exp(-t), m, n, h)
    f2= eval_f_on_gauss_pts((x,y)->(x^2 + y^2)*exp(-t) - 7.90123456790123*exp(-t), m, n, h)
    F[i,:] = compute_fem_source_term(f1*0.1, f2*0.1, m, n, h)
    
    F[i, [bd; bd.+(m+1)*(n+1)]] .= 0.0
    # @info sum(F[i,:])
    # globdat.time = t 
    # fbody = getBodyForce(domain, globdat)
    # g = zeros((m+1)*(n+1)*2)
    # g[nbd] = fbody
    # @info norm(F[i,:] - g[nbd])
end
x = Float64[]
y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
abd = zeros(NT, 2length(bd))
for i = 1:NT 
    t = ts[i]
    abd[i, :] = [(@. x[bd]^2+y[bd]^2);(@. x[bd]^2-y[bd]^2)]* exp(-t) * 0.1
end
abd = constant(abd)

dbd = zeros(NT, 2length(bd))
for i = 1:NT 
    t = ts[i]
    dbd[i, :] = [(@. x[bd]^2+y[bd]^2);(@. x[bd]^2-y[bd]^2)]* exp(-t) * 0.1
end


d0 = @. [x^2+y^2;x^2-y^2]* 0.1
u0 =  @. -[x^2+y^2;x^2-y^2]* 0.1
a0 =  @. [x^2+y^2;x^2-y^2]* 0.1
d, u, a = αscheme(M, spzero(2(m+1)*(n+1)), K, F, d0, u0, a0, Δt*ones(NT); extsolve=solver, ρ = 0.0  )
# d, u, a = fast_αscheme(m, n, h, M, spzero(2(m+1)*(n+1)), K, F, d0, u0, a0, Δt, bd, dbd, -dbd, abd )

sess = Session(); init(sess)
D = run(sess, d)
# PoreFlow.visualize_displacement(D, m, n, h)

plot(D[:,(div(n,2)+1)*(m+1)+div(m,2)+1])

ts = LinRange(0,1,NT+1)
x0 = div(m,2)*h 
y0 = (div(n,2)+1)*h 
plot((@. (x0^2+y0^2)*exp(-ts))*0.1,"--")