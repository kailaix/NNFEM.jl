function gradtest(f, x0, n=nothing)
    v0 = rand(Float64,size(x0))
    γs = 0.001 ./10 .^(1:5)
    err1 = []
    err2 = []
    f0, J = f(x0)
    q0 = rand(Float64,size(f0))
    f0 = f0'*q0
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        f1 = f1'*q0
        
        push!(err1, norm(f0-f1))
        push!(err2, norm(f1-f0-q0'*J*v0*γs[i]))
        @show norm(f0-f1),norm(f1-f0-q0'*J*v0*γs[i])
    end
    close("all")
    loglog(γs, err1, "*-", label="finite difference")
    loglog(γs, err2, "+-", label="Jacobian")
    loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
end