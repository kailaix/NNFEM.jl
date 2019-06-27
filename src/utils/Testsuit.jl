export gradtest

function gradtest(f, x0, n=nothing)
    v0 = rand(Float64,size(x0))
    γs = 100 ./10 .^(1:7)
    err2 = []
    f0, J = f(x0)
    for i = 1:7
        f1, _ = f(x0+γs[i]*v0)
        f2, _ = f(x0-γs[i]*v0)
        # push!(err2, norm((f1-f2)/(2γs[i])-J*v0))
        push!(err2, norm(f1-f2-2γs[i]*J*v0))
        # push!(err2, norm((f1-f2)/(2γs[i])-J*v0))
        @info "test ", f1, f2, f1-f2
    end
    loglog(γs, err2)
    loglog(γs, γs.^3 * 0.5*abs(err2[1])/γs[1]^3, "--",label="\$\\mathcal{O}(\\gamma^3)\$")
    # loglog(γs, γs * 0.5*abs(err2[2])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
end