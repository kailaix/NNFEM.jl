
@testset "compute_body_force_terms1" begin 

    m = 10
    n = 10
    h = 1.0
    domain = example_static_domain1(m, n, h)
    fext = compute_body_force_terms1(domain, (x,y)->x^2+y^2+1)
    f = eval_f_on_gauss_pts((x,y)->x^2+y^2+1, m, n, h)
    pf_f = compute_fem_source_term1(f, m, n, h)
    @test fext ≈ pf_f

end

@testset "compute_stiffness_matrix_and_internal_force1" begin 
    m = 10
    n = 10
    h = 1.0
    domain = example_static_domain1(m, n, h)
    K, fint = compute_stiffness_matrix_and_internal_force1(domain)
    pf_K = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
    @test pf_K≈K
end

@testset "compute_stiffness_matrix_and_internal_force1" begin 
    m = 10
    n = 10
    h = 1.0
    domain = example_static_domain1(m, n, h)
    N = getNGauss(domain)
    k = zeros(N, 2, 2)
    for i = 1:N 
        k[i,:,:] = diagm(0=>ones(2))
    end
    init_nnfem(domain)
    K = s_compute_stiffness_matrix1(k, domain)
    Kref, _ = compute_stiffness_matrix_and_internal_force1(domain)
    K = run(sess, K)
    @test Kref≈K
end

