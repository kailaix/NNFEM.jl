@testset "getGaussPoints" begin 
    domain = example_domain(1, 1, 1.0)
    @test getGaussPoints(domain) ≈ [0.21132486540518713 0.21132486540518713; 0.21132486540518713 0.7886751345948129; 0.7886751345948129 0.21132486540518713; 0.7886751345948129 0.7886751345948129]
end 

@testset "getStrain" begin
    domain = example_domain(10,10,0.1)
    domain.state = rand(domain.neqs)
    E = getStrain(domain)
    E2 = eval_strain_on_gauss_pts(domain.state, 10,10,0.1)
    @test E≈E2
end 