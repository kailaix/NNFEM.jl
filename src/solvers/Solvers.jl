export ExplicitSolver
function ExplicitSolver(Δt, globdat, domain)
    globdat.time  += Δt
    
    disp = globdat.state
    velo = globdat.velo
    acce = globdat.acce

    fext = globdat.fext
    
    velo += 0.5*Δt * acce
    disp += Δt * velo
    
    fint  = assembleInternalForce( globdat, domain )

    if length(globdat.M)==0
        error("globalDat is not initialized, call `assembleMassMatrix!(globaldat, domain)`")
    end
    acce = (fext-fint)./globdat.Mlumped
       
    velo += 0.5 * Δt * acce

    globdat.acce[:] = acce[:]
  
    # globdat.elements.commitHistory()
end