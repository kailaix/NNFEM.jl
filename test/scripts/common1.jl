using MATLAB
using Random
function matplot(domain, sol)
    nodes = domain.nodes
    @mput nodes sol
    mat"""
    x = nodes(:,1);
    y = nodes(:,2);
    tri = delaunay(x,y);
    plot(x,y,'.')
    h = trisurf(tri, x, y, sol)
    % savefig('figures/1.png')
    """
end

function matpcolor(domain, sol)
    nodes = domain.nodes
    if length(sol)!=domain.nnodes
        nodes = getGaussPoints(domain)
    end
        
    @mput nodes sol

    mat"""
    x = nodes(:,1);
    y = nodes(:,2);
    d = linspace(-1,1,500);
    [xq,yq] = meshgrid(d,d);
    vq = griddata(x,y,sol,xq,yq);
    idx = (xq<0) & (yq<0);
    vq(idx) = nan;
    h = pcolor(xq, yq, vq)
    set(h, 'EdgeColor', 'none');
    colorbar()
    % savefig('figures/1.png')
    """
end

function mathold()
    mat"""hold;"""
end

function matscatter(x, y)
    @mput x y
    mat"""
    scatter(x, y, 'r')
    """
end

function sample_interior(N, n, bd)
    Random.seed!(233+n)
    s = Set([])
    while length(s)<n 
        i = rand(1:N)
        if i in s || i in bd 
            continue 
        else
            push!(s, i)
        end
    end
    Int64.(collect(s))
end