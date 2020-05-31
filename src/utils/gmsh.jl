export gmsh, init_gmsh, addPoint, addLine, addCurveLoop,
addPlaneSurface, addPhysicalGroup, setPhysicalName, addCircleArc, addSurfaceFilling,
embedPointInLine, embedPointInSurface, embedLine, sync, finalize_gmsh, meshsize, addCircle, compound,
get_entities


include(joinpath(@__DIR__, "..", "..", "deps", "Gmsh", "gmsh.jl"))
import .gmsh

gmsh_number = 1
is_sync = false

function init_gmsh(name::String = "default")
    global is_sync = false
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("General.Terminal", gmsh_number)
    global gmsh_number += 1
    gmsh.model.add(name)
end 

function addPoint(x, y, meshSize = 0., tag = -1)
    gmsh.model.geo.addPoint(x, y, 0.0, meshSize, tag)
end

function addLine(startTag, endTag, tag = -1)
    gmsh.model.geo.addLine(startTag, endTag, tag)
end

"""
Add a curve loop (a closed wire) formed by the curves curveTags.
"""
function addCurveLoop(curveTags, tag = -1)
    gmsh.model.geo.addCurveLoop(curveTags, -1)
end

function addPlaneSurface(wireTags, tag = -1)
    gmsh.model.geo.addPlaneSurface(wireTags,  -1)
end

function addPhysicalGroup(dim, tags, tag = -1)
    gmsh.model.addPhysicalGroup(dim, tags,  -1)
end

function setPhysicalName(dim, tag, name)
    gmsh.model.setPhysicalName(dim, tag, name)
end


function addCircleArc(startTag, centerTag, endTag, tag = -1)
    gmsh.model.geo.addCircleArc(startTag, centerTag, endTag, tag)
end

function addSurfaceFilling(wireTags, tag = -1)
    gmsh.model.geo.addSurfaceFilling(wireTags, tag, -1)
end

function embedPointInLine(tags, inTag)
    sync()
    gmsh.model.mesh.embed(0, tags, 1, inTag)
end

function embedPointInSurface(tags, inTag)
    sync()
    gmsh.model.mesh.embed(0, tags, 2, inTag)
end

function embedLine(tags, inTag)
    sync()
    gmsh.model.mesh.embed(1, tags, 2, inTag)
end

function meshsize(fn::String)
    sync()
    field = gmsh.model.mesh.field
    field.add("MathEval", 1)
    field.setString(1, "F", fn)
    field.setAsBackgroundMesh(1)
end

function sync()
    global is_sync = true
    gmsh.model.geo.synchronize()
end

function addCircle(x, y, r, tag=-1)
    pts = [
        addPoint(x, y-r)
        addPoint(x+r, y)
        addPoint(x,y+r)
        addPoint(x-r,y)
        addPoint(x,y)
    ]
    arcs = [
        addCircleArc(pts[1],pts[5],pts[2])
        addCircleArc(pts[2],pts[5],pts[3])
        addCircleArc(pts[3],pts[5],pts[4])
        addCircleArc(pts[4],pts[5],pts[1])
    ]
    addCurveLoop(arcs)
end

function get_entities(tag = -1)
    gmsh.model.getEntities(tag)
end

function compound(tags)
    sync()
    gmsh.model.mesh.setCompound(2, tags)
end
function finalize_gmsh(visualize=false)
    sync()
    ss = get_entities(2)
    for (_, s) in ss
        gmsh.model.mesh.setRecombine(2, s)
    end
    gmsh.model.mesh.generate(2)
    name = tempname()*".msh"
    gmsh.write(name)
    if visualize
        gmsh.fltk.run()
    end
    gmsh.finalize()
    return name
end