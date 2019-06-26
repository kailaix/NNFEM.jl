using JLD2

function save(file::String, domain::Domain, globaldata::GlobalData)
    @save file domain globaldata
end

function load(file::String)
    @load file domain globaldata
    return domain, globaldata
end