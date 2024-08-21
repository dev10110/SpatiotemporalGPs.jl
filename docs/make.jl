push!(LOAD_PATH, "../")

using SpatiotemporalGPs
using Documenter

DocMeta.setdocmeta!(
    SpatiotemporalGPs,
    :DocTestSetup,
    :(using SpatiotemporalGPs);
    recursive = true,
)

makedocs(;
    modules = [SpatiotemporalGPs],
    authors = "Devansh Ramgopal Agrawal <devansh@umich.edu> and contributors",
    sitename = "SpatiotemporalGPs.jl",
    format = Documenter.HTML(;
        canonical = "https://dev10110.github.io/SpatiotemporalGPs.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Kalman Filters" => "kf.md",
        "API" => ["Kalman Filters" => "api/kf.md", "STGPKF" => "api/stgpkf.md"],
    ],
)

deploydocs(; repo = "github.com/dev10110/SpatiotemporalGPs.jl", devbranch = "main")
