using RecipesBase

function get_axes(problem::STGPKFProblem)
    if length(problem.pts[1]) != 2
        error("problem.pts must be 2 dimensional")
    end

    xs = [p[1] for p in problem.pts] |> unique |> sort
    ys = [p[2] for p in problem.pts] |> unique |> sort

    if (length(xs) * length(ys)) != length(problem.pts)
        error("Input problem.pts isnt on a regular grid")
    end

    #TODO(dev): reenable this check!
    # grid_pts = reshape(problem.pts, length(xs), length(ys))

    # # check if the reshaping worked correctly
    # for (i, x) in enumerate(xs)
    #     for (j, y) in enumerate(ys)
    #         @assert grid_pts[i, j] == SVector([x, y])
    #     end
    # end

    return xs, ys
end

"""
    quantile(μ, σ, q)

For a normal distribution with mean μ and standard deviation σ, this function returns the q-th quantile.
"""
function quantile(μ, σ, q)
    return μ + sqrt(2) * σ * SpecialFunctions.erfinv(-1 + 2 * q)
end

@recipe function f(
        problem::STGPKFProblem, state::KFState; plot_type = :estimate, percentile = 0.95)

    # check that only one of the plotting options is selected
    if !(plot_type in (:estimate, :std, :clarity, :percentile))
        error("plot_type must be one of (:estimate, :std, :clarity)")
    end

    checkdims(problem, state)

    # extract the x and y axes from the problem
    xs, ys = get_axes(problem)
    if plot_type == :estimate
        est = get_estimate(problem, state)
        M = reshape(est, length(xs), length(ys))
        title = "estimate"
    elseif plot_type == :std
        σs = get_estimate_std(problem, state)
        M = reshape(σs, length(xs), length(ys))
        title = "std"
    elseif plot_type == :clarity
        qs = get_estimate_clarity(problem, state)
        M = reshape(qs, length(xs), length(ys))
        title = "clarity"
    elseif plot_type == :percentile
        ps = get_estimate_percentile(problem, state, percentile)
        M = reshape(ps, length(xs), length(ys))
        title = "$(round(100*percentile))% percentile"
    end

    @series begin
        seriestype --> :heatmap
        xlabel --> "x"
        ylabel --> "y"
        title --> title
        xs, ys, M'
    end
end

@recipe function f(sp_data::SpatiotemporalData2D, time)

    # check that the time is in the data
    if time ∉ sp_data.ts
        error("time must be in the data")
    end

    # extract the x and y axes from the problem
    xs = sp_data.xs
    ys = sp_data.ys

    # get the data at the time
    M = sp_data.data[:, :, findfirst(sp_data.ts .== time)]

    @series begin
        seriestype --> :heatmap
        xlabel --> "x"
        ylabel --> "y"
        title --> "data at t = $time"
        xs, ys, M'
    end
end
