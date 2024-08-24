
function SpatiotemporalData(grid_pts, ts, data)
    return SpatiotemporalData(grid_pts, ts, data, nothing)
end

function SpatiotemporalData2D(xs, ys, ts, data)
    itp = Interpolations.cubic_spline_interpolation((xs, ys, ts), data)
    return SpatiotemporalData2D(xs, ys, ts, data, itp)
end

function generate_temporal_process(N, ΔT, kt::AbstractMaternKernel)
    ss_model = state_space_model(kt, ΔT)
    Σ0 = initial_covariance(kt)

    Φ = ss_model.Φ
    W = ss_model.W
    C = ss_model.C

    lowerW = cholesky(W).L
    lowerΣ0 = cholesky(Σ0).L

    # dimension of the state
    n = size(Φ, 1)

    # allocate output
    zs = zeros(N)

    # create the first state
    x = lowerΣ0 * randn(n)
    zs[1] = (C * x)[1]

    # loop
    for i in 2:N
        x = Φ * x + lowerW * randn(n)
        zs[i] = (C * x)[1]
    end
    return zs
end

function generate_spatiotemporal_process(xs, ys, ΔT, tmax, ks, kt)

    # create a single list of grid points
    grid_pts = vec([(@SVector [x, y]) for x in xs, y in ys])

    Nx = length(xs)
    Ny = length(ys)

    F = generate_spatiotemporal_process(grid_pts, ΔT, tmax, ks, kt)

    ts = F.ts
    Nt = length(ts)

    # reshape into xs x ys x ts matrix
    M = reshape(F.data, Nx, Ny, Nt)

    data = SpatiotemporalData2D(xs, ys, ts, M)

    return data
end

function generate_spatiotemporal_process(grid_pts, ΔT, tmax, ks, kt)

    # construct the STGPKF problem
    stgpkf_problem = STGPKFProblem(grid_pts, ks, kt, ΔT)

    Ng = length(grid_pts)
    Nt = Int(ceil(tmax / ΔT))

    # create Ng independent temporal processes
    zs = [generate_temporal_process(Nt, ΔT, stgpkf_problem.kt) for i in 1:Ng]

    # stack into a matrix of size (Ng x Nt)
    Z = (hcat(zs...)')

    # multiply by the sqrt_K_gg matrix to get the f at each time (in each col)
    F = stgpkf_problem.sqrt_K_gg * Z

    return SpatiotemporalData(grid_pts, ΔT * (0:(Nt - 1)), F)
end
