### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a76f6dfc-7bdd-4110-a88f-271a17c6a7c6
begin
    projectdir = dirname(Base.current_project())

    import Pkg
    Pkg.activate(projectdir)
    Pkg.instantiate()

    using Pollen

    using DataFrames
    using CSV
    using WGLMakie
    using CairoMakie
    using Dates
    using Statistics
    using StatsBase
    using Dates
    using OrderedCollections
    using MLJ
    using Interpolations
    using Random
    using MLJXGBoostInterface
    using NearestNeighborModels
    using MLJModels
    using MLJLinearModels
    using MLJDecisionTreeInterface
    using MLJFlux
    import Flux
    using CUDA
    using cuDNN
	using FileIO
	using ImageMagick
	using Latexify
	using HypothesisTests
	using FreqTables
end

# ╔═╡ d73fd678-da81-413b-a28c-0f37ae1545bc
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 2000px;
    		padding-left: max(283px, 10%);
    		padding-right: max(383px, 10%); 
            # 383px to accomodate TableOfContents(aside=true)
		}
	}
</style>
"""

# ╔═╡ efd3112b-b0cc-4b07-a07e-f5f2c50ca743
CUDA.versioninfo()

# ╔═╡ 47689b8c-c6e8-472b-8108-736b5bd86e52
device = Flux.gpu_device()

# ╔═╡ a5a52cf3-2088-4b23-bbdb-213bde5ba851
begin
    datadir = joinpath(projectdir, "data")
    aerodir = joinpath(datadir, "aero")
    meteodir = joinpath(datadir, "meteo", "ecad", "balice")
	assetsdir = joinpath(projectdir, "assets")
	exportdir = joinpath(datadir, "export")
end

# ╔═╡ bcf31b99-c99d-45ef-b163-b23dab2c224c
seedsfile = joinpath(aerodir, "baza_10_24.csv")

# ╔═╡ a88b7a62-730b-4499-810b-db32c4d27194
seedsdf_raw = CSV.File(seedsfile; delim=";") |> DataFrame

# ╔═╡ db7bc4ec-3d7f-4733-8760-98e569b7f15c
begin
    seedsdf = select(seedsdf_raw, [:DATA, :BETULA, :POACEAE])

    rename!(seedsdf, :DATA => :date, :BETULA => :betula, :POACEAE => :poaceae)

    seedsdf.date = map(seedsdf.date) do ds
        if ismissing(ds)
            missing
        else
            try
                Date(ds, dateformat"yyyy.mm.dd")
            catch
                Date(ds, dateformat"dd.mm.yyyy")
            end
        end
    end |> collect

    seedsdf = seedsdf[seedsdf.date.>=Date(1991, 1, 1).&&seedsdf.date.<=Date(2024, 8, 31), :]
    # seedsdf = seedsdf[seedsdf.date .>= Date(1998, 1, 1) .&& seedsdf.date .<= Date(2024, 8, 31), :]

    indices = filter(1:nrow(seedsdf)) do i
        !ismissing(seedsdf.betula[i]) || !ismissing(seedsdf.poaceae[i])
    end |> collect
    seedsdf = seedsdf[indices, :]

    sort!(seedsdf, [:date])

    seedsdf
end

# ╔═╡ 809e2431-5589-4182-89d0-9c219932e1d2
begin
    betuladf = select(seedsdf, [:date, :betula]) |> dropmissing
    rename!(betuladf, :betula => :seeds_m3_24h)
    betuladf.month = Dates.month.(betuladf.date)
    betuladf.week = Dates.week.(betuladf.date)


    poaceaedf = select(seedsdf, [:date, :poaceae]) |> dropmissing
    rename!(poaceaedf, :poaceae => :seeds_m3_24h)
    poaceaedf.month = Dates.month.(poaceaedf.date)
    poaceaedf.week = Dates.week.(poaceaedf.date)

    # betuladf, poaceaedf
    nothing
end

# ╔═╡ 75c9a372-81c4-4380-b413-6ab254283a1b
begin
    year2df = OrderedDict()

    for (df, species) in [(betuladf, :betula), (poaceaedf, :poaceae)]
        year2df[species] = OrderedDict()
        for currentyear in unique(year.(df.date))
            cdf = df[year.(df.date).==currentyear, :]

            itp = interpolate((Dates.dayofyear.(cdf.date),), cdf.seeds_m3_24h, Gridded(Linear()))

            olddates = sort(cdf.date)
            startdate = first(olddates)
            enddate = last(olddates)

            dates = Date[]
            seeds = Float64[]

            for (i, day) in enumerate(startdate:Day(1):enddate)
                seed = itp[Dates.dayofyear(day)]
                push!(dates, day)
                push!(seeds, seed)
            end

            year2df[species][currentyear] = DataFrame(:date => dates, :seeds_m3_24h => seeds, :dayofyear => dayofyear.(dates))
        end
    end

    # year2df
end

# ╔═╡ 44e4c211-cb09-4d2e-aa92-5048f3f0357b
begin
	exportfeatures = [:date, :seeds_m3_24h, :cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
	exportyears = [2022, 2023]
	for taxon in [:betula, :poaceae]
		for year in exportyears
			exportdf = year2df[taxon][year][!, exportfeatures]
			CSV.write(joinpath(exportdir, string(taxon, "_", year, ".csv")), exportdf)
		end
	end
end

# ╔═╡ 7f9efe69-448e-4ff9-8556-78b580da2383
let
    # WGLMakie.activate!()
    CairoMakie.activate!()
    # set_theme!(theme_dark())
    set_theme!(theme_light())

	firstyear_value = Dates.value(Year(seedsdf.date |> first))
	lastyear_value = Dates.value(Year(seedsdf.date |> last))
	yearsdiff = lastyear_value - firstyear_value

	fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 3.5 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)
    ax1 = Axis(
		fig[1, 1], 
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="A",
		# title="Daily birch pollen concentration",
		# subtitle="Collected at the Jagiellonian University Collegium Medicum in the years 1991 - 2024",
		titlefont="Times New Roman",
		titlecolor=:gray25,
		titlesize=fontsize,
		xticklabelsvisible = false,
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)
	step = 9.05 / yearsdiff
	intervals = 0.475:step:9.575
	ax1a = Axis(
		fig[1, 1], 
		xticks = (
			intervals, 
			map(intervals) do x
				  string(Int(round((x - 0.475) * (1 / step); digits=0) + firstyear_value))
			end
		),
		yticklabelsvisible = false,
		xlabel = "year",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)

    for (year, df) in year2df[:betula]
        lines!(
            ax1, df[!, :date], df[!, :seeds_m3_24h];
            color=:khaki4,
            linewidth=1.0,
        )
    end

	ax2 = Axis(
		fig[2, 1], 
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="B",
		titlefont="Times New Roman",
		titlecolor=:gray25,
		titlesize=fontsize,
		xticklabelsvisible = false,
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)
	step = 9.0 / yearsdiff
	intervals = 0.48:step:9.575
	ax2a = Axis(
		fig[2, 1], 
		xticks = (
			intervals, 
			map(intervals) do x
				  string(Int(round((x - 0.48) * (1 / step); digits=0) + firstyear_value))
			end
		),
		yticklabelsvisible = false,
		xlabel = "year",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)

    for (year, df) in year2df[:poaceae]
        lines!(
            ax2, df[!, :date], df[!, :seeds_m3_24h];
            color=:springgreen4,
            linewidth=1.0
        )
    end

	save(joinpath(assetsdir, "Fig1a.png"), fig, px_per_unit=1)
	img = FileIO.load(joinpath(assetsdir, "Fig1a.png"))
	save(joinpath(assetsdir, "Fig1a.tif"), img)

    fig
end

# ╔═╡ c849c164-c932-4fcd-89cf-c4a901a6d9a2
let
    CairoMakie.activate!()
    set_theme!(theme_light())

	betuladf = vcat(values(year2df[:betula])...)
	betulastats = combine(groupby(betuladf, :dayofyear)) do sub
	    (; dayofyear = first(sub.dayofyear),
	       median = median(sub.seeds_m3_24h),
	       q1 = quantile(sub.seeds_m3_24h, 0.25),
	       q3 = quantile(sub.seeds_m3_24h, 0.75))
	end

	fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 3.5 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black, figure_padding = 25)
    ax1 = Axis(
		fig[1, 1], 
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="A",
		titlefont="Times New Roman",
		titlecolor=:gray25,
		titlesize=1.0fontsize,
		xticklabelsvisible = true,
		limits=((60, 270), (0, 400)),
		xticks = 60:30:270,
		xlabel = "day of year",
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2,
		yticks = 0:100:300,
	)

	lines!(ax1, betulastats.dayofyear, betulastats.median, label = "median", linewidth = 2, color = :royalblue3)
	lines!(ax1, betulastats.dayofyear, betulastats.q1, linestyle = :dash, label = "first quartile", color = :gray25)
	lines!(ax1, betulastats.dayofyear, betulastats.q3, linestyle = :dash, label = "third quartile", color = :goldenrod2)

	axislegend(ax1; position = :rt)

	poaceaedf = vcat(values(year2df[:poaceae])...)
	poaceaestats = combine(groupby(poaceaedf, :dayofyear)) do sub
	    (; dayofyear = first(sub.dayofyear),
	       median = median(sub.seeds_m3_24h),
	       q1 = quantile(sub.seeds_m3_24h, 0.25),
	       q3 = quantile(sub.seeds_m3_24h, 0.75))
	end

	ax2 = Axis(
		fig[2, 1], 
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="B",
		titlefont="Times New Roman",
		titlecolor=:gray25,
		titlesize=1.0fontsize,
		xticklabelsvisible = true,
		limits=((60, 270), (0, 100)),
		xticks = 60:30:270,
		xlabel = "day of year",
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2,
		yticks = 0:25:75,
	)

	lines!(ax2, poaceaestats.dayofyear, poaceaestats.median, label = "median", linewidth = 2, color = :springgreen4)
	lines!(ax2, poaceaestats.dayofyear, poaceaestats.q1, linestyle = :dash, label = "first quartile", color = :gray50)
	lines!(ax2, poaceaestats.dayofyear, poaceaestats.q3, linestyle = :dash, label = "third quartile", color = :yellow3)

	axislegend(ax2; position = :rt)

	save(joinpath(assetsdir, "Fig1.png"), fig, px_per_unit=1)
	img = FileIO.load(joinpath(assetsdir, "Fig1.png"))
	save(joinpath(assetsdir, "Fig1.tif"), img)

    fig
end

# ╔═╡ 6430a249-a8fd-4224-b2b4-6711d79384e1
std(betuladf.seeds_m3_24h), std(poaceaedf.seeds_m3_24h)

# ╔═╡ af5dd9ad-d16c-4ae4-965b-ffe2010195fc
betuladf |> nrow, poaceaedf |> nrow

# ╔═╡ eafaee48-5a9c-4284-ac13-709703e5106f
ShapiroWilkTest(betuladf.seeds_m3_24h)

# ╔═╡ 99aa91d4-90c5-4b14-b91f-bb0367811902
ShapiroWilkTest(poaceaedf.seeds_m3_24h)

# ╔═╡ e1023cdb-c613-48fe-9d12-f1ff371eae03
begin
	betuladescription = describe(betuladf)[1:2, :]
	betuladescription.variable = [:betula_date, :betula_seeds_m3_24h]
	poaceaedescription = describe(poaceaedf)[1:2, :]
	poaceaedescription.variable = [:poaceae_date, :poaceae_seeds_m3_24h]
	datadescription = vcat(betuladescription, poaceaedescription)
	latexify(datadescription; env = :table, booktabs = false, latex = false) |> print
end

# ╔═╡ 2d4a0d12-8416-4b7f-9cd0-2225c0601a25
combine(
    groupby(betuladf, :month),
    :date => length => :no_samples,
    :seeds_m3_24h => mean => :seeds_m3_24h_mean,
    :seeds_m3_24h => std => :seeds_m3_24h_std,
    :seeds_m3_24h => (x -> percentile(x, 5)) => :seeds_m3_24h_percentile_5,
    :seeds_m3_24h => (x -> percentile(x, 95)) => :seeds_m3_24h_percentile_95,
)

# ╔═╡ a06addad-bd13-4b5c-bbdb-f95e282f6a5d
combine(
    groupby(poaceaedf, :month),
    :date => length => :no_samples,
    :seeds_m3_24h => mean => :seeds_m3_24h_mean,
    :seeds_m3_24h => std => :seeds_m3_24h_std,
    :seeds_m3_24h => (x -> percentile(x, 5)) => :seeds_m3_24h_percentile_5,
    :seeds_m3_24h => (x -> percentile(x, 95)) => :seeds_m3_24h_percentile_95,
)

# ╔═╡ fbd8ad41-d7f9-4288-9fb7-58b1327b8e15
function meteo2df(filepath::String; stationid::Int=25133, verbose::Bool=false)::DataFrame
    df = CSV.File(filepath) |> DataFrame

    for name in names(df)
        rename!(df, name => Symbol(lowercase(strip(name))))
    end

    datetime_format = dateformat"yyyymmdd"
    df[!, :date] = Date.(string.(df[!, :date]), datetime_format)

    target = Symbol(last(split(splitext(basename(filepath))[1], "balice_")))
    target_quality = Symbol("q_", target)

    missings = combine(groupby(df, target_quality), nrow => :frequency)
    if verbose
        println("processing ", target)
        println("quality control:\n", missings)
        println(sum(df[!, target_quality] .== 9), " missing values")
        println("target mean before substitution: ", mean(df[df[!, target_quality].==9, target]))
    end
    for i in 1:nrow(df)
        if df[i, target_quality] == 9
            if i > 1
                df[i, target] = df[i-1, target]
            else
                df[i, target] = 0
            end
        end
    end
    if verbose
        println("target mean after substitution: ", mean(df[df[!, target_quality].==9, target]))
        println()
    end

    df[!, target] = convert.(Float64, df[!, target])

    @assert length(unique(df.staid)) == 1 && first(unique(df.staid)) == stationid

    df
end

# ╔═╡ 5ee2617a-7e1b-4a6b-a6bf-c1d2960c28ee
begin
    meteodfs = OrderedDict()
    for file in readdir(meteodir)
        target = Symbol(last(split(splitext(file)[1], "balice_")))
        meteodfs[target] = meteo2df(joinpath(meteodir, file))
    end
    # meteodfs
end

# ╔═╡ 6df4c654-3dbe-42e5-89e6-4953a4278852
meteodfs

# ╔═╡ 4c2e680e-d156-4767-b985-2425f8c2b390
begin
    ppdf = meteodfs[:pp]
    lastpressure = 0
    for i in 1:nrow(ppdf)
        pressure = ppdf[i, :pp]
        if pressure > 0
            lastpressure = pressure
        else
            ppdf[i, :pp] = lastpressure
        end
    end
    # ppdf
end

# ╔═╡ 8c28f97e-4a30-4270-9031-faeb2e7a221c
begin
    for (target, targetdf) in meteodfs
        for species in keys(year2df)
            for year in keys(year2df[species])
                n = nrow(year2df[species][year])
                year2df[species][year][!, target] = fill(typemin(Float64), n)
                for i in 1:n
                    index = findfirst(==(year2df[species][year][i, :date]), targetdf.date)
                    value = targetdf[findfirst(==(year2df[species][year][i, :date]), targetdf.date), target]
                    year2df[species][year][i, target] = value
                    @assert year2df[species][year][i, target] != typemin(Float64)
                end
            end
        end
    end
    # year2df
end

# ╔═╡ 8b0c719f-9721-4dac-a242-b31b4c2d7dcd
function parametrizepast!(df::DataFrame, columns::Vector{Symbol}; window::Int=7, datecolumn::Symbol=:date)
    for column in columns
        for w in 1:window
            df[!, Symbol(column, "_n", w)] = zeros(nrow(df))
        end
    end
    for i in 1:nrow(df)
        for column in columns
            for w in 1:window
                index = max(1, i - w)
                df[i, Symbol(column, "_n", w)] = df[index, column]
            end
        end
    end
end

# ╔═╡ bf10c43b-a135-439d-80cb-e7e69f1aca07
begin
    for species in keys(year2df)
        for year in keys(year2df[species])
            parametrizepast!(year2df[species][year], [:seeds_m3_24h]; window=14)
            parametrizepast!(year2df[species][year], [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]; window=14)
        end
    end
    # year2df
end

# ╔═╡ f1716b72-0982-4db4-a3df-52177a86997c
begin
    betula_years = collect(keys(year2df[:betula]))
    shuffle!(Xoshiro(35), betula_years)
    betula_ttratio = 0.7
    betula_pivotindex = floor(Int, length(betula_years) * betula_ttratio)
    betula_trainyears = betula_years[1:betula_pivotindex] |> sort
    betula_testyears = betula_years[(betula_pivotindex+1):end] |> sort

    betula_alldf = DataFrame()
	
    betula_traindf = DataFrame()
    for year in betula_trainyears
        global betula_traindf = vcat(betula_traindf, year2df[:betula][year])
        global betula_alldf = vcat(betula_alldf, year2df[:betula][year])
    end

    betula_testdf = DataFrame()
    for year in betula_testyears
        global betula_testdf = vcat(betula_testdf, year2df[:betula][year])
        global betula_alldf = vcat(betula_alldf, year2df[:betula][year])
    end
    betula_features = [
        :seeds_m3_24h,

        # :cc_n1,
        # :fg_n1,
        # :hu_n1,
        # :pp_n1,
        # :qq_n1,
        # :sd_n1,
        # :ss_n1,
        # :tg_n1,
        # :tn_n1,
        # :tx_n1,

        # :seeds_m3_24h_n35,
        # :seeds_m3_24h_n34,
        # :seeds_m3_24h_n33,
        # :seeds_m3_24h_n32,
        # :seeds_m3_24h_n31,
        # :seeds_m3_24h_n30,
        # :seeds_m3_24h_n29,
        # :seeds_m3_24h_n28,
        # :seeds_m3_24h_n27,
        # :seeds_m3_24h_n26,
        # :seeds_m3_24h_n25,
        # :seeds_m3_24h_n24,
        # :seeds_m3_24h_n23,
        # :seeds_m3_24h_n22,
        # :seeds_m3_24h_n21,
        # :seeds_m3_24h_n20,
        # :seeds_m3_24h_n19,
        # :seeds_m3_24h_n18,
        # :seeds_m3_24h_n17,
        # :seeds_m3_24h_n16,
        # :seeds_m3_24h_n15,
        # :seeds_m3_24h_n14,
        # :seeds_m3_24h_n13, 
        # :seeds_m3_24h_n12,
        # :seeds_m3_24h_n11,
        # :seeds_m3_24h_n10,
        # :seeds_m3_24h_n9,
        # :seeds_m3_24h_n8, 
        :seeds_m3_24h_n7,
        :seeds_m3_24h_n6,
        :seeds_m3_24h_n5,
        :seeds_m3_24h_n4,
        :seeds_m3_24h_n3,
        :seeds_m3_24h_n2,
        :seeds_m3_24h_n1,
    ]
    select!(betula_traindf, betula_features)
    select!(betula_testdf, betula_features)

    # betula_traindf, betula_testdf
    nothing
end

# ╔═╡ e120f506-ac7d-4dcb-9029-3148b6620630
begin
    poaceae_years = collect(keys(year2df[:poaceae]))
    shuffle!(Xoshiro(35), poaceae_years)
    poaceae_ttratio = 0.7
    poaceae_pivotindex = floor(Int, length(poaceae_years) * poaceae_ttratio)
    poaceae_trainyears = poaceae_years[1:poaceae_pivotindex] |> sort
    poaceae_testyears = poaceae_years[(poaceae_pivotindex+1):end] |> sort

	poaceae_alldf = DataFrame()

    poaceae_traindf = DataFrame()
    for year in poaceae_trainyears
        global poaceae_traindf = vcat(poaceae_traindf, year2df[:poaceae][year])
        global poaceae_alldf = vcat(poaceae_alldf, year2df[:poaceae][year])
    end

    poaceae_testdf = DataFrame()
    for year in poaceae_testyears
        global poaceae_testdf = vcat(poaceae_testdf, year2df[:poaceae][year])
        global poaceae_alldf = vcat(poaceae_alldf, year2df[:poaceae][year])
    end
    poaceae_features = [
        :seeds_m3_24h,

        # :cc_n1,
        # :fg_n1,
        # :hu_n1,
        # :pp_n1,
        # :qq_n1,
        # :sd_n1,
        # :ss_n1,
        # :tg_n1,
        # :tn_n1,
        # :tx_n1,

        # :seeds_m3_24h_n35,
        # :seeds_m3_24h_n34,
        # :seeds_m3_24h_n33,
        # :seeds_m3_24h_n32,
        # :seeds_m3_24h_n31,
        # :seeds_m3_24h_n30,
        # :seeds_m3_24h_n29,
        # :seeds_m3_24h_n28,
        # :seeds_m3_24h_n27,
        # :seeds_m3_24h_n26,
        # :seeds_m3_24h_n25,
        # :seeds_m3_24h_n24,
        # :seeds_m3_24h_n23,
        # :seeds_m3_24h_n22,
        # :seeds_m3_24h_n21,
        # :seeds_m3_24h_n20,
        # :seeds_m3_24h_n19,
        # :seeds_m3_24h_n18,
        # :seeds_m3_24h_n17,
        # :seeds_m3_24h_n16,
        # :seeds_m3_24h_n15,
        # :seeds_m3_24h_n14,
        # :seeds_m3_24h_n13, 
        # :seeds_m3_24h_n12,
        # :seeds_m3_24h_n11,
        # :seeds_m3_24h_n10,
        # :seeds_m3_24h_n9,
        # :seeds_m3_24h_n8, 
        :seeds_m3_24h_n7,
        :seeds_m3_24h_n6,
        :seeds_m3_24h_n5,
        :seeds_m3_24h_n4,
        :seeds_m3_24h_n3,
        :seeds_m3_24h_n2,
        :seeds_m3_24h_n1,
    ]
    select!(poaceae_traindf, poaceae_features)
    select!(poaceae_testdf, poaceae_features)

    # poaceae_traindf, poaceae_testdf
    nothing
end

# ╔═╡ 9a40b000-5636-42a1-976a-4e819a865a2e
meteofeatures = Dict(
	:cc => "cloud cover", 
	:fg => "mean wind speed", 
	:hu => "humidity", 
	:pp => "mean sea level pressure", 
	:qq => "global radiation", 
	:sd => "snow depth", 
	:ss => "sunshine duration", 
	:tg => "mean temperature", 
	:tn => "minimum temperature", 
	:tx => "maximum temperature"
)

# ╔═╡ 074c97c2-55e5-42a2-9a6e-fbbaf1e84b36
meteounits = Dict(
	:cc => "okta", 
	:fg => "0.1 m/s", 
	:hu => "1 \\%", 
	:pp => "0.1 hPa", 
	:qq => "\$W/m^2\$", 
	:sd => "1 cm", 
	:ss => "0.1 hour", 
	:tg => "\$0.1^{\\circ}C\$", 
	:tn => "\$0.1^{\\circ}C\$", 
	:tx => "\$0.1^{\\circ}C\$"
)

# ╔═╡ e70d8987-c4a3-44be-8ab5-bda6fbbf9961
let
	meteocols = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
	betulastats_data = select(betula_alldf, meteocols)
	betulastats = select(describe(betulastats_data), Not(:nmissing, :eltype))
	betulastats.feature = map(meteocols) do feature
		meteofeatures[feature]
	end
	betulastats.unit = map(meteocols) do feature
		meteounits[feature]
	end
	betulastats.std = map(meteocols) do feature
		std(betula_alldf[!, feature])
	end
	betulastats.distribution = map(meteocols) do feature
		pv = ShapiroWilkTest(betula_alldf[!, feature]) |> pvalue
		if pv < 0.001
			"normal (p \$<\$ 0.001)"
		else
			"not normal (p = $(round.(pv; digits=3)))"
		end
	end
	betulastats.empty = map(x -> "", meteocols)
	for colname in names(betulastats)
		if eltype(betulastats[!, colname]) <: AbstractFloat && colname != "distribution"
			betulastats[!, colname] = round.(betulastats[!, colname]; digits=3)
		end
	end
	betulastats = select(betulastats, [:empty, :feature, :unit, :mean, :std, :min, :median, :max, :distribution])
	latexify(betulastats; env = :table, booktabs = false, latex = false) |> print
end

# ╔═╡ fb1d5dfb-46b8-4905-9701-d72fafe54de7
let
	meteocols = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
	poaceaestats_data = select(poaceae_alldf, meteocols)
	poaceaestats = select(describe(poaceaestats_data), Not(:nmissing, :eltype))
	poaceaestats.feature = map(meteocols) do feature
		meteofeatures[feature]
	end
	poaceaestats.unit = map(meteocols) do feature
		meteounits[feature]
	end
	poaceaestats.std = map(meteocols) do feature
		std(poaceae_alldf[!, feature])
	end
	poaceaestats.distribution = map(meteocols) do feature
		pv = ShapiroWilkTest(poaceae_alldf[!, feature]) |> pvalue
		if pv < 0.001
			"normal (p \$<\$ 0.001)"
		else
			"not normal (p = $(round.(pv; digits=3)))"
		end
	end
	poaceaestats.empty = map(x -> "", meteocols)
	for colname in names(poaceaestats)
		if eltype(poaceaestats[!, colname]) <: AbstractFloat && colname != "distribution"
			poaceaestats[!, colname] = round.(poaceaestats[!, colname]; digits=3)
		end
	end
	poaceaestats = select(poaceaestats, [:empty, :feature, :unit, :mean, :std, :min, :median, :max, :distribution])
	latexify(poaceaestats; env = :table, booktabs = false, latex = false) |> print
end

# ╔═╡ 6d7d9fa5-4e56-4b59-ac1f-b10118504bcc
betula_trainyears, betula_testyears, poaceae_trainyears, poaceae_testyears

# ╔═╡ 08895a92-a901-4c70-9629-d6f5bd52553e
function seed2label_betula(seed::Real)::String
    # 1–10
    # 11–75
    # > 75
    if seed >= 75
        "high"
    elseif seed >= 11
        "medium"
    else
        "low"
    end
end

# ╔═╡ 75394666-e3d6-4a74-a83d-6500108f6313
function seed2label_poaceae(seed::Real)::String
    # 1–10
    # 11–50
    # > 50
    if seed >= 50
        "high"
    elseif seed >= 11
        "medium"
    else
        "low"
    end
end

# ╔═╡ 3a7c788a-9c01-40fe-a8e3-d1c74122583f
function makelabels_betula!(df::DataFrame)
    df.label = map(df.seeds_m3_24h) do s
        seed2label_betula(s)
    end
end

# ╔═╡ b0b39260-a9b3-4aaf-b9b4-6ed00f2c1a58
function makelabels_poaceae!(df::DataFrame)
    df.label = map(df.seeds_m3_24h) do s
        seed2label_poaceae(s)
    end
end

# ╔═╡ c72fae4c-181b-4c14-b327-2e389f0f93e8
begin
	makelabels_betula!(betula_alldf)
	makelabels_poaceae!(poaceae_alldf)
	betulaft = freqtable(betula_alldf.label)
	poaceaeft = freqtable(poaceae_alldf.label)
	names(betulaft), betulaft, names(poaceaeft), poaceaeft
	DataFrame(
		:taxon => [:betula, :poaceae],
		:low => ["betulaft[2] ($(betulaft[2] / nrow(betula_alldf)))", "poaceaeft[2] ($(poaceaeft[2] / nrow(poaceae_alldf)))"],
		:medium => ["betulaft[3] ($(betulaft[3] / nrow(betula_alldf)))", "poaceaeft[3] ($(poaceaeft[3] / nrow(poaceae_alldf)))"],
		:high => ["betulaft[1] ($(betulaft[1] / nrow(betula_alldf)))", "poaceaeft[1] ($(poaceaeft[1] / nrow(poaceae_alldf)))"],
	
	)
end

# ╔═╡ b143ab24-7f2c-4f4f-bc8c-d4934382fdce
# begin
# makelabels!(betula_traindf)
# makelabels!(betula_testdf)
# coerce!(betula_traindf, :label => Multiclass)
# coerce!(betula_testdf, :label => Multiclass)

# makelabels!(poaceae_traindf)
# makelabels!(poaceae_testdf)
# coerce!(poaceae_traindf, :label => Multiclass)
# coerce!(poaceae_testdf, :label => Multiclass)

# 	betula_traindf, betula_testdf, poaceae_traindf, poaceae_testdf
# end

# ╔═╡ 894132e2-7999-4665-8a2e-473beef46ebf
begin
    betula_trainindices = 1:nrow(betula_traindf) |> vec
    betula_testindices = (nrow(betula_traindf)+1):(nrow(betula_traindf)+nrow(betula_testdf)) |> vec

    betula_ytrain, betula_Xtrain = unpack(betula_traindf, ==(:seeds_m3_24h))
    betula_ytest, betula_Xtest = unpack(betula_testdf, ==(:seeds_m3_24h))

    betula_mldf = vcat(betula_traindf, betula_testdf)
    betula_y, betula_X = unpack(betula_mldf, ==(:seeds_m3_24h))

    betula_trainindices, betula_testindices, schema(betula_mldf) |> pretty
end

# ╔═╡ 9450436a-cc60-4df3-9831-76075967a8ea
begin
    poaceae_trainindices = 1:nrow(poaceae_traindf) |> vec
    poaceae_testindices = (nrow(poaceae_traindf)+1):(nrow(poaceae_traindf)+nrow(poaceae_testdf)) |> vec

    poaceae_ytrain, poaceae_Xtrain = unpack(poaceae_traindf, ==(:seeds_m3_24h))
    poaceae_ytest, poaceae_Xtest = unpack(poaceae_testdf, ==(:seeds_m3_24h))

    poaceae_mldf = vcat(poaceae_traindf, poaceae_testdf)
    poaceae_y, poaceae_X = unpack(poaceae_mldf, ==(:seeds_m3_24h))

    poaceae_trainindices, poaceae_testindices, schema(poaceae_mldf) |> pretty
end

# ╔═╡ 410a70f5-3166-4818-b8d5-4d36b4b7b1bd
let
    betula_traindf_copy = copy(betula_traindf)
    betula_testdf_copy = copy(betula_testdf)
    poaceae_traindf_copy = copy(poaceae_traindf)
    poaceae_testdf_copy = copy(poaceae_testdf)
    makelabels_betula!(betula_traindf_copy)
    makelabels_betula!(betula_testdf_copy)
    makelabels_poaceae!(poaceae_traindf_copy)
    makelabels_poaceae!(poaceae_testdf_copy)
    betula_traindf_copy = select(betula_traindf_copy, Not(:seeds_m3_24h))
    betula_testdf_copy = select(betula_testdf_copy, Not(:seeds_m3_24h))
    poaceae_traindf_copy = select(poaceae_traindf_copy, Not(:seeds_m3_24h))
    poaceae_testdf_copy = select(poaceae_testdf_copy, Not(:seeds_m3_24h))
    CSV.write("../data/tt/betula/betula_train.csv", betula_traindf_copy)
    CSV.write("../data/tt/betula/betula_test.csv", betula_testdf_copy)
    CSV.write("../data/tt/poaceae/poaceae_train.csv", poaceae_traindf_copy)
    CSV.write("../data/tt/poaceae/poaceae_test.csv", poaceae_testdf_copy)
end

# ╔═╡ db84ef65-0c01-44c4-85fb-230241392434
# models(matching(betula_Xtrain, betula_ytrain))

# ╔═╡ 3e7f3a3d-26ad-41ec-a90f-ef463ccf009d
begin
    struct EncoderRegressor{E,C}
        encoder::E
        regressor::C
    end

    Flux.@functor EncoderRegressor

    function (m::EncoderRegressor)(x)
        state = m.encoder(x)[:, end]
        Flux.reset!(m.encoder)
        m.regressor(state)
    end

    function buildlstm(inputs, hidden, outputs)
        # encoder = Chain(Dense(args.alphabet_len, args.N, σ), LSTM(args.N, args.N))
        encoder = Flux.LSTM(inputs, hidden)
        regressor = Flux.Dense(hidden, outputs, identity)
        return EncoderRegressor(encoder, regressor)
    end

    betula_lstmbuilder = MLJFlux.@builder begin
        buildlstm(length(betula_features) - 1, 50, 1)
    end
    poaceae_lstmbuilder = MLJFlux.@builder begin
        buildlstm(length(poaceae_features) - 1, 50, 1)
    end
end

# ╔═╡ 3e147298-e362-4eb3-9642-ff2b05927f39
begin
    #    betula_model = @load(
    # 	NeuralNetworkRegressor, pkg = "MLJFlux", verbosity=0
    # )(builder=betula_lstmbuilder, epochs=5, acceleration=CUDALibs())
    #    poaceae_model = @load(
    # 	NeuralNetworkRegressor, pkg = "MLJFlux", verbosity=0
    # )(builder=poaceae_lstmbuilder, epochs=5, acceleration=CUDALibs())

	# densebranch = Flux.Chain(
	#     Flux.Dense(10 => 256, Flux.relu)
	# ) #|> device
	# lstmbranch = Flux.Chain(
	# 	Flux.LSTM(5 => 512),
	# 	Flux.LSTM(512 => 256),
	# 	x -> x[:, end, :],
	# ) #|> device
	# builder = MLJFlux.@builder begin
	# 	Flux.Chain(
	# 		(x_feat) -> Flux.cat(
	# 			densebranch(x_feat[1:10, :]), 
	# 			lstmbranch(x_feat[11:15, :]), 
	# 			dims=1
	# 		),
	# 		Flux.Dense(512 => 256, Flux.relu),
	# 		Flux.Dense(256 => 128, Flux.relu),
	# 		Flux.Dense(128 => 1)
	# 	)
	# end
	
    # nfeatures = length(betula_features) - 1
    # builder = MLJFlux.@builder begin
    #     Flux.Chain(
    #         Flux.LSTM(nfeatures => 512),
    #         Flux.LSTM(512 => 256),
    #         x -> x[:, end, :],
    #         Flux.Dense(256 => 1)
    #     )
    # end
    # betula_model = @load(NeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
    #     builder=builder, epochs=10, acceleration=CUDALibs()
    # )
    # poaceae_model = @load(NeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
    #     builder=builder, epochs=10, acceleration=CUDALibs()
    # )

	betula_model = @load(DecisionTreeRegressor, pkg = "DecisionTree", verbosity = 0)()
	poaceae_model = @load(DecisionTreeRegressor, pkg = "DecisionTree", verbosity = 0)()

    # model = @load(DecisionTreeRegressor, pkg = "DecisionTree", verbosity=0)()
    # model = @load(XGBoostRegressor, pkg = "XGBoost", verbosity=0)(max_depth=100)
    # model = @load(RandomForestRegressor, pkg = "DecisionTree", verbosity=0)(n_trees=5, max_depth=150, sampling_fraction=0.9)
    # model = @load(KNNRegressor, pkg = "NearestNeighborModels")()
    # model = @load(DeterministicConstantClassifier, pkg = "MLJModels", verbosity=0)()

    # builder = MLJFlux.@builder begin
    # 	Chain(
    # 		Dense(ncol(X) => 64, relu),
    # 		Dense(64 => 32, relu),
    # 		Dense(32 => length(unique(y))),
    # 		softmax
    # 	)
    # end
    # model = @load(NeuralNetworkRegressor, pkg = "MLJFlux", verbosity=0)(
    # 	builder=builder, epochs=5, acceleration=CUDALibs()
    # )
end

# ╔═╡ 38e75e0e-3b5c-4679-905d-ce51af8d7930
begin
    # betula_mach = machine(betula_model, betula_Xtrain, betula_ytrain)
    # fit!(betula_mach, rows=betula_trainindices, force=true)
    betula_mach = machine(betula_model, betula_Xtrain, betula_ytrain) |> fit!
end

# ╔═╡ a26ac5a7-2e69-4572-8184-b3849b0ab238
begin
    # poaceae_mach = machine(poaceae_model, poaceae_Xtrain, poaceae_ytrain)
    # fit!(poaceae_mach, rows=poaceae_trainindices, force=true)
    poaceae_mach = machine(poaceae_model, poaceae_Xtrain, poaceae_ytrain) |> fit!
end

# ╔═╡ 282d4f1c-dae2-46de-bfb6-3d706c481151
begin
    betula_yhat = MLJ.predict(betula_mach, betula_Xtest)
    betula_yref = betula_ytest

    betula_labelsref = seed2label_betula.(betula_yref)
    betula_labelshat = seed2label_betula.(betula_yhat)

    betula_maeres = mean(abs.(betula_yhat .- betula_yref))
    betula_rmseres = sqrt(mean((betula_yhat .- betula_yref) .^ 2))

    betula_accuracyres = sum(betula_labelsref .== betula_labelshat) / length(betula_labelsref)

    betula_accuracyres, betula_maeres, betula_rmseres
end

# ╔═╡ 0bb5eedf-d3a7-4122-8bb5-92ed39bf60ff
begin
    poaceae_yhat = MLJ.predict(poaceae_mach, poaceae_Xtest)
    poaceae_yref = poaceae_ytest

    poaceae_labelsref = seed2label_poaceae.(poaceae_yref)
    poaceae_labelshat = seed2label_poaceae.(poaceae_yhat)

    poaceae_maeres = mean(abs.(poaceae_yhat .- poaceae_yref))
    poaceae_rmseres = sqrt(mean((poaceae_yhat .- poaceae_yref) .^ 2))

    poaceae_accuracyres = sum(poaceae_labelsref .== poaceae_labelshat) / length(poaceae_labelsref)

    poaceae_accuracyres, poaceae_maeres, poaceae_rmseres
end

# ╔═╡ b3578c5d-8a79-4424-b6e0-12c9090e2663
let
    WGLMakie.activate!()
    set_theme!(theme_dark())

    fig = Figure(size=(1315, 385), fontsize=15)
	
    ax1 = Axis(fig[1, 1], yautolimitmargin=(0.1, 0.1), xautolimitmargin=(0.1, 0.1), title="betula")
    n = 1:length(betula_yref)
    lines!(ax1, n, betula_yref; color=:teal, linewidth=1.0)
    lines!(ax1, n, betula_yhat; color=:grey, linewidth=1.0)

	ax2 = Axis(fig[2, 1], yautolimitmargin=(0.1, 0.1), xautolimitmargin=(0.1, 0.1), title="poaceae")
    n = 1:length(poaceae_yref)

    lines!(ax2, n, poaceae_yref; color=:orange, linewidth=1.0)
    lines!(ax2, n, poaceae_yhat; color=:grey, linewidth=1.0)

    fig
end

# ╔═╡ d9840f08-48dc-4f66-979a-0287479caddf
xxx = unpack(betula_traindf, [==(:seeds_m3_24h), ==(:seeds_m3_24h_n1)]...)

# ╔═╡ 05a623dd-5022-4b93-9af2-567b6d91ddbe
test1, train1 = xxx[1:end-1], xxx[end]

# ╔═╡ 3b0c1834-03a5-4085-9a4f-3bbdd2b541f2
hcat(test1...)

# ╔═╡ 74aaddf7-c16e-486f-9e60-1cbc057d3cd6
# ╠═╡ disabled = true
#=╠═╡
begin
    betula_cv = CV(nfolds=3)
    evaluate!(betula_mach, resampling=betula_cv, measure=[rmse, mae], verbosity=0)
end
  ╠═╡ =#

# ╔═╡ ead091b3-92ad-4a9a-8075-003638965526
# ╠═╡ disabled = true
#=╠═╡
begin
    poaceae_cv = CV(nfolds=3)
    evaluate!(poaceae_mach, resampling=poaceae_cv, measure=[rmse, mae], verbosity=0)
end
  ╠═╡ =#

# ╔═╡ 646ad1e3-d26b-4aaf-8b50-344f2f1d22f3
let
	for file in readdir(assetsdir)
		if endswith(file, ".png")
			img = FileIO.load(joinpath(assetsdir, file))
			filename = splitext(file)[1]
			save(joinpath(assetsdir, "$filename.tif"), img)
			println("saved $filename.tif")
		end
	end
end

# ╔═╡ Cell order:
# ╠═d73fd678-da81-413b-a28c-0f37ae1545bc
# ╠═a76f6dfc-7bdd-4110-a88f-271a17c6a7c6
# ╠═efd3112b-b0cc-4b07-a07e-f5f2c50ca743
# ╠═47689b8c-c6e8-472b-8108-736b5bd86e52
# ╠═a5a52cf3-2088-4b23-bbdb-213bde5ba851
# ╠═bcf31b99-c99d-45ef-b163-b23dab2c224c
# ╠═a88b7a62-730b-4499-810b-db32c4d27194
# ╠═db7bc4ec-3d7f-4733-8760-98e569b7f15c
# ╠═809e2431-5589-4182-89d0-9c219932e1d2
# ╠═75c9a372-81c4-4380-b413-6ab254283a1b
# ╠═44e4c211-cb09-4d2e-aa92-5048f3f0357b
# ╟─7f9efe69-448e-4ff9-8556-78b580da2383
# ╟─c849c164-c932-4fcd-89cf-c4a901a6d9a2
# ╠═6430a249-a8fd-4224-b2b4-6711d79384e1
# ╠═af5dd9ad-d16c-4ae4-965b-ffe2010195fc
# ╠═eafaee48-5a9c-4284-ac13-709703e5106f
# ╠═99aa91d4-90c5-4b14-b91f-bb0367811902
# ╠═e1023cdb-c613-48fe-9d12-f1ff371eae03
# ╠═2d4a0d12-8416-4b7f-9cd0-2225c0601a25
# ╠═a06addad-bd13-4b5c-bbdb-f95e282f6a5d
# ╠═fbd8ad41-d7f9-4288-9fb7-58b1327b8e15
# ╠═5ee2617a-7e1b-4a6b-a6bf-c1d2960c28ee
# ╠═6df4c654-3dbe-42e5-89e6-4953a4278852
# ╠═4c2e680e-d156-4767-b985-2425f8c2b390
# ╠═8c28f97e-4a30-4270-9031-faeb2e7a221c
# ╠═8b0c719f-9721-4dac-a242-b31b4c2d7dcd
# ╠═bf10c43b-a135-439d-80cb-e7e69f1aca07
# ╠═f1716b72-0982-4db4-a3df-52177a86997c
# ╠═e120f506-ac7d-4dcb-9029-3148b6620630
# ╠═9a40b000-5636-42a1-976a-4e819a865a2e
# ╠═074c97c2-55e5-42a2-9a6e-fbbaf1e84b36
# ╠═e70d8987-c4a3-44be-8ab5-bda6fbbf9961
# ╠═fb1d5dfb-46b8-4905-9701-d72fafe54de7
# ╠═6d7d9fa5-4e56-4b59-ac1f-b10118504bcc
# ╠═08895a92-a901-4c70-9629-d6f5bd52553e
# ╠═75394666-e3d6-4a74-a83d-6500108f6313
# ╠═3a7c788a-9c01-40fe-a8e3-d1c74122583f
# ╠═b0b39260-a9b3-4aaf-b9b4-6ed00f2c1a58
# ╠═c72fae4c-181b-4c14-b327-2e389f0f93e8
# ╠═b143ab24-7f2c-4f4f-bc8c-d4934382fdce
# ╠═894132e2-7999-4665-8a2e-473beef46ebf
# ╠═9450436a-cc60-4df3-9831-76075967a8ea
# ╠═410a70f5-3166-4818-b8d5-4d36b4b7b1bd
# ╠═db84ef65-0c01-44c4-85fb-230241392434
# ╠═3e7f3a3d-26ad-41ec-a90f-ef463ccf009d
# ╠═3e147298-e362-4eb3-9642-ff2b05927f39
# ╠═38e75e0e-3b5c-4679-905d-ce51af8d7930
# ╠═a26ac5a7-2e69-4572-8184-b3849b0ab238
# ╠═282d4f1c-dae2-46de-bfb6-3d706c481151
# ╠═0bb5eedf-d3a7-4122-8bb5-92ed39bf60ff
# ╠═b3578c5d-8a79-4424-b6e0-12c9090e2663
# ╠═d9840f08-48dc-4f66-979a-0287479caddf
# ╠═05a623dd-5022-4b93-9af2-567b6d91ddbe
# ╠═3b0c1834-03a5-4085-9a4f-3bbdd2b541f2
# ╠═74aaddf7-c16e-486f-9e60-1cbc057d3cd6
# ╠═ead091b3-92ad-4a9a-8075-003638965526
# ╠═646ad1e3-d26b-4aaf-8b50-344f2f1d22f3
