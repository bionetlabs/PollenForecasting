### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 5b13b446-3e34-4637-bc4e-aa14fcc8d813
# ╠═╡ show_logs = false
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
    using MLJGLMInterface
    using MLJMultivariateStatsInterface
    import Flux
    using CUDA
    using cuDNN
	using FileIO
	using ImageMagick
	using Latexify
	using HypothesisTests
end

# ╔═╡ 6d3cf23e-f29e-11ef-2283-1f2f3c102ac9
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

# ╔═╡ 414a2f89-2f01-4a6b-8e76-90ff38d0de39
begin
	# CUDA.versioninfo()
	CUDA.device!(1)
	device = Flux.gpu_device()
end

# ╔═╡ 8f9e15e7-966d-4e43-a489-b1c5485976d9
md"### data preparation"

# ╔═╡ c19e0f7f-3569-4671-875c-9a21ad89e4be
begin
    datadir = joinpath(projectdir, "data")
    aerodir = joinpath(datadir, "aero")
    meteodir = joinpath(datadir, "meteo", "ecad", "balice")
	assetsdir = joinpath(projectdir, "assets")
	outputdir = joinpath(projectdir, "output")
	seedsfile = joinpath(aerodir, "baza_10_24.csv")
	md"dirs"
end

# ╔═╡ 903503fe-41cc-4ee5-9dbb-91552507685e
begin
	seedsdf_raw = CSV.File(seedsfile; delim=";") |> DataFrame
	
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

    betuladf = select(seedsdf, [:date, :betula]) |> dropmissing
    rename!(betuladf, :betula => :seeds_m3_24h)
    betuladf.month = Dates.month.(betuladf.date)
    betuladf.week = Dates.week.(betuladf.date)


    poaceaedf = select(seedsdf, [:date, :poaceae]) |> dropmissing
    rename!(poaceaedf, :poaceae => :seeds_m3_24h)
    poaceaedf.month = Dates.month.(poaceaedf.date)
    poaceaedf.week = Dates.week.(poaceaedf.date)

    # betuladf, poaceaedf
    md"betuladf, poaceaedf"
end

# ╔═╡ 44be3ac2-45b6-4462-9f45-860dc6ca4d63
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


            year2df[species][currentyear] = DataFrame(:date => dates, :seeds_m3_24h => seeds)
        end
    end

    # year2df
	md"year2df"
end

# ╔═╡ 0025c127-306d-4c5f-ad0c-ea01e50685bd
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

# ╔═╡ 08448004-cf4d-46d9-a9f5-4c82ec17215e
begin
    meteodfs = OrderedDict()
    for file in readdir(meteodir)
        target = Symbol(last(split(splitext(file)[1], "balice_")))
        meteodfs[target] = meteo2df(joinpath(meteodir, file))
    end
    # meteodfs
	md"meteodfs"
end

# ╔═╡ 7d678caa-441a-47b6-8d19-2f88062862f1
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
    md"year2df with meteo data"
end

# ╔═╡ 8cbb45cd-4598-4e2f-b183-cb6aa605be81
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

# ╔═╡ 7578bb56-5425-494a-8f45-c520d7ff8dc1
begin
	function first_valid(x::AbstractArray{T})::Int where {T<:Real}
	    if !isnan(x[1])
	        return 1
	    else
	        @inbounds for i in 2:length(x)
	            if !isnan(x[i])
	                return i
	            end
	        end
	    end
	    return 0
	end

	function ema(x::AbstractArray{T}; n::Int64=10, alpha::T=2.0 / (n + 1), wilder::Bool=false) where {T<:Real}
	    @assert n < size(x, 1) && n > 0 "Argument n out of bounds."
	    if wilder
	        alpha = 1.0 / n
	    end
	    out = zeros(size(x))
	    i = first_valid(x)
	    out[1:n+i-2] .= NaN
	    out[n+i-1] = mean(x[i:n+i-1])
	    @inbounds for i = n+i:size(x, 1)
	        out[i] = alpha * (x[i] - out[i-1]) + out[i-1]
	    end
	    return out
	end

	function parametrizemean!(df::DataFrame, columns::Vector{Symbol}; shortwindow::Int=3, longwindow::Int=7, extralongwindow::Int=20)
	    for column in columns
			short = Symbol(column, "_ema", shortwindow)
			nowshort = Symbol(column, "_ema1", shortwindow)
			long = Symbol(column, "_ema", longwindow)
			shortlong = Symbol(column, "_ema", shortwindow, longwindow)
			extralong = Symbol(column, "_ema", extralongwindow)
			longextralong = Symbol(column, "_ema", longwindow, extralongwindow)
			df[!, short] = zeros(nrow(df))
			df[!, long] = zeros(nrow(df))
			df[!, extralong] = zeros(nrow(df))
			df[!, nowshort] = zeros(nrow(df))
			df[!, shortlong] = zeros(nrow(df))
			df[!, longextralong] = zeros(nrow(df))

			for i in shortwindow:nrow(df)
				index3 = max(1, i - (shortwindow - 1)):i
				df[i, short] = ema(df[index3, column]; n=(shortwindow - 1)) |> last
				df[i, nowshort] = if df[i, short] == 0 0.0 else df[i, column] / df[i, short] end
				if i >= longwindow
					index7 = max(1, i - (longwindow - 1)):i
					df[i, long] = ema(df[index7, column]; n=(longwindow - 1)) |> last
					df[i, shortlong] = if df[i, long] == 0 0.0 else df[i, short] / df[i, long] end
				end
				if i >= extralongwindow
					index20 = max(1, i - (extralongwindow - 1)):i
					df[i, extralong] = ema(df[index20, column]; n=(extralongwindow - 1)) |> last
					df[i, longextralong] = if df[i, extralong] == 0 0.0 else df[i, long] / df[i, extralong] end
				end
			end
	    end
	end
end

# ╔═╡ c0564018-8964-425e-a54e-d600065a8994
begin
    for species in keys(year2df)
        for year in keys(year2df[species])
			df = year2df[species][year]

			columns = [:seeds_m3_24h, :cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
			
			parametrizemean!(df, columns; shortwindow=3, longwindow=7, extralongwindow=20)

			shortwindow = 3
			longwindow = 7
			extralongwindow = 20
			allcolumns = copy(columns)
			for column in columns
				short = Symbol(column, "_ema", shortwindow)
				nowshort = Symbol(column, "_ema1", shortwindow)
				long = Symbol(column, "_ema", longwindow)
				shortlong = Symbol(column, "_ema", shortwindow, longwindow)
				extralong = Symbol(column, "_ema", extralongwindow)
				longextralong = Symbol(column, "_ema", longwindow, extralongwindow)
				append!(allcolumns, [short, nowshort, long, shortlong, extralong, longextralong])
			end
			
            parametrizepast!(df, allcolumns; window=100)
        end
    end
    year2df
    md"year2df with parametrizepast"
end

# ╔═╡ d5f362b5-1826-4b73-aefc-11d2033b34a9
function betulatt(features::Vector{Symbol}; ttratio = 0.7, seed=35)
    betula_years = collect(keys(year2df[:betula]))
    shuffle!(Xoshiro(seed), betula_years)
    betula_pivotindex = floor(Int, length(betula_years) * ttratio)
    betula_trainyears = betula_years[1:betula_pivotindex] |> sort
    betula_testyears = betula_years[(betula_pivotindex+1):end] |> sort

    betula_alldf = DataFrame()
	
    betula_traindf = DataFrame()
    for year in betula_trainyears
        betula_traindf = vcat(betula_traindf, year2df[:betula][year])
        betula_alldf = vcat(betula_alldf, year2df[:betula][year])
    end

    betula_testdf = DataFrame()
    for year in betula_testyears
        betula_testdf = vcat(betula_testdf, year2df[:betula][year])
        betula_alldf = vcat(betula_alldf, year2df[:betula][year])
    end
	
    select!(betula_traindf, features)
    select!(betula_testdf, features)

    betula_traindf, betula_testdf, betula_trainyears, betula_testyears
end

# ╔═╡ 5363d19a-89ec-49fc-955a-469582f2ef46
function poaceaett(features::Vector{Symbol}; ttratio = 0.7, seed=35)
    poaceae_years = collect(keys(year2df[:poaceae]))
    shuffle!(Xoshiro(seed), poaceae_years)
    poaceae_pivotindex = floor(Int, length(poaceae_years) * ttratio)
    poaceae_trainyears = poaceae_years[1:poaceae_pivotindex] |> sort
    poaceae_testyears = poaceae_years[(poaceae_pivotindex+1):end] |> sort

    poaceae_alldf = DataFrame()
	
    poaceae_traindf = DataFrame()
    for year in poaceae_trainyears
        poaceae_traindf = vcat(poaceae_traindf, year2df[:poaceae][year])
        poaceae_alldf = vcat(poaceae_alldf, year2df[:poaceae][year])
    end

    poaceae_testdf = DataFrame()
    for year in poaceae_testyears
        poaceae_testdf = vcat(poaceae_testdf, year2df[:poaceae][year])
        poaceae_alldf = vcat(poaceae_alldf, year2df[:poaceae][year])
    end
	
    select!(poaceae_traindf, features)
    select!(poaceae_testdf, features)

    poaceae_traindf, poaceae_testdf, poaceae_trainyears, poaceae_testyears
end

# ╔═╡ 95e4836e-a44d-4a85-90da-6fa6bd4af31b
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

# ╔═╡ eff8cd20-c3e1-430f-bf45-3c380c8ff420
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

# ╔═╡ 52263ba2-8e20-4301-8ad6-f89424855b5a
function makelabels_betula!(df::DataFrame)
    df.label = map(df.seeds_m3_24h) do s
        seed2label_betula(s)
    end
end

# ╔═╡ b9357a17-a938-4d1c-845f-4b7906750ea6
function makelabels_poaceae!(df::DataFrame)
    df.label = map(df.seeds_m3_24h) do s
        seed2label_poaceae(s)
    end
end

# ╔═╡ b73221de-8a07-43fe-ae3d-f52f84c8f8b3
function betula_ttdata(betula_traindf, betula_testdf, betula_trainyears, betula_testyears, notargets=1; save=true)
    betula_trainindices = 1:nrow(betula_traindf) |> vec
    betula_testindices = (nrow(betula_traindf)+1):(nrow(betula_traindf)+nrow(betula_testdf)) |> vec
	
	if save
		betula_traindf_copy = copy(betula_traindf)
    	betula_testdf_copy = copy(betula_testdf)
	    makelabels_betula!(betula_traindf_copy)
	    makelabels_betula!(betula_testdf_copy)
	    betula_traindf_copy = select(betula_traindf_copy, Not(:seeds_m3_24h))
	    betula_testdf_copy = select(betula_testdf_copy, Not(:seeds_m3_24h))
	    CSV.write("../data/tt/betula/betula_train.csv", betula_traindf_copy)
	    CSV.write("../data/tt/betula/betula_test.csv", betula_testdf_copy)
	end

	if notargets > 1
		targets = [==(:seeds_m3_24h)]
		for i in 1:(notargets - 1)
			push!(targets, ==(Symbol(:seeds_m3_24h_n, i)))
		end
	    betula_trainpack = unpack(betula_traindf, targets...)
		betula_ytrain, betula_Xtrain = betula_trainpack[1:end-1], betula_trainpack[end]
		betula_ytrain = hcat(betula_ytrain...)
		betula_testpack = unpack(betula_testdf, targets...)
		betula_ytest, betula_Xtest = betula_testpack[1:end-1], betula_testpack[end]
		betula_ytest = hcat(betula_ytest...)

    	betula_Xtrain, betula_ytrain, betula_Xtest, betula_ytest, betula_trainindices, betula_testindices
	else
		betula_ytrain, betula_Xtrain = unpack(betula_traindf, ==(:seeds_m3_24h))
		betula_ytest, betula_Xtest = unpack(betula_testdf, ==(:seeds_m3_24h))

    	betula_Xtrain, betula_ytrain, betula_Xtest, betula_ytest, betula_trainindices, betula_testindices
	end
end

# ╔═╡ c26f3744-b418-410f-a67b-09edcaa7dedc
function poaceae_ttdata(poaceae_traindf, poaceae_testdf, poaceae_trainyears, poaceae_testyears, notargets=1; save=true)
    poaceae_trainindices = 1:nrow(poaceae_traindf) |> vec
    poaceae_testindices = (nrow(poaceae_traindf)+1):(nrow(poaceae_traindf)+nrow(poaceae_testdf)) |> vec
	
	if save
		poaceae_traindf_copy = copy(poaceae_traindf)
    	poaceae_testdf_copy = copy(poaceae_testdf)
	    makelabels_poaceae!(poaceae_traindf_copy)
	    makelabels_poaceae!(poaceae_testdf_copy)
	    poaceae_traindf_copy = select(poaceae_traindf_copy, Not(:seeds_m3_24h))
	    poaceae_testdf_copy = select(poaceae_testdf_copy, Not(:seeds_m3_24h))
	    CSV.write("../data/tt/poaceae/poaceae_train.csv", poaceae_traindf_copy)
	    CSV.write("../data/tt/poaceae/poaceae_test.csv", poaceae_testdf_copy)
	end

	if notargets > 1
		targets = [==(:seeds_m3_24h)]
		for i in 1:(notargets - 1)
			push!(targets, ==(Symbol(:seeds_m3_24h_n, i)))
		end
	    poaceae_trainpack = unpack(poaceae_traindf, targets...)
		poaceae_ytrain, poaceae_Xtrain = poaceae_trainpack[1:end-1], poaceae_trainpack[end]
		poaceae_ytrain = hcat(poaceae_ytrain...)
		poaceae_testpack = unpack(poaceae_testdf, targets...)
		poaceae_ytest, poaceae_Xtest = poaceae_testpack[1:end-1], poaceae_testpack[end]
		poaceae_ytest = hcat(poaceae_ytest...)

    	poaceae_Xtrain, poaceae_ytrain, poaceae_Xtest, poaceae_ytest, poaceae_trainindices, poaceae_testindices
	else
		poaceae_ytrain, poaceae_Xtrain = unpack(poaceae_traindf, ==(:seeds_m3_24h))
		poaceae_ytest, poaceae_Xtest = unpack(poaceae_testdf, ==(:seeds_m3_24h))

    	poaceae_Xtrain, poaceae_ytrain, poaceae_Xtest, poaceae_ytest, poaceae_trainindices, poaceae_testindices
	end
end

# ╔═╡ d6ee104e-8678-4945-9760-1292e07cc53e
function trainpredict(model, Xtrain, ytrain, Xtest, ytest, seed2label)
	mach = machine(model, Xtrain, ytrain) |> fit!

	yrefs = ytest
	yhats = MLJ.predict(mach, Xtest)
	yhat = yhats .|> last
    yref = yrefs .|> last
	yref_mean = mean(yrefs)

    labelsref = seed2label.(yref)
    labelshat = seed2label.(yhat)

    maeres = mean(abs.(yhat .- yref))
    rmseres = sqrt(mean((yhat .- yref) .^ 2))
	r2 = 1.0 - (sum((yhat .- yref) .^ 2) / sum((yref .- yref_mean) .^ 2))

    accuracyres = sum(labelsref .== labelshat) / length(labelsref)

	labels = unique(labelsref)
	precisions = Dict()
	recalls = Dict()
	labelfreq = Dict()
	for label in labels
		labelfreq[label] = sum(labelsref .== label) / length(labelsref)
		labelsref_ok = labelsref .== label
		labelshat_ok = labelshat .== label
		tp = sum(labelsref_ok .&& labelshat_ok)
		fp = sum(labelshat_ok .&& .!(labelsref_ok))
		fn = sum(.!(labelshat_ok) .&& labelsref_ok)
		precision = tp / (tp + fp)
		precisions[label] = precision
		recall = tp / (tp + fn)
		recalls[label] = recall
	end
	# meanprecision = sum(map(l -> precisions[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	# meanrecall = sum(map(l -> recalls[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	meanprecision = sum(map(l -> precisions[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	meanrecall =  sum(map(l -> recalls[l] * labelfreq[l], labels)) / sum(values(labelfreq))

    mach, accuracyres, meanprecision, meanrecall, maeres, rmseres, r2, yrefs, yhats
end

# ╔═╡ a15a9728-4a28-40a9-a266-953ad343868d
function predict(mach, Xtest, ytest, day::Int, seed2label)
	yrefs = ytest
	yhats = MLJ.predict(mach, Xtest)
	yhat = yhats[:, day]
    yref = yrefs[:, day]
	yref_mean = mean(yrefs)

    labelsref = seed2label.(yref)
    labelshat = seed2label.(yhat)

    maeres = mean(abs.(yhat .- yref))
    rmseres = sqrt(mean((yhat .- yref) .^ 2))
	r2 = 1.0 - (sum((yhat .- yref) .^ 2) / sum((yref .- yref_mean) .^ 2))

    accuracyres = sum(labelsref .== labelshat) / length(labelsref)

	labels = unique(labelsref)
	precisions = Dict()
	recalls = Dict()
	labelfreq = Dict()
	for label in labels
		labelfreq[label] = sum(labelsref .== label) / length(labelsref)
		labelsref_ok = labelsref .== label
		labelshat_ok = labelshat .== label
		tp = sum(labelsref_ok .&& labelshat_ok)
		fp = sum(labelshat_ok .&& .!(labelsref_ok))
		fn = sum(.!(labelshat_ok) .&& labelsref_ok)
		precision = tp / (tp + fp)
		precisions[label] = precision
		recall = tp / (tp + fn)
		recalls[label] = recall
	end
	# meanprecision = sum(map(l -> precisions[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	# meanrecall = sum(map(l -> recalls[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	meanprecision = sum(map(l -> precisions[l] * labelfreq[l], labels)) / sum(values(labelfreq))
	meanrecall =  sum(map(l -> recalls[l] * labelfreq[l], labels)) / sum(values(labelfreq))

    mach, accuracyres, meanprecision, meanrecall, maeres, rmseres, r2, yrefs, yhats
end

# ╔═╡ bea4ee9a-1239-4ce6-9cfc-7711a62daa18
function betula_e1(features, model, notargets=1; save=true)
	Xtrain, ytrain, Xtest, ytest, trainindices, testindices = betula_ttdata(
		betulatt(features; ttratio = 0.7, seed=35)..., notargets; save=save
	)
	
	mach, acc, precision, recall, maeres, rmseres, r2, yrefs, yhats = trainpredict(model, Xtrain, ytrain, Xtest, ytest, seed2label_betula)
end

# ╔═╡ 3849b919-d356-40f2-a220-69ebf122d0c7
function betula_e1predict(features, mach, notargets, day::Int; save=true)
	Xtrain, ytrain, Xtest, ytest, trainindices, testindices = betula_ttdata(
		betulatt(features; ttratio = 0.7, seed=35)..., notargets; save=save
	)
	
	mach, acc, precision, recall, maeres, rmseres, r2, yrefs, yhats = predict(mach, Xtest, ytest, day, seed2label_betula)
end

# ╔═╡ e4f6f34d-f887-41af-bda3-fd968bade141
function poaceae_e1(features, model, notargets=1; save=true)
	Xtrain, ytrain, Xtest, ytest, trainindices, testindices = poaceae_ttdata(
		poaceaett(features; ttratio = 0.7, seed=35)..., notargets; save=save
	)
	
	mach, acc, precision, recall, maeres, rmseres, r2, yrefs, yhats = trainpredict(model, Xtrain, ytrain, Xtest, ytest, seed2label_poaceae)
end

# ╔═╡ 5cfe527a-324b-4348-a6ca-d11ba72e3bdc
function poaceae_e1predict(features, mach, notargets, day::Int; save=true)
	Xtrain, ytrain, Xtest, ytest, trainindices, testindices = poaceae_ttdata(
		poaceaett(features; ttratio = 0.7, seed=35)..., notargets; save=save
	)
	
	mach, acc, precision, recall, maeres, rmseres, r2, yrefs, yhats = predict(mach, Xtest, ytest, day, seed2label_poaceae)
end

# ╔═╡ 69b8981b-b733-47f5-bd00-f545e39c29ab
function makefeatures(;
	futureday::Int,
	seedsinterval::Int,
	seedsn::Int,
	meteointerval::Int,
	meteon::Int,
	futureseeds=false,
	futurefeatures=false,
	seedsema=true,
	featuresema=true,
	featuresema_future=true,
	shortwindow=3,
	longwindow=7,
	extralongwindow=20,
	seedsema_extralong=true,
	featuresema_extralong=false,
	featuresema_future_extralong=false,
	meteofeatures = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx],
	allfeatures_reverse=false
)
	seeds = :seeds_m3_24h
	
	ret = [seeds]
	
	if seedsema
		seeds_short = Symbol(seeds, "_ema", shortwindow, "_n", futureday)
		seeds_nowshort = Symbol(seeds, "_ema1", shortwindow, "_n", futureday)
		seeds_long = Symbol(seeds, "_ema", longwindow, "_n", futureday)
		seeds_shortlong = Symbol(seeds, "_ema", shortwindow, longwindow, "_n", futureday)
		append!(ret, [seeds_short, seeds_nowshort, seeds_long, seeds_shortlong])
		if seedsema_extralong
			seeds_extralong = Symbol(seeds, "_ema", extralongwindow, "_n", futureday)
			seeds_longextralong = Symbol(seeds, "_ema", longwindow, extralongwindow, "_n", futureday)
			append!(ret, [seeds_extralong, seeds_longextralong])
		end
	end
	
	if futureseeds
		for i in 1:(futureday - 1)
			push!(ret, Symbol(seeds, "_n", i))
		end
	end
	for i in futureday:seedsinterval:(futureday + (seedsn - 1))
		push!(ret, Symbol(seeds, "_n", i))
	end
	for feature in meteofeatures
		if featuresema
			short = Symbol(feature, "_ema", shortwindow, "_n", futureday)
			nowshort = Symbol(feature, "_ema1", shortwindow, "_n", futureday)
			long = Symbol(feature, "_ema", longwindow, "_n", futureday)
			shortlong = Symbol(feature, "_ema", shortwindow, longwindow, "_n", futureday)
			append!(ret, [short, nowshort, long, shortlong])
			if featuresema_extralong
				extralong = Symbol(feature, "_ema", extralongwindow, "_n", futureday)
				longextralong = Symbol(feature, "_ema", longwindow, extralongwindow, "_n", futureday)
				append!(ret, [extralong, longextralong])
			end
		end
		if featuresema_future
			# push!(ret, feature)
			short = Symbol(feature, "_ema", shortwindow)
			nowshort = Symbol(feature, "_ema1", shortwindow)
			long = Symbol(feature, "_ema", longwindow)
			shortlong = Symbol(feature, "_ema", shortwindow, longwindow)
			append!(ret, [short, nowshort, long, shortlong])
			if featuresema_future_extralong
				extralong = Symbol(feature, "_ema", extralongwindow)
				longextralong = Symbol(feature, "_ema", longwindow, extralongwindow)
				append!(ret, [extralong, longextralong])
			end
		end
		
		if futurefeatures
			push!(ret, feature)
			for i in 1:meteointerval:(futureday - 1)
				push!(ret, Symbol(feature, "_n", i))
			end
		end
		for i in futureday:meteointerval:(futureday + (meteon - 1))
			push!(ret, Symbol(feature, "_n", i))
		end
	end
	allfeatures_reverse ? reverse(ret) : ret
end

# ╔═╡ 48996027-79ab-4b89-9d27-0de42a918d54
tabularfeatures1 = makefeatures(
	futureday=1,
	seedsinterval=1,
	seedsn=14,
	meteointerval=1,
	meteon=1,
	futureseeds=false,
	futurefeatures=false,
	seedsema=true,
	featuresema=false,
	featuresema_future=true,
	extralongwindow=20,
	seedsema_extralong=true,
	featuresema_extralong=false,
	featuresema_future_extralong=true,
	meteofeatures = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
)

# ╔═╡ 8769438a-9daa-4de6-ba89-d1c126dabc50
tabularfeatures4 = makefeatures(
	futureday=4,
	seedsinterval=1,
	seedsn=14,
	meteointerval=1,
	meteon=1,
	futureseeds=false,
	futurefeatures=false,
	seedsema=true,
	featuresema=true,
	featuresema_future=true,
	extralongwindow=20,
	seedsema_extralong=true,
	featuresema_extralong=false,
	featuresema_future_extralong=true,
	meteofeatures = [:cc, :hu, :ss, :tg]
)

# ╔═╡ 85268b9b-9a45-400e-a226-1e46e0d50302
tabularfeatures7 = makefeatures(
	futureday=7,
	seedsinterval=1,
	seedsn=14,
	meteointerval=1,
	meteon=0,
	futureseeds=false,
	futurefeatures=false,
	seedsema=true,
	featuresema=true,
	featuresema_future=true,
	extralongwindow=20,
	seedsema_extralong=true,
	featuresema_extralong=false,
	featuresema_future_extralong=true,
	meteofeatures = [:tg]
)

# ╔═╡ 3cd2b392-a09a-4493-92c1-baf9e0bd8ad3
tabularfeatures = makefeatures(
	futureday=1,
	seedsinterval=1,
	seedsn=14,
	meteointerval=1,
	meteon=1,
	futureseeds=false,
	futurefeatures=false,
	seedsema=true,
	featuresema=false,
	featuresema_future=true,
	extralongwindow=20,
	seedsema_extralong=true,
	featuresema_extralong=false,
	featuresema_future_extralong=true,
	meteofeatures = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
	# meteofeatures = [:cc, :fg, :hu, :pp, :qq, :sd, :ss, :tg, :tn, :tx]
	# cloud cover - ok'
	# average wind speed - ok' chociaż nie jest to ważny parametr z punktu widzenia wydzielania pyłku
	# relative humidity - ok'
	# mean sea level pressure - raczej nie ma uzasadnieniea, ale w niektórych publikacjach się pojawia
	# global radiation - nie ma uzasadnienia
	# snow depth - tak, ale tylko dla drzew wczesnowiosennych, leszczyny, olszy
	# duration of sunshine (wcześniej używałam relative sunshine, ale może też być liczba godzin słonecznych/dobę, a nie tylko w stosunku do długości dnia - relative)
	# temperature (mean, minimum, and maximum) - ważne
)

# ╔═╡ 5282c362-871d-4592-9377-91949802be28
md"### experiment 1"

# ╔═╡ b4242277-6771-4912-95f4-1ddd07b1789a
# ╠═╡ show_logs = false
let
	# WGLMakie.activate!()
    CairoMakie.activate!()
    # set_theme!(theme_dark())
    set_theme!(theme_light())
	
	betula_model = @load(XGBoostRegressor, pkg = "XGBoost", verbosity=0)(max_depth=25)
	_, _, _, _, _, _, _, betula_yref, betula_yhat = betula_e1(tabularfeatures, betula_model; save=true)

	poaceae_model = @load(XGBoostRegressor, pkg = "XGBoost", verbosity=0)(max_depth=25)
	_, _, _, _, _, _, _, poaceae_yref, poaceae_yhat = poaceae_e1(tabularfeatures, poaceae_model; save=true)

    fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 3.5 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)
	
    ax1 = Axis(
		fig[1, 1],
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="A",
		titlefont="Times New Roman", titlecolor=:gray25, titlesize=fontsize,
		xticklabelsvisible = false,
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)
    n = 1:(1080 - 575 + 1) # 1:length(betula_yref)
    lines!(ax1, n, betula_yref[575:1080]; color=:gray50, linewidth=1.0)
    lines!(ax1, n, betula_yhat[575:1080]; color=:teal, linewidth=1.0)

	ax2 = Axis(
		fig[2, 1], 
		yautolimitmargin=(0.05, 0.05),
		xautolimitmargin=(0.05, 0.05),
		titlealign=:left,
		title="B",
		titlefont="Times New Roman", titlecolor=:gray25, titlesize=fontsize,
		xticklabelsvisible = false,
		ylabel = L"\frac{\textrm{pollen*m}^\textrm{-3}}{\textrm{24h}}",
		ylabelfont = :bold,
		ylabelsize = fontsize * 1.2
	)
    n = 1:(1275 - 800 + 1) # 1:length(poaceae_yref)

    lines!(ax2, n, poaceae_yref[800:1275]; color=:gray50, linewidth=1.0)
    lines!(ax2, n, poaceae_yhat[800:1275]; color=:orange, linewidth=1.0)

	save(joinpath(assetsdir, "Fig3.png"), fig, px_per_unit=1)
	img = FileIO.load(joinpath(assetsdir, "Fig3.png"))
	save(joinpath(assetsdir, "Fig3.tif"), img)

    fig
	# md"prediction plot"
end

# ╔═╡ f3f3a7bd-c6b7-4d31-b382-e1f51f15f449
function e1(model, features, notargets=1)
	betulatime = @elapsed begin
		betulamem = @allocated begin
			betulamach, betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2 = betula_e1(features, model, notargets; save=true)
			betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2 = round.(
				(betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2); digits=3
			)
		end
		betulamem = round(betulamem / 10 ^ 6; digits=0) |> Int
	end

	poaceaetime = @elapsed begin
		poaceaemem = @allocated begin
			poaceaemach, poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2 = poaceae_e1(features, model, notargets; save=true)
			poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2 = round.(
				(poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2); digits=3
			)
		end
		poaceaemem = round(poaceaemem / 10 ^ 6; digits=0) |> Int
	end
	
	e1_resultdf = DataFrame(
		:taxon => [:betula, :poaceae, :mean],
		:accuracy => [betula_acc, poaceae_acc, round(mean([betula_acc, poaceae_acc]); digits=3)],
		:precision => [betula_precision, poaceae_precision, round(mean([betula_precision, poaceae_precision]); digits=3)],
		:recall => [betula_recall, poaceae_recall, round(mean([betula_recall, poaceae_recall]); digits=3)],
		:mae => [betula_mae, poaceae_mae, round(mean([betula_mae, poaceae_mae]); digits=3)],
		:rmse => [betula_rmse, poaceae_rmse, round(mean([betula_rmse, poaceae_rmse]); digits=3)],
		:r2 => [betula_r2, poaceae_r2, round(mean([betula_r2, poaceae_r2]); digits=3)],
		:time => [betulatime, poaceaetime, round(mean([betulatime, poaceaetime,]); digits=3)],
		:memory => [betulamem, poaceaemem, round(mean([betulamem, poaceaemem,]); digits=3)],
	)

	e1_resultdf, betulamach, poaceaemach
end

# ╔═╡ e37eebbf-8d9c-4bae-bc6b-3b9a5d101205
function e1predict(betulamach, poaceaemach, features, notargets, day::Int)
	betulatime = @elapsed begin
		betulamem = @allocated begin
			betulamach, betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2 = betula_e1predict(
				features, betulamach, notargets, day; save=false
			)
			betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2 = round.(
				(betula_acc, betula_precision, betula_recall, betula_mae, betula_rmse, betula_r2); digits=3
			)
		end
		betulamem = round(betulamem / 10 ^ 6; digits=0) |> Int
	end

	poaceaetime = @elapsed begin
		poaceaemem = @allocated begin
			poaceaemach, poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2 = poaceae_e1predict(
				features, poaceaemach, notargets, day; save=false
			)
			poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2 = round.(
				(poaceae_acc, poaceae_precision, poaceae_recall, poaceae_mae, poaceae_rmse, poaceae_r2); digits=3
			)
		end
		poaceaemem = round(poaceaemem / 10 ^ 6; digits=0) |> Int
	end
	
	e1_resultdf = DataFrame(
		:taxon => [:betula, :poaceae, :mean],
		:accuracy => [betula_acc, poaceae_acc, round(mean([betula_acc, poaceae_acc]); digits=3)],
		:precision => [betula_precision, poaceae_precision, round(mean([betula_precision, poaceae_precision]); digits=3)],
		:recall => [betula_recall, poaceae_recall, round(mean([betula_recall, poaceae_recall]); digits=3)],
		:mae => [betula_mae, poaceae_mae, round(mean([betula_mae, poaceae_mae]); digits=3)],
		:rmse => [betula_rmse, poaceae_rmse, round(mean([betula_rmse, poaceae_rmse]); digits=3)],
		:r2 => [betula_r2, poaceae_r2, round(mean([betula_r2, poaceae_r2]); digits=3)],
		:time => [betulatime, poaceaetime, round(mean([betulatime, poaceaetime,]); digits=3)],
		:memory => [betulamem, poaceaemem, round(mean([betulamem, poaceaemem,]); digits=3)],
	)

	e1_resultdf
end

# ╔═╡ 3e740355-90a6-4de4-ab68-58ef2df4dbe8
# ╠═╡ show_logs = false
# begin
# 	knnmodel = @load(KNNRegressor, pkg = "NearestNeighborModels", verbosity=0)(K=5)
# 	e1knn, knnmach_betula, knnmach_poaceae = e1(knnmodel, tabularfeatures)
# 	nameof(typeof(knnmodel)), e1knn
# end

# ╔═╡ 1100195a-0219-40a2-bfed-dbb8f32d5cf9
# ╠═╡ show_logs = false
# begin
# 	lrmodel = @load(LinearRegressor, pkg = "MLJLinearModels", verbosity=0)()
# 	e1lr, lrmach_betula, lrmach_poaceae = e1(lrmodel, tabularfeatures)
# 	nameof(typeof(lrmodel)), e1lr
# end

# ╔═╡ cfc95d44-ddf3-49a7-899b-a8976f87da28
# ╠═╡ show_logs = false
# begin
# 	rfmodel = @load(RandomForestRegressor, pkg = "DecisionTree", verbosity=0)(max_depth=25)
# 	e1rf, rfmach_betula, rfmach_poaceae = e1(rfmodel, tabularfeatures)
# 	nameof(typeof(rfmodel)), e1rf
# end

# ╔═╡ 5e6fc543-5d65-4d0d-a8bb-d58e39f41924
# ╠═╡ show_logs = false
# begin
# 	xgmodel = @load(XGBoostRegressor, pkg = "XGBoost", verbosity=0)(max_depth=25)
# 	e1xg, xgmach_betula, xgmach_poaceae = e1(xgmodel, tabularfeatures)
# 	nameof(typeof(xgmodel)), e1xg
# end

# ╔═╡ 8b22b31e-2566-49fa-abd4-b74c7167988c
# ╠═╡ show_logs = false
# begin
# 	dtmodel = @load(DecisionTreeRegressor, pkg = "DecisionTree", verbosity=0)()
# 	e1dt, dtmach_betula, dtmach_poaceae = e1(dtmodel, tabularfeatures)
# 	nameof(typeof(dtmodel)), e1dt
# end

# ╔═╡ 8dc3fef7-79c2-4e58-988a-aa6a9e0ce083
# begin
# 	betulaacc = maximum([e1dt.accuracy[1], e1xg.accuracy[1], e1rf.accuracy[1], e1lr.accuracy[1], e1knn.accuracy[1]])
# 	peaceaeacc = maximum([e1dt.accuracy[2], e1xg.accuracy[2], e1rf.accuracy[2], e1lr.accuracy[2], e1knn.accuracy[2]])
# 	meanacc = maximum([e1dt.accuracy[3], e1xg.accuracy[3], e1rf.accuracy[3], e1lr.accuracy[3], e1knn.accuracy[3]])
# 	md"[top accuracy for tabular features] betula: $betulaacc %, poaceae: $peaceaeacc %, mean: $meanacc %"
# end

# ╔═╡ 5484df43-9759-4e65-818a-7ce7025bca83
# begin
# 	notargets = 7
# 	sequentialfeatures = makefeatures(
# 		futureday=notargets,
# 		seedsinterval=1,
# 		seedsn=20,
# 		meteointerval=1,
# 		meteon=0,
# 		futureseeds=true,
# 		futurefeatures=false,
# 		seedsema=false,
# 		featuresema=false,
# 		featuresema_future=false,
# 		extralongwindow=20,
# 		seedsema_extralong=false,
# 		featuresema_extralong=false,
# 		featuresema_future_extralong=false,
# 		meteofeatures = []
# 	)
# 	nfeatures = length(sequentialfeatures) - notargets
#     # builder = MLJFlux.@builder begin
#     #     Flux.Chain(
#     #         Flux.LSTM(nfeatures => 512),
#     #         Flux.LSTM(512 => 256),
#     #         x -> x[:, end, :],
#     #         Flux.Dense(256 => notargets)
#     #     )
#     # end
#     builder = MLJFlux.@builder begin
#         Flux.Chain(
#             x -> reshape(x, (nfeatures, 1, :)),
# 			Flux.Conv((3,), 1 => 64, Flux.relu; pad=2, stride=2),
# 		    Flux.MaxPool((2,)),
# 			Flux.Conv((5,), 64 => 32, Flux.relu; pad=0, stride=1),
# 		    Flux.flatten,
# 		    Flux.LSTM(32 => 128),
# 		    Flux.LSTM(128 => 64),
# 		    # LSTM(64 => 128),
# 		    # LSTM(seq_length÷2 * 16 => 50),
			
# 		    # LSTM(seq_length => 50),
			
# 			x -> x[:, end, :],
			
# 		    Flux.Dense(64 => notargets)
#         )
#     end
# 	lstmmodel = @load(MultitargetNeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
# 		builder=builder, epochs=30, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.001)
# 	)
# 	e1lstm, lstmmach_betula, lstmmach_poaceae = e1(lstmmodel, sequentialfeatures, notargets)
# 	nameof(typeof(lstmmodel)), e1lstm
# end

# ╔═╡ 5ff3a379-e54a-4de5-a31c-f915b88b1161
# let
# 	notargets = 7
# 	sequentialfeatures = makefeatures(
# 		futureday=notargets,
# 		seedsinterval=1,
# 		seedsn=20,
# 		meteointerval=1,
# 		meteon=0,
# 		futureseeds=true,
# 		futurefeatures=false,
# 		seedsema=false,
# 		featuresema=true,
# 		featuresema_future=true,
# 		extralongwindow=20,
# 		seedsema_extralong=false,
# 		featuresema_extralong=false,
# 		featuresema_future_extralong=true,
# 		meteofeatures = [:tg]
# 	)
# 	targetfeatures = [:seeds_m3_24h]
# 	for i in 1:(notargets - 1)
# 		push!(targetfeatures, Symbol(:seeds_m3_24h, "_n", i))
# 	end
# 	seedsfeatures = filter(sequentialfeatures) do x
# 		startswith(string(x), "seeds_m3_24h") && !(x in targetfeatures)
# 	end |> collect
# 	no_seedsfeatures = seedsfeatures |> length
# 	meteofeatures = filter(x -> !startswith(string(x), "seeds_m3_24h"), sequentialfeatures) |> collect
# 	no_meteofeatures = meteofeatures |> length
# 	println("seedsfeatures[$(no_seedsfeatures)]: ", seedsfeatures)
# 	println("meteofeatures[$(no_meteofeatures)]: ", meteofeatures)

# 	nfeatures_sequential = length(sequentialfeatures) - no_meteofeatures
# 	densebranch = Flux.Chain(
# 	    Flux.Dense(no_meteofeatures => 64, Flux.relu),
# 	    Flux.Dense(64 => 32, Flux.relu),
# 	)
# 	lstmbranch = Flux.Chain(
# 		x -> reshape(x, (no_seedsfeatures, 1, :)),
# 			Flux.Conv((3,), 1 => 64, Flux.relu; pad=2, stride=2),
# 		    Flux.MaxPool((2,)),
# 			Flux.Conv((5,), 64 => 32, Flux.relu; pad=0, stride=1),
# 		    Flux.flatten,
# 		    Flux.LSTM(32 => 256),
# 		    Flux.LSTM(256 => 128),
# 			x -> x[:, end, :],
# 	)
# 	builder = MLJFlux.@builder begin
# 		Flux.Chain(
# 			(x_feat) -> Flux.cat(
# 				lstmbranch(x_feat[1:no_seedsfeatures, :]), 
# 				densebranch(x_feat[(no_seedsfeatures + 1):end, :]), 
# 				dims=1
# 			),
# 			Flux.Dense(160 => 96),
# 			Flux.Dense(96 => notargets)
# 		)
# 	end

# 	lstmmodel = @load(MultitargetNeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
# 		builder=builder, epochs=30, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.001)
# 	)
# 	e1lstm, lstmmach_betula, lstmmach_poaceae = e1(lstmmodel, sequentialfeatures, notargets)
# 	nameof(typeof(lstmmodel)), e1lstm
# end

# ╔═╡ a248509d-f0de-4b1c-83d1-e420b4eea746
# begin
# 	local notargets = 7
# 	local meteofeatures = [:tg, :hu]
# 	local no_meteofeatures = meteofeatures |> length
# 	local sequentialfeatures = makefeatures(
# 		futureday=notargets,
# 		seedsinterval=1,
# 		seedsn=20,
# 		meteointerval=1,
# 		meteon=(13 + (7 - notargets)),
# 		futureseeds=true,
# 		futurefeatures=true,
# 		seedsema=false,
# 		featuresema=false,
# 		featuresema_future=false,
# 		extralongwindow=20,
# 		seedsema_extralong=false,
# 		featuresema_extralong=false,
# 		featuresema_future_extralong=false,
# 		meteofeatures = meteofeatures,
# 		allfeatures_reverse=true
# 	)
# 	local nofeatures = no_meteofeatures + 1
# 	local sequencelen = (length(sequentialfeatures) - notargets) ÷ nofeatures
#     local builder = MLJFlux.@builder begin
#         Flux.Chain(
#             x -> reshape(x, (sequencelen, nofeatures, :)),
# 			Flux.Conv((3,), nofeatures => 64, Flux.relu; pad=2, stride=2),
# 		    Flux.MaxPool((2,)),
# 			Flux.Conv((5,), 64 => nofeatures * 32, Flux.relu; pad=0, stride=1),
# 		    Flux.flatten,
# 		    Flux.LSTM(nofeatures * 32 => nofeatures * 128),
# 		    Flux.LSTM(nofeatures * 128 => nofeatures * 64),
# 			x -> x[:, end, :],
# 		    Flux.Dense(nofeatures * 64 => notargets)
#         )
#     end
# 	lstmmodel = @load(MultitargetNeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
# 		builder=builder, epochs=100, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.00005)
# 	)
# 	_e1lstm, lstmmach_betula, lstmmach_poaceae = e1(lstmmodel, sequentialfeatures, notargets)
# 	e1lstm_d1 = e1predict(lstmmach_betula, lstmmach_poaceae, sequentialfeatures, notargets, 1)
# 	e1lstm_d4 = e1predict(lstmmach_betula, lstmmach_poaceae, sequentialfeatures, notargets, 4)
# 	e1lstm_d7 = e1predict(lstmmach_betula, lstmmach_poaceae, sequentialfeatures, notargets, 7)
# 	nameof(typeof(lstmmodel)), e1lstm_d1 , e1lstm_d4 , e1lstm_d7
# end

# ╔═╡ 497b0c74-e7b2-4b07-b191-b1b413fbbcb2
# begin
# 	local notargets = 4
# 	local meteofeatures = [:tg, :hu]
# 	local no_meteofeatures = meteofeatures |> length
# 	local sequentialfeatures = makefeatures(
# 		futureday=notargets,
# 		seedsinterval=1,
# 		seedsn=20,
# 		meteointerval=1,
# 		meteon=(13 + (7 - notargets)),
# 		futureseeds=true,
# 		futurefeatures=true,
# 		seedsema=false,
# 		featuresema=false,
# 		featuresema_future=false,
# 		extralongwindow=20,
# 		seedsema_extralong=false,
# 		featuresema_extralong=false,
# 		featuresema_future_extralong=false,
# 		meteofeatures = meteofeatures,
# 		allfeatures_reverse=true
# 	)
# 	local nofeatures = no_meteofeatures + 1
# 	local sequencelen = (length(sequentialfeatures) - notargets) ÷ nofeatures
#     local builder = MLJFlux.@builder begin
#         Flux.Chain(
#             x -> reshape(x, (sequencelen, nofeatures, :)),
# 			Flux.Conv((3,), nofeatures => 64, Flux.relu; pad=2, stride=2),
# 		    Flux.MaxPool((2,)),
# 			Flux.Conv((5,), 64 => nofeatures * 32, Flux.relu; pad=0, stride=1),
# 		    Flux.flatten,
# 		    Flux.GRUv3(nofeatures * 32 => nofeatures * 128),
# 		    Flux.GRUv3(nofeatures * 128 => nofeatures * 64),
# 			x -> x[:, end, :],
# 		    Flux.Dense(nofeatures * 64 => notargets)
#         )
#     end
# 	grumodel = @load(MultitargetNeuralNetworkRegressor, pkg = "MLJFlux", verbosity = 0)(
# 		builder=builder, epochs=100, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.00005)
# 	)
# 	_e1gru, grumach_betula, grumach_poaceae = e1(grumodel, sequentialfeatures, notargets)
# 	e1gru_d1 = e1predict(grumach_betula, grumach_poaceae, sequentialfeatures, notargets, 1)
# 	e1gru_d4 = e1predict(grumach_betula, grumach_poaceae, sequentialfeatures, notargets, 4)
# 	# e1gru_d7 = e1predict(grumach_betula, grumach_poaceae, sequentialfeatures, notargets, 7)
# 	nameof(typeof(grumodel)), e1gru_d1, e1gru_d4 #, e1gru_d7
# end

# ╔═╡ 2b36e835-4f81-45be-9ee2-7ab1a8c6ada9
# ╠═╡ disabled = true
#=╠═╡
begin
	betularesults = DataFrame(
		:model => ["K-Nearest Neighbors", "Linear Regression", "Decision Trees", "Random Forest", "XGBoost"],
		:accuracy => [e1knn[1, :accuracy], e1lr[1, :accuracy], e1dt[1, :accuracy], e1rf[1, :accuracy], e1xg[1, :accuracy]],
		:precision => [e1knn[1, :precision], e1lr[1, :precision], e1dt[1, :precision], e1rf[1, :precision], e1xg[1, :precision]],
		:recall => [e1knn[1, :recall], e1lr[1, :recall], e1dt[1, :recall], e1rf[1, :recall], e1xg[1, :recall]],
		:mae => [e1knn[1, :mae], e1lr[1, :mae], e1dt[1, :mae], e1rf[1, :mae], e1xg[1, :mae]],
		:rmse => [e1knn[1, :rmse], e1lr[1, :rmse], e1dt[1, :rmse], e1rf[1, :rmse], e1xg[1, :rmse]],
		:time => [e1knn[1, :time], e1lr[1, :time], e1dt[1, :time], e1rf[1, :time], e1xg[1, :time]],
		:memory => [e1knn[1, :memory], e1lr[1, :memory], e1dt[1, :memory], e1rf[1, :memory], e1xg[1, :memory]],
	)
	CSV.write(joinpath(outputdir, "betularesults.csv"), betularesults)

	poaceaeresults = DataFrame(
		:model => ["K-Nearest Neighbors", "Linear Regression", "Decision Trees", "Random Forest", "XGBoost"],
		:accuracy => [e1knn[2, :accuracy], e1lr[2, :accuracy], e1dt[2, :accuracy], e1rf[2, :accuracy], e1xg[2, :accuracy]],
		:precision => [e1knn[2, :precision], e1lr[2, :precision], e1dt[2, :precision], e1rf[2, :precision], e1xg[2, :precision]],
		:recall => [e1knn[2, :recall], e1lr[2, :recall], e1dt[2, :recall], e1rf[2, :recall], e1xg[2, :recall]],
		:mae => [e1knn[2, :mae], e1lr[2, :mae], e1dt[2, :mae], e1rf[2, :mae], e1xg[2, :mae]],
		:rmse => [e1knn[2, :rmse], e1lr[2, :rmse], e1dt[2, :rmse], e1rf[2, :rmse], e1xg[2, :rmse]],
		:time => [e1knn[2, :time], e1lr[2, :time], e1dt[2, :time], e1rf[2, :time], e1xg[2, :time]],
		:memory => [e1knn[2, :memory], e1lr[2, :memory], e1dt[2, :memory], e1rf[2, :memory], e1xg[2, :memory]],
	)
	CSV.write(joinpath(outputdir, "poaceaeresults.csv"), poaceaeresults)

	md"save results"
end
  ╠═╡ =#

# ╔═╡ dac107cb-687d-4f3a-acdd-167e4fdfc446
# ╠═╡ disabled = true
#=╠═╡
begin
	betulafe = feature_importances(xgmach_betula)
	poaceaefe = feature_importances(rfmach_poaceae)
	betulafe_df = DataFrame(:feature => first.(betulafe), :score => last.(betulafe))
	poaceaefe_df = DataFrame(:feature => first.(poaceaefe), :score => last.(poaceaefe))
	CSV.write(joinpath(outputdir, "betulafe_xgboost.csv"), betulafe_df)
	CSV.write(joinpath(outputdir, "poaceaefe_xgboost.csv"), poaceaefe_df)
	betulafe_df, poaceaefe_df
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─6d3cf23e-f29e-11ef-2283-1f2f3c102ac9
# ╟─5b13b446-3e34-4637-bc4e-aa14fcc8d813
# ╠═414a2f89-2f01-4a6b-8e76-90ff38d0de39
# ╟─8f9e15e7-966d-4e43-a489-b1c5485976d9
# ╟─c19e0f7f-3569-4671-875c-9a21ad89e4be
# ╟─903503fe-41cc-4ee5-9dbb-91552507685e
# ╟─44be3ac2-45b6-4462-9f45-860dc6ca4d63
# ╟─0025c127-306d-4c5f-ad0c-ea01e50685bd
# ╟─08448004-cf4d-46d9-a9f5-4c82ec17215e
# ╟─7d678caa-441a-47b6-8d19-2f88062862f1
# ╟─8cbb45cd-4598-4e2f-b183-cb6aa605be81
# ╟─7578bb56-5425-494a-8f45-c520d7ff8dc1
# ╟─c0564018-8964-425e-a54e-d600065a8994
# ╟─d5f362b5-1826-4b73-aefc-11d2033b34a9
# ╟─5363d19a-89ec-49fc-955a-469582f2ef46
# ╟─95e4836e-a44d-4a85-90da-6fa6bd4af31b
# ╟─eff8cd20-c3e1-430f-bf45-3c380c8ff420
# ╟─52263ba2-8e20-4301-8ad6-f89424855b5a
# ╟─b9357a17-a938-4d1c-845f-4b7906750ea6
# ╟─b73221de-8a07-43fe-ae3d-f52f84c8f8b3
# ╟─c26f3744-b418-410f-a67b-09edcaa7dedc
# ╟─d6ee104e-8678-4945-9760-1292e07cc53e
# ╟─a15a9728-4a28-40a9-a266-953ad343868d
# ╟─bea4ee9a-1239-4ce6-9cfc-7711a62daa18
# ╟─3849b919-d356-40f2-a220-69ebf122d0c7
# ╟─e4f6f34d-f887-41af-bda3-fd968bade141
# ╟─5cfe527a-324b-4348-a6ca-d11ba72e3bdc
# ╟─69b8981b-b733-47f5-bd00-f545e39c29ab
# ╟─48996027-79ab-4b89-9d27-0de42a918d54
# ╟─8769438a-9daa-4de6-ba89-d1c126dabc50
# ╟─85268b9b-9a45-400e-a226-1e46e0d50302
# ╠═3cd2b392-a09a-4493-92c1-baf9e0bd8ad3
# ╟─5282c362-871d-4592-9377-91949802be28
# ╠═b4242277-6771-4912-95f4-1ddd07b1789a
# ╟─f3f3a7bd-c6b7-4d31-b382-e1f51f15f449
# ╟─e37eebbf-8d9c-4bae-bc6b-3b9a5d101205
# ╠═3e740355-90a6-4de4-ab68-58ef2df4dbe8
# ╠═1100195a-0219-40a2-bfed-dbb8f32d5cf9
# ╠═cfc95d44-ddf3-49a7-899b-a8976f87da28
# ╠═5e6fc543-5d65-4d0d-a8bb-d58e39f41924
# ╠═8b22b31e-2566-49fa-abd4-b74c7167988c
# ╠═8dc3fef7-79c2-4e58-988a-aa6a9e0ce083
# ╟─5484df43-9759-4e65-818a-7ce7025bca83
# ╟─5ff3a379-e54a-4de5-a31c-f915b88b1161
# ╠═a248509d-f0de-4b1c-83d1-e420b4eea746
# ╠═497b0c74-e7b2-4b07-b191-b1b413fbbcb2
# ╟─2b36e835-4f81-45be-9ee2-7ab1a8c6ada9
# ╟─dac107cb-687d-4f3a-acdd-167e4fdfc446
