### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ c596973a-d074-4ad4-b6b5-4f7e3449c99b
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
    using FileIO
    using ImageMagick
    using Latexify
    using OrderedCollections
end

# ╔═╡ 049662d0-f6db-11ef-13ba-a35a789e5b2a
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

# ╔═╡ 9f797ff7-3124-40c3-8ae2-886c7407a40b
begin
    outputdir = joinpath(projectdir, "output")
    assetsdir = joinpath(projectdir, "assets")
    md"paths"
end

# ╔═╡ 2fe4f0f3-9add-4780-9e6c-c5ef53879112
begin
    resultsfeatures = [:model, :accuracy, :mae, :rmse, :time, :memory]

    betularesults1 = CSV.File(joinpath(outputdir, "betularesults1.csv")) |> DataFrame
    betularesults1_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.84, 0.719, 0.916],
        :precision => [0.877, 0.864, 0.723],
        :recall => [0.84, 0.719, 0.783],
        :mae => [33.279, 37.767, 26.937],
        :rmse => [158.074, 159.795, 130.66],
        :time => [1602.9, 3148.41, 7.332],
        :memory => [214337.0, 228738.0, 837.536],
    )
    betularesults1 = vcat(betularesults1, betularesults1_manual)
    sort!(betularesults1, :accuracy, rev=true)

    poaceaeresults1 = CSV.File(joinpath(outputdir, "poaceaeresults1.csv")) |> DataFrame
    poaceaeresults1_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.787, 0.725, 0.857],
        :precision => [0.755, 0.78, 0.689],
        :recall => [0.787, 0.725, 0.752],
        :mae => [12.325, 12.979, 8.852],
        :rmse => [31.441, 29.396, 25.320],
        :time => [4509.94, 5493.77, 14.549],
        :memory => [390109.0, 422627.0, 737.967],
    )
    poaceaeresults1 = vcat(poaceaeresults1, poaceaeresults1_manual)
    sort!(poaceaeresults1, :accuracy, rev=true)

    latexify(select(betularesults1, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    latexify(select(poaceaeresults1, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    betularesults1, poaceaeresults1
end

# ╔═╡ 179fa328-f95b-4026-9669-b28c4e732951
begin
    betularesults4 = CSV.File(joinpath(outputdir, "betularesults4.csv")) |> DataFrame
    betularesults4_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.844, 0.814, 0.883],
        :precision => [0.877, 0.879, 0.609],
        :recall => [0.844, 0.814, 0.706],
        :mae => [31.051, 36.706, 39.327],
        :rmse => [152.126, 159.676, 160.510],
        :time => [1602.9, 3148.41, 4.342],
        :memory => [2995.14, 228738.0, 561.252],
    )
    betularesults4 = vcat(betularesults4, betularesults4_manual)
    sort!(betularesults4, :accuracy, rev=true)

    poaceaeresults4 = CSV.File(joinpath(outputdir, "poaceaeresults4.csv")) |> DataFrame
    poaceaeresults4_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.796, 0.762, 0.802],
        :precision => [0.799, 0.813, 0.596],
        :recall => [0.796, 0.762, 0.625],
        :mae => [10.525, 11.83, 10.598],
        :rmse => [27.265, 27.84, 27.360],
        :time => [2995.14, 5493.77, 7.851],
        :memory => [391027.0, 422627.0, 665.004],
    )
    poaceaeresults4 = vcat(poaceaeresults4, poaceaeresults4_manual)
    sort!(poaceaeresults4, :accuracy, rev=true)

    latexify(select(betularesults4, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    latexify(select(poaceaeresults4, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    betularesults4, poaceaeresults4
end

# ╔═╡ 49454344-e2f9-4b63-8939-3fb0156956f2
begin
    betularesults7 = CSV.File(joinpath(outputdir, "betularesults7.csv")) |> DataFrame
    betularesults7_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.83, 0.808, 0.872],
        :precision => [0.866, 0.876, 0.578],
        :recall => [0.83, 0.808, 0.695],
        :mae => [40.005, 39.95, 42.069],
        :rmse => [159.748, 165.033, 168.523],
        :time => [2183.45, 3148.41, 2.020],
        :memory => [208284.0, 228738.0, 484.7],
    )
    betularesults7 = vcat(betularesults7, betularesults7_manual)
    sort!(betularesults7, :accuracy, rev=true)

    poaceaeresults7 = CSV.File(joinpath(outputdir, "poaceaeresults7.csv")) |> DataFrame
    poaceaeresults7_manual = DataFrame(
        :model => ["Conv-LSTM DNN", "Conv-GRU DNN", "MAGN"],
        :accuracy => [0.8, 0.754, 0.771],
        :precision => [0.804, 0.802, 0.522],
        :recall => [0.8, 0.754, 0.549],
        :mae => [10.416, 12.268, 12.235],
        :rmse => [27.833, 29.294, 29.771],
        :time => [4509.94, 5493.77, 3.985],
        :memory => [390109.0, 422627.0, 419.144],
    )
    poaceaeresults7 = vcat(poaceaeresults7, poaceaeresults7_manual)
    sort!(poaceaeresults7, :accuracy, rev=true)

    latexify(select(betularesults7, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    latexify(select(poaceaeresults7, resultsfeatures); env=:table, booktabs=false, latex=false) |> print
    betularesults7, poaceaeresults7
end

# ╔═╡ 1d8a14a6-05b6-40ef-b92e-1ffa03c946c6
begin
    CSV.write(joinpath(outputdir, "betula1_plos.csv"), betularesults1)
    CSV.write(joinpath(outputdir, "poaceae1_plos.csv"), poaceaeresults1)
    CSV.write(joinpath(outputdir, "betula4_plos.csv"), betularesults4)
    CSV.write(joinpath(outputdir, "poaceae4_plos.csv"), poaceaeresults4)
    CSV.write(joinpath(outputdir, "betula7_plos.csv"), betularesults7)
    CSV.write(joinpath(outputdir, "poaceae7_plos.csv"), poaceaeresults7)
end

# ╔═╡ a1d436a4-202c-4d34-abf9-ca628f84ad26
begin
    betulafe = CSV.File(joinpath(outputdir, "betulafe_xgboost.csv")) |> DataFrame
    poaceaefe = CSV.File(joinpath(outputdir, "poaceaefe_xgboost.csv")) |> DataFrame
    betulanmi = CSV.File(joinpath(outputdir, "betulanmi3.csv")) |> DataFrame
    poaceaenmi = CSV.File(joinpath(outputdir, "poaceaenmi3.csv")) |> DataFrame
    rename!(betulanmi, :name => :feature)
    rename!(poaceaenmi, :name => :feature)
    betulafe, poaceaefe, betulanmi, poaceaenmi
end

# ╔═╡ 0930a647-f2b2-45c9-8764-c5a1d6e97cf9
meteofeatures = Dict(
    "seeds" => "pollen concetration",
    "cc" => "cloud cover",
    "fg" => "wind speed",
    "hu" => "humidity",
    "pp" => "sea level pressure",
    "qq" => "radiation",
    "sd" => "snow depth",
    "ss" => "sunshine duration",
    "tg" => "mean temperature",
    "tn" => "minimum temperature",
    "tx" => "maximum temperature",
    "tgtntx" => "temperature",
)

# ╔═╡ 7bc51dff-56c8-4b8e-8d7b-e1c4e662ca9c
function combinefeautres(df::DataFrame, prefixes::Vector)::DataFrame
    scores = Dict(
        map(prefixes) do prefix
            if prefix isa Vector
                string(prefix...) => []
            else
                prefix => []
            end
        end
    )
    for prefix in prefixes
        if prefix isa Vector
            commonname = string(prefix...)
            for exactprefix in prefix
                for (i, name) in enumerate(df.feature)
                    if startswith(name, exactprefix)
                        push!(scores[commonname], df[i, :score])
                    end
                end
            end
        else
            for (i, name) in enumerate(df.feature)
                if startswith(name, prefix)
                    push!(scores[prefix], df[i, :score])
                end
            end
        end
    end
    prefixes = keys(scores) |> collect
    scores = map(prefixes) do prefix
        sum(scores[prefix])
    end |> collect
    maxscore = maximum(scores)
    scores = round.(scores ./ maxscore; digits=3)
    resdf = DataFrame(
        :feature => get.(Ref(meteofeatures), prefixes, ""),
        :score => round.(scores .* 100; digits=1)
    )
    sort(resdf, :score, rev=true)
end

# ╔═╡ 40ab72c7-151a-49dc-950b-3187a1a7c14b
let
    betulafi_df = sort(combinefeautres(betulafe, ["seeds", "cc", "fg", "hu", "pp", "qq", "sd", "ss", ["tg", "tn", "tx"]]), :score)
    poaceaefi_df = sort(combinefeautres(poaceaefe, ["seeds", "cc", "fg", "hu", "pp", "qq", "sd", "ss", ["tg", "tn", "tx"]]), :score)
    betulanmi_df = sort(combinefeautres(betulanmi, ["seeds", "cc", "fg", "hu", "pp", "qq", "sd", "ss", ["tg", "tn", "tx"]]), :score)
    poaceaenmi_df = sort(combinefeautres(poaceaenmi, ["seeds", "cc", "fg", "hu", "pp", "qq", "sd", "ss", ["tg", "tn", "tx"]]), :score)

    meteofeatures_id = Dict(map(x -> x[2] => x[1], enumerate(betulafi_df.feature)))

    CairoMakie.activate!()
    set_theme!(theme_light())

    fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 5.0 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    ax1 = Axis(
        fig[1, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="A1",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(betulafi_df.feature), betulafi_df.feature),
        yticklabelsize=fontsize * 1.125,
        xlabel="relative feature importance [%]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:100,
    )
    barplot!(
        ax1,
        1:length(betulafi_df.feature),
        betulafi_df.score,
        bar_labels=:y,
        direction=:x,
        label_formatter=(x -> string(x, " %")),
        colormap=:seaborn_colorblind,
        color=map(x -> meteofeatures_id[x], betulafi_df.feature)
    )

    ax2 = Axis(
        fig[1, 2],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="B1",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(poaceaefi_df.feature), poaceaefi_df.feature),
        yticklabelsize=fontsize * 1.125,
        xlabel="relative feature importance [%]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:100,
    )
    barplot!(
        ax2,
        1:length(poaceaefi_df.feature),
        poaceaefi_df.score,
        bar_labels=:y,
        direction=:x,
        label_formatter=(x -> string(x, " %")),
        colormap=:seaborn_colorblind,
        color=map(x -> meteofeatures_id[x], poaceaefi_df.feature)
    )

    ax3 = Axis(
        fig[2, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="A2",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(betulanmi_df.feature), betulanmi_df.feature),
        yticklabelsize=fontsize * 1.125,
        xlabel="relative normalized mutual information [%]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:100,
    )
    barplot!(
        ax3,
        1:length(betulanmi_df.feature),
        betulanmi_df.score,
        bar_labels=:y,
        direction=:x,
        label_formatter=(x -> string(x, " %")),
        colormap=:seaborn_colorblind,
        color=map(x -> meteofeatures_id[x], betulanmi_df.feature)
    )

    ax4 = Axis(
        fig[2, 2],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="B2",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(poaceaenmi_df.feature), poaceaenmi_df.feature),
        yticklabelsize=fontsize * 1.125,
        xlabel="relative normalized mutual information [%]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:100,
    )
    barplot!(
        ax4,
        1:length(poaceaenmi_df.feature),
        poaceaenmi_df.score,
        bar_labels=:y,
        direction=:x,
        label_formatter=(x -> string(x, " %")),
        colormap=:seaborn_colorblind,
        color=map(x -> meteofeatures_id[x], poaceaenmi_df.feature)
    )

    save(joinpath(assetsdir, "Fig4.png"), fig, px_per_unit=1)
    img = FileIO.load(joinpath(assetsdir, "Fig4.png"))
    save(joinpath(assetsdir, "Fig4.tif"), img)

    fig
end

# ╔═╡ 1798569f-d7e2-43ca-9f03-5d1ae8864988
let
    modelids = Dict(map(m -> m[2] => m[1], enumerate(betularesults1.model)))

    accuracies = Dict(map(m -> m => [], betularesults1.model))
    maes = Dict(map(m -> m => [], betularesults1.model))
    rmses = Dict(map(m -> m => [], betularesults1.model))
    times = Dict(map(m -> m => [], betularesults1.model))
    memories = Dict(map(m -> m => [], betularesults1.model))

    for result in (betularesults1, betularesults4, betularesults7, poaceaeresults1, poaceaeresults4, poaceaeresults7)
        for (i, model) in enumerate(result.model)
            push!(accuracies[model], result.accuracy[i])
            push!(maes[model], result.mae[i])
            push!(rmses[model], result.rmse[i])
            push!(times[model], result.time[i])
            push!(memories[model], result.memory[i])
        end
    end

    accuraciesmean = map(x -> x[1] => round(mean(x[2]) .* 100; digits=1), accuracies |> collect) |> collect
    sort!(accuraciesmean, by=last, rev=false)
    maesmean = map(x -> x[1] => round(mean(x[2]); digits=1), maes |> collect) |> collect
    sort!(maesmean, by=last, rev=true)
    rmsesmean = map(x -> x[1] => round(mean(x[2]); digits=1), rmses |> collect) |> collect
    sort!(rmsesmean, by=last, rev=true)
    timesmean = map(x -> x[1] => round(mean(x[2]); digits=0), times |> collect) |> collect
    sort!(timesmean, by=last, rev=true)
    memoriesmean = map(x -> x[1] => round(mean(x[2]); digits=0), memories |> collect) |> collect
    sort!(memoriesmean, by=last, rev=true)

    # accuraciesmean, maesmean, rmsesmean, timesmean, memoriesmean

    CairoMakie.activate!()
    set_theme!(theme_light())

    fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 5.0 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    ax1 = Axis(
        fig[1, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="A",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(accuraciesmean), first.(accuraciesmean)),
        yticklabelsize=fontsize * 1.125,
        xlabel="mean accuracy [%]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:100,
    )
    barplot!(
        ax1,
        1:length(accuraciesmean),
        last.(accuraciesmean),
        bar_labels=:y,
        direction=:x,
        label_formatter=(x -> string(x, " %")),
        colormap=:seaborn_colorblind,
        color=map(x -> modelids[x], first.(accuraciesmean))
    )

    ax2 = Axis(
        fig[1, 2],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="B",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(maesmean), first.(maesmean)),
        yticklabelsize=fontsize * 1.125,
        xlabel="mean MAE",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:10:50,
    )
    barplot!(
        ax2,
        1:length(maesmean),
        last.(maesmean),
        bar_labels=:y,
        direction=:x,
        colormap=:seaborn_colorblind,
        color=map(x -> modelids[x], first.(maesmean))
    )

    ax3 = Axis(
        fig[2, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="C",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(timesmean), first.(timesmean)),
        yticklabelsize=fontsize * 1.125,
        xlabel="mean execution time [s]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:1000:5000,
    )
    barplot!(
        ax3,
        1:length(timesmean),
        last.(timesmean),
        bar_labels=:y,
        direction=:x,
        colormap=:seaborn_colorblind,
        color=map(x -> modelids[x], first.(timesmean))
    )

    ax4 = Axis(
        fig[2, 2],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="D",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yticklabelsvisible=true,
        yticks=(1:length(memoriesmean), first.(memoriesmean)),
        yticklabelsize=fontsize * 1.125,
        xlabel="mean memory consumption [megabytes]",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.2,
        xticks=0:1e5:4e5,
    )
    barplot!(
        ax4,
        1:length(memoriesmean),
        last.(memoriesmean),
        bar_labels=:y,
        direction=:x,
        colormap=:seaborn_colorblind,
        color=map(x -> modelids[x], first.(memoriesmean))
    )

    save(joinpath(assetsdir, "Fig5.png"), fig, px_per_unit=1)
    img = FileIO.load(joinpath(assetsdir, "Fig5.png"))
    save(joinpath(assetsdir, "Fig5.tif"), img)

    fig
end

# ╔═╡ ae77c45d-8be6-405d-9a1c-60ad0dafabf5
begin
    betularules = CSV.File(joinpath(outputdir, "betularules.csv")) |> DataFrame
    poaceaerules = CSV.File(joinpath(outputdir, "poaceaerules.csv")) |> DataFrame
    # betularules, poaceaerules

    CairoMakie.activate!()
    set_theme!(theme_light())

    fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 2.5 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    ax1 = Axis(
        fig[1, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="A",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        xlabel="support",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.35,
        # xticks=0:10:100,
        ylabel="confidence",
        ylabelfont=:regular,
        ylabelsize=fontsize * 1.35,
        # yticks=0:10:100,
        yticklabelsize=fontsize * 1.125,
    )
    scatter!(
        ax1,
        betularules.support .* 3700,
        betularules.confidence,
        colormap=:rainbow_bgyr_35_85_c72_n256,
        color=betularules.lift,
        markersize=betularules.support .* 10000
    )
    Colorbar(
        fig[1, 2],
        limits=(0, maximum(betularules.lift)),
        colormap=:rainbow_bgyr_35_85_c72_n256,
        vertical=true,
        width=20,
        label="lift"
    )

    ax2 = Axis(
        fig[1, 4],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        titlealign=:left,
        title="B",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        xlabel="support",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.35,
        # xticks=0:10:100,
        ylabel="confidence",
        ylabelfont=:regular,
        ylabelsize=fontsize * 1.35,
        # yticks=0:10:100,
        yticklabelsize=fontsize * 1.125,
    )
    scatter!(
        ax2,
        poaceaerules.support .* 3700,
        poaceaerules.confidence,
        colormap=:rainbow_bgyr_35_85_c72_n256,
        color=poaceaerules.lift,
        markersize=poaceaerules.support .* 10000
    )
    Colorbar(
        fig[1, 5],
        limits=(0, maximum(poaceaerules.lift)),
        colormap=:rainbow_bgyr_35_85_c72_n256,
        vertical=true,
        width=20,
        label="lift"
    )

    colsize!(fig.layout, 3, Relative(0.05))

    save(joinpath(assetsdir, "Fig6.png"), fig, px_per_unit=1)
    img = FileIO.load(joinpath(assetsdir, "Fig6.png"))
    save(joinpath(assetsdir, "Fig6.tif"), img)

    fig
end

# ╔═╡ 6de65a3b-42e8-409a-ad9c-85944212004a
begin
    meteosymbols = ["cc", "fg", "hu", "pp", "qq", "sd", "ss", "tg", "tn", "tx"]
    scores = OrderedDict("feature" => meteosymbols, "ema3" => [], "ema7" => [], "ema20" => [], "emasum" => [])
    for meteofeature in meteosymbols
        ema3_index = findfirst(x -> x == "$(meteofeature)_ema3", betulanmi.feature)
        push!(scores["ema3"], betulanmi[ema3_index, :score])
        ema7_index = findfirst(x -> x == "$(meteofeature)_ema7", betulanmi.feature)
        push!(scores["ema7"], betulanmi[ema7_index, :score])
        ema20_index = findfirst(x -> x == "$(meteofeature)_ema20", betulanmi.feature)
        push!(scores["ema20"], betulanmi[ema20_index, :score])
        push!(scores["emasum"], sum([last(scores["ema3"]), last(scores["ema7"]), last(scores["ema20"])]))
    end

    push!(scores["feature"], "tgtntx")
    push!(scores["ema3"], sum(scores["ema3"][end-2:end]))
    push!(scores["ema7"], sum(scores["ema7"][end-2:end]))
    push!(scores["ema20"], sum(scores["ema20"][end-2:end]))
    push!(scores["emasum"], sum(scores["emasum"][end-2:end]))

    laggedfi = DataFrame(scores...)
end

# ╔═╡ 49c6d059-5706-4256-b4d4-21872049c289
laggedfi_p = let
    meteosymbols = ["cc", "fg", "hu", "pp", "qq", "sd", "ss", "tg", "tn", "tx"]
    scores = OrderedDict("feature" => meteosymbols, "ema3" => [], "ema7" => [], "ema20" => [], "emasum" => [])
    for meteofeature in meteosymbols
        ema3_index = findfirst(x -> x == "$(meteofeature)_ema3", poaceaenmi.feature)
        push!(scores["ema3"], poaceaenmi[ema3_index, :score])
        ema7_index = findfirst(x -> x == "$(meteofeature)_ema7", poaceaenmi.feature)
        push!(scores["ema7"], poaceaenmi[ema7_index, :score])
        ema20_index = findfirst(x -> x == "$(meteofeature)_ema20", poaceaenmi.feature)
        push!(scores["ema20"], poaceaenmi[ema20_index, :score])
        push!(scores["emasum"], sum([last(scores["ema3"]), last(scores["ema7"]), last(scores["ema20"])]))
    end

    push!(scores["feature"], "tgtntx")
    push!(scores["ema3"], sum(scores["ema3"][end-2:end]))
    push!(scores["ema7"], sum(scores["ema7"][end-2:end]))
    push!(scores["ema20"], sum(scores["ema20"][end-2:end]))
    push!(scores["emasum"], sum(scores["emasum"][end-2:end]))

    DataFrame(scores...)
end

# ╔═╡ afc79506-13cb-4d68-aded-6d9dd02a1269
let
    fontsize = 12 * 2
    fig = Figure(size=(7.5 * 300, 5.0 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    df = copy(laggedfi[1:7, :])

    ax1 = Axis(
        fig[1, 1],
        yautolimitmargin=(0.15, 0.15),
        xautolimitmargin=(0.15, 0.15),
        # titlealign=:left,
        # title="A",
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        xlabel="support",
        xlabelfont=:regular,
        xlabelsize=fontsize * 1.35,
        # xticks=0:10:100,
        ylabel="confidence",
        ylabelfont=:regular,
        ylabelsize=fontsize * 1.35,
        # yticks=0:10:100,
        yticklabelsize=fontsize * 1.125,
        # yscale = :log10,
    )

    series = []
    for i in 1:nrow(df)
        s = scatterlines!(
            ax1,
            collect(1:3),
            collect(df[i, 2:4]),
            marker=:hexagon,
            strokewidth=1,
            markersize=20,
            linewidth=3,
            # color = :orange,
        )
        push!(series, s)
    end

    Legend(fig[1, 2],
        series,
        map(k -> meteofeatures[k], string.(df[1:length(series), :feature])) |> collect
    )

    save(joinpath(assetsdir, "Fig8.png"), fig, px_per_unit=1)
    img = FileIO.load(joinpath(assetsdir, "Fig8.png"))
    save(joinpath(assetsdir, "Fig8.tif"), img)

    fig
end

# ╔═╡ a54a1247-e84c-4f7d-9646-6746cb5cb4ed
sort(laggedfi, :emasum, rev=true)

# ╔═╡ ac46efae-f40f-4428-8f62-844abeaf29ab
sort(laggedfi_p, :emasum, rev=true)

# ╔═╡ 009ffc44-cf01-4bc9-b714-0000efd15589
OrderedDict(
    meteofeatures["qq"] => 0.0635639,
    meteofeatures["cc"] => 0.0532244,
    meteofeatures["fg"] => 0.0387565,
    meteofeatures["pp"] => 0.0218597,
    meteofeatures["sd"] => 0.015743,
)

# ╔═╡ 553679ab-7e7a-471c-8e01-bbcabde11111
OrderedDict(
    meteofeatures["qq"] => 0.0635639,
    meteofeatures["cc"] => 0.0532244,
    meteofeatures["fg"] => 0.0387565,
    meteofeatures["pp"] => 0.0218597,
    meteofeatures["sd"] => 0.015743,
)

# ╔═╡ Cell order:
# ╟─049662d0-f6db-11ef-13ba-a35a789e5b2a
# ╟─c596973a-d074-4ad4-b6b5-4f7e3449c99b
# ╟─9f797ff7-3124-40c3-8ae2-886c7407a40b
# ╠═2fe4f0f3-9add-4780-9e6c-c5ef53879112
# ╠═179fa328-f95b-4026-9669-b28c4e732951
# ╠═49454344-e2f9-4b63-8939-3fb0156956f2
# ╠═1d8a14a6-05b6-40ef-b92e-1ffa03c946c6
# ╠═a1d436a4-202c-4d34-abf9-ca628f84ad26
# ╠═0930a647-f2b2-45c9-8764-c5a1d6e97cf9
# ╠═7bc51dff-56c8-4b8e-8d7b-e1c4e662ca9c
# ╟─40ab72c7-151a-49dc-950b-3187a1a7c14b
# ╟─1798569f-d7e2-43ca-9f03-5d1ae8864988
# ╟─ae77c45d-8be6-405d-9a1c-60ad0dafabf5
# ╠═6de65a3b-42e8-409a-ad9c-85944212004a
# ╠═49c6d059-5706-4256-b4d4-21872049c289
# ╠═afc79506-13cb-4d68-aded-6d9dd02a1269
# ╠═a54a1247-e84c-4f7d-9646-6746cb5cb4ed
# ╠═ac46efae-f40f-4428-8f62-844abeaf29ab
# ╠═009ffc44-cf01-4bc9-b714-0000efd15589
# ╠═553679ab-7e7a-471c-8e01-bbcabde11111
