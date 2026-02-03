using JLD2
using CairoMakie
using FileIO

# ---------------------------- Paths ------------------------------------------
const PTWA_DIR = "PWTA/imbalance_differentSize"
const ED_DIR   = "ED_new"
const OUT_DIR  = "plots"
mkpath(OUT_DIR)

# ---------------------------- Utilities --------------------------------------

function load_tI(path)
    d = load(path)
    t = d["t"]
    I = d["I"]
    return t, I
end

function ptwa_file(L, W, g)
    return joinpath(PTWA_DIR, "pTWA_NN_Z3_domainwall_L$(L)_W$(W)_g$(g)_OPT.jld2")
    
    
end

function ed_file(L, W, g)
    patterns = [
        #joinpath(ED_DIR, "ED_Krylov_fixedN_domainwall_Z3_L$(L)_W$(W)_g$(g).jld2"),
        joinpath(ED_DIR, "ED_Krylov_fixedN_domainwall_Z3_L$(L)_W$(W)_g$(g)_new.jld2")
    ]
    
    for pattern in patterns
        if isfile(pattern)
            return pattern
        end
    end
    error("No ED file found for L=$L, W=$W, g=$g")
end

# ---------------------------- Plot -------------------------------------------

function plot_comparison()
    # Parameters
    W_list = [2.5, 4.5, 6.0]
    L_ptwa = [12, 24, 48]
    L_ed = 12
    g = 0.5
    
    # Colors and styles
    colors = Dict(12 => :black, 24 => :blue, 48 => :red)
    line_styles = Dict(12 => :solid, 24 => :dash, 48 => :dot)
    
    # Create figure
    fig = Figure(size=(1200, 360), fontsize=18)
    
    # Store elements for legend
    legend_elements = []
    legend_labels = []
    
    for (i, W) in enumerate(W_list)
        ax = Axis(fig[1, i])
        
        # Plot pTWA lines (all have dt=0.1)
        for L in L_ptwa
            try
                t_ptwa, I_ptwa = load_tI(ptwa_file(L, W, g))
                line = lines!(ax, t_ptwa, I_ptwa, 
                      linewidth=2.5, 
                      color=colors[L],
                      linestyle=line_styles[L])
                
                # Add to legend (only once)
                if i == 1
                    push!(legend_elements, line)
                    push!(legend_labels, "pTWA L=$L")
                end
            catch e
                println("Skipping pTWA L=$L, W=$W: $e")
            end
        end
        
        # Plot ED points (dt=0.2) - use every 5th point (every 1.0 second)
        try
            t_ed, I_ed = load_tI(ed_file(L_ed, W, g))
            
            # ED has 1001 points (0:0.2:200), take every 5th => 201 points
            indices = 1:20:length(t_ed)
            scatter_plot = scatter!(ax, t_ed[indices], I_ed[indices],
                    marker=:circle,
                    markersize=8,
                    strokewidth=1.5,
                    color=:red,
                    label="ED L=$L_ed")
            
            # Add to legend (only once)
            if i == 1
                push!(legend_elements, scatter_plot)
                push!(legend_labels, "ED L=$L_ed")
            end
        catch e
            println("Skipping ED L=$L_ed, W=$W: $e")
        end
        
        # Axis labels
        ax.xlabel = "time t"
        ax.ylabel = i == 1 ? "Imbalance I(t)" : ""
        ax.title = "W = $W, g = $g"
        
        # Grid and limits
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = (:gray, 0.3)
        ax.ygridcolor = (:gray, 0.3)
        ylims!(ax, 0.7, 1.1)
        xlims!(ax, 0, 200)
    end
    
    # Add legend
    Legend(fig[1, end+1], legend_elements, legend_labels, 
           orientation=:vertical, tellwidth=false, tellheight=false)
    
    # Save
    save(joinpath(OUT_DIR, "comparison.pdf"), fig)
    save(joinpath(OUT_DIR, "comparison.png"), fig)
    
    println("Saved comparison.{pdf,png}")
    println("Note: ED data has dt=0.2, pTWA has dt=0.1")
    println("ED markers shown every 1.0 time unit")
    return fig
end

# ---------------------------- Run --------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    fig = plot_comparison()
    display(fig)
end
