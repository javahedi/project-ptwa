using CairoMakie, JLD2, Statistics, LaTeXStrings
const L=12; const g=0.3; const Ndis=100; const Nmc=100; const chi=100; const dt=0.1
const W_list=[0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]; const seed_list=0:8
const tmin=2.0; const tcut=20.0
window_average(t,I)=mean(I[(t.>=tmin).&(t.<=tcut)])
function make_plot(sampler::Symbol)
    set_theme!(Theme(fontsize=23,Axis=(xticklabelsize=18,yticklabelsize=18,xlabelsize=28,ylabelsize=28)))
    fig=Figure(size=(600,420))
    ax=Axis(fig[1,1],xlabel="time",ylabel="Imbalance",xgridvisible=true,ygridvisible=true)
    text!(ax,0.05,0.05,text=string(sampler),space=:relative,fontsize=24,color=:black)
    cmap=:viridis; crange=(minimum(W_list),maximum(W_list)); Iavg=Float64[]; I_ed_avg=Float64[]
    for W in W_list
        d=load(joinpath(@__DIR__,"data","ptwa","pTWA_Z3_parafermion_L$(L)_g$(g)_W$(W)_Ndis$(Ndis)_Nmc$(Nmc)_$(sampler).jld2"))
        lines!(ax,d["times"],d["I_mean"];linewidth=1,color=W,colormap=cmap,colorrange=crange,linestyle=:dash)
        push!(Iavg,window_average(d["times"],d["I_mean"]))
    end
    for (seed,W) in zip(seed_list,W_list)
        d=load(joinpath(@__DIR__,"data","ed","ED_Z3_L$(L)_g$(g)_W$(W)_Ndis$(Ndis)_seed$(seed)_chi$(chi)_dt$(dt).jld2"))
        lines!(ax,d["times"],d["imbalance_avg"];linewidth=1,color=W,colormap=cmap,colorrange=crange,linestyle=:solid)
        push!(I_ed_avg,window_average(d["times"],d["imbalance_avg"]))
    end
    Colorbar(fig[1,2],colormap=cmap,limits=crange,label=L"\mathrm{Disorder\ strength}\ W",width=14,labelsize=16)
    ax_in=Axis(fig[1,1],width=Relative(0.38),height=Relative(0.38),halign=0.97,valign=0.97,xlabel=L"W",ylabel=L"\bar{\mathcal{I}}_{t\in[2,20]}",xticklabelsize=9,yticklabelsize=9,xlabelsize=16,ylabelsize=14)
    scatter!(ax_in,W_list,Iavg;marker=:xcross,color=:black,markersize=9)
    lines!(ax_in,W_list,Iavg;color=:black,linewidth=1.5)
    scatter!(ax_in,W_list,I_ed_avg;marker=:circle,markersize=8,color=:red)
    save(joinpath(@__DIR__,"Fig3_Imbalance_L$(L)_$(sampler).pdf"),fig)
    save(joinpath(@__DIR__,"Fig3_Imbalance_L$(L)_$(sampler).png"),fig,px_per_unit=2)
end
make_plot(:gaussian); make_plot(:discrete)
