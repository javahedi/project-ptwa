using CairoMakie, JLD2, LaTeXStrings
function make_plot()
    set_theme!(Theme(fontsize=20,Axis=(xlabelsize=22,ylabelsize=22,xticklabelsize=18,yticklabelsize=18)))
    L=24
    Ntraj=2000
    chi=80
    dt=0.05
    ns=collect(3:7)
    heat_ns=[3,5,7]
    ptdir=joinpath(@__DIR__,"data","ptwa")
    tedir=joinpath(@__DIR__,"data","tebd")
    qT=Dict(); ST=Dict(); tT=Dict(); IT=Dict(); qP=Dict(); SP=Dict(); tP=Dict(); IP=Dict()
    for n in ns
        te=load(joinpath(tedir,"TEBD_Zn_domainwall_singlehop_n$(n)_L$(L)_chi$(chi)_dt$(dt).jld2"))
        pt=load(joinpath(ptdir,"pTWA_Zn_domainwall_singlehop_n$(n)_L$(L)_gaussian_N$(Ntraj).jld2"))
        qT[n]=te["qs"]; ST[n]=te["Sa"]; tT[n]=te["times"]; IT[n]=te["I"]
        qP[n]=pt["qs"]; SP[n]=pt["Sa_avg"]; tP[n]=pt["times"]; IP[n]=pt["I_avg"]
    end
    fig=Figure(size=(1000,800))
    colgap!(fig.layout,5) 
    rowgap!(fig.layout,5)
    ax1=Axis(fig[1,1],title=L"TEBD,\;n=3",ylabel="time")
     ax2=Axis(fig[1,2],title=L"TEBD,\;n=5")
      ax3=Axis(fig[1,3],title=L"TEBD,\;n=7")
    ax4=Axis(fig[2,1],title=L"pTWA,\;n=3",xlabel=L"q",ylabel="time")
    ax5=Axis(fig[2,2],title=L"pTWA,\;n=5",xlabel=L"q")
    ax6=Axis(fig[2,3],title=L"pTWA,\;n=7",xlabel=L"q")

    smax=maximum(vcat([maximum(ST[n]) for n in heat_ns],[maximum(SP[n]) for n in heat_ns])); cr=(0.0,smax)
   
    hm=heatmap!(ax1,qT[3],tT[3],ST[3],colormap=:tofino25,colorrange=cr)
    heatmap!(ax2,qT[5],tT[5],ST[5],colormap=:tofino25,colorrange=cr)
    heatmap!(ax3,qT[7],tT[7],ST[7],colormap=:tofino25,colorrange=cr)
    heatmap!(ax4,qP[3],tP[3],SP[3],colormap=:tofino25,colorrange=cr)
    heatmap!(ax5,qP[5],tP[5],SP[5],colormap=:tofino25,colorrange=cr)
    heatmap!(ax6,qP[7],tP[7],SP[7],colormap=:tofino25,colorrange=cr)
    Colorbar(fig[1:2,4],hm,label=L"S_0(q,t)")
    colsize!(fig.layout,4,Auto(0.2))
    ax7=Axis(fig[3,1:3],xlabel="time",ylabel="Imbalance",title="Domain-wall imbalance"); colors=[:blue,:orange,:green,:red,:purple]
    for (i,n) in enumerate(ns)
        lines!(ax7,tT[n],IT[n],color=colors[i],linewidth=1)
    end
    for (i,n) in enumerate(ns) 
        lines!(ax7,tP[n],IP[n],color=colors[i],linestyle=:dash,linewidth=1)
    end
    Legend(fig[3,4],[LineElement(color=:black,linewidth=1),LineElement(color=:black,linestyle=:dash,linewidth=1),[LineElement(color=colors[i],linewidth=1) for i in 1:5]...],["TEBD","pTWA",L"n=3",L"n=4",L"n=5",L"n=6",L"n=7"],framevisible=false)

    resize_to_layout!(fig); 
    save(joinpath(@__DIR__,"Fig6_Zn.png"),fig,px_per_unit=2); 
    save(joinpath(@__DIR__,"Fig6_Zn.pdf"),fig)
end
make_plot()
