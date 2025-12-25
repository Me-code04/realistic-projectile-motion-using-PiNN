using CSV, DataFrames, Plots

df = CSV.read("traj.csv", DataFrame)

# 3D trajectory
plt = plot(df.x, df.y, df.z, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
           lw=2, legend=false, title="PINN Pitch Trajectory")
display(plt)

# Animation
anim = @animate for k in 2:nrow(df)
    plot(df.x[1:k], df.y[1:k], df.z[1:k],
         xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
         lw=2, legend=false, title="PINN Pitch Trajectory")
end

gif(anim, "pitch.gif", fps=60)
println("Saved pitch.gif")
