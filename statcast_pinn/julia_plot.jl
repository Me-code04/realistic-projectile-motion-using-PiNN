using CSV, DataFrames, Plots

# -----------------------------
# Helpers
# -----------------------------
norm3(x,y,z) = sqrt(x*x + y*y + z*z)

function unitvec(x,y,z; eps=1e-9)
    n = norm3(x,y,z)
    n < eps && return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)
end

function read_traj(path::String)
    df = CSV.read(path, DataFrame)
    return df
end

# Strike zone (MLB rulebook: 17" plate wide; zone depends on batter.
# We'll use a configurable "typical" zone in meters.)
const IN_TO_M = 0.0254
plate_half_width = (17/2) * IN_TO_M  # half width of home plate, meters (approx)
zone_z_low = 0.50      # meters (adjust)
zone_z_high = 1.10     # meters (adjust)

# You must decide your coordinate frame:
# We'll assume:
#   x = horizontal (left/right), z = vertical (up), y = direction toward plate.
# Choose a plate plane location y_plate (meters) to draw zone there.
# If your y increases away from plate, set accordingly.
y_plate = 0.0          # set this to your "plate plane" y coordinate in your frame

function draw_plate_and_zone!(plt; y_plate=0.0, plate_half_width=0.216, zlow=0.50, zhigh=1.10)
    # Draw strike zone rectangle on plane y=y_plate: x in [-w,w], z in [zlow, zhigh]
    xs = [-plate_half_width, plate_half_width, plate_half_width, -plate_half_width, -plate_half_width]
    ys = fill(y_plate, length(xs))
    zs = [zlow, zlow, zhigh, zhigh, zlow]
    plot!(plt, xs, ys, zs, lw=3, label="Strike zone")

    # Draw a simple home plate outline (approx) on z=0 plane at y=y_plate
    # We’ll draw a 5-sided polygon (rough):
    w = plate_half_width
    d = 0.216  # ~8.5" depth in meters (rough); tweak
    xhp = [-w,  w,  w,  0.0, -w, -w]
    yhp = [y_plate, y_plate, y_plate + d, y_plate + d + 0.10, y_plate + d, y_plate]
    zhp = zeros(length(xhp))
    plot!(plt, xhp, yhp, zhp, lw=2, label="Home plate")
end

# -----------------------------
# Load one or multiple trajectories
# -----------------------------
# Put multiple CSVs here to overlay:
traj_files = ["traj.csv"]   # e.g. ["traj_fastball.csv", "traj_slider.csv"]

dfs = [read_traj(f) for f in traj_files]

# -----------------------------
# Static 3D plot with overlays
# -----------------------------
plt = plot(title="PINN Pitch Trajectories", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", legend=:topright)

for (i, df) in enumerate(dfs)
    plot!(plt, df.x, df.y, df.z, lw=2, label="traj $(i)")
end

draw_plate_and_zone!(plt; y_plate=y_plate, plate_half_width=plate_half_width, zlow=zone_z_low, zhigh=zone_z_high)

display(plt)

# -----------------------------
# Animation: trajectory + spin & Magnus direction vectors
# -----------------------------
# Toggle these:
DO_ANIM = true
VECTOR_SCALE = 0.25   # arrow length scale (meters)
STEP = 2              # use every STEP-th sample to speed up

if DO_ANIM
    # Precompute axis limits from all trajs
    allx = vcat([df.x for df in dfs]...)
    ally = vcat([df.y for df in dfs]...)
    allz = vcat([df.z for df in dfs]...)
    xlims = (minimum(allx), maximum(allx))
    ylims = (minimum(ally), maximum(ally))
    zlims = (minimum(allz), maximum(allz))

    # We animate just the first trajectory for vectors (easy to expand to all)
    df = dfs[1]
    n = nrow(df)

    anim = @animate for k in 2:STEP:n
        p = plot(title="Trajectory + Spin & Magnus (frame $k/$n)",
                 xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                 xlims=xlims, ylims=ylims, zlims=zlims,
                 legend=:topright)

        # Draw all trajectories faintly (history)
        for (i, dfi) in enumerate(dfs)
            plot!(p, dfi.x, dfi.y, dfi.z, lw=1, label=(i==1 ? "traj 1" : "traj $i"))
        end

        draw_plate_and_zone!(p; y_plate=y_plate, plate_half_width=plate_half_width, zlow=zone_z_low, zhigh=zone_z_high)

        # Current point
        x = df.x[k]; y = df.y[k]; z = df.z[k]
        scatter!(p, [x], [y], [z], ms=4, label="ball")

        # Spin vector omega (unit)
        ox = df.ox[k]; oy = df.oy[k]; oz = df.oz[k]
        (ux,uy,uz) = unitvec(ox,oy,oz)
        # Arrow end
        x2 = x + VECTOR_SCALE*ux
        y2 = y + VECTOR_SCALE*uy
        z2 = z + VECTOR_SCALE*uz
        plot!(p, [x,x2], [y,y2], [z,z2], lw=3, label="spin ω̂")

        # Magnus direction ~ ω × u, where u = v - w
        vx = df.vx[k]; vy = df.vy[k]; vz = df.vz[k]
        wx = df.wx[k]; wy = df.wy[k]; wz = df.wz[k]
        uxrel = vx - wx; uyrel = vy - wy; uzrel = vz - wz

        mx = oy*uzrel - oz*uyrel
        my = oz*uxrel - ox*uzrel
        mz = ox*uyrel - oy*uxrel
        (mxu,myu,mzu) = unitvec(mx,my,mz)

        x3 = x + VECTOR_SCALE*mxu
        y3 = y + VECTOR_SCALE*myu
        z3 = z + VECTOR_SCALE*mzu
        plot!(p, [x,x3], [y,y3], [z,z3], lw=3, label="Magnus dir (ω×u)̂")
    end

    gif(anim, "pitch_vectors.gif", fps=60)
    println("Saved pitch_vectors.gif")
end

println("Generating 2D trajectory plots...")

# Use first trajectory as reference (easy to extend to multiple)
df = dfs[1]

# -----------------------------
# x–y (Top view)
# -----------------------------
plt_xy = plot(
    df.x, df.y,
    xlabel="x (m)",
    ylabel="y (m)",
    title="Top View: x–y",
    lw=2,
    legend=false,
    aspect_ratio=:equal
)
savefig(plt_xy, "traj_xy.png")

# -----------------------------
# y–z (Side view)
# -----------------------------
plt_yz = plot(
    df.y, df.z,
    xlabel="y (m)",
    ylabel="z (m)",
    title="Side View: y–z",
    lw=2,
    legend=false
)
savefig(plt_yz, "traj_yz.png")

# -----------------------------
# x–z (Front view)
# -----------------------------
plt_xz = plot(
    df.x, df.z,
    xlabel="x (m)",
    ylabel="z (m)",
    title="Front View: x–z",
    lw=2,
    legend=false,
    aspect_ratio=:equal
)
savefig(plt_xz, "traj_xz.png")

println("Saved: traj_xy.png, traj_yz.png, traj_xz.png")

speed = sqrt.(df.vx.^2 .+ df.vy.^2 .+ df.vz.^2)

plt_vt = plot(
    df.t, speed,
    xlabel="time (s)",
    ylabel="|v| (m/s)",
    title="Speed vs Time",
    lw=2,
    legend=false
)
savefig(plt_vt, "speed_vs_time.png")
plt_zt = plot(
    df.t, df.z,
    xlabel="time (s)",
    ylabel="z (m)",
    title="Height vs Time",
    lw=2,
    legend=false
)
savefig(plt_zt, "height_vs_time.png")
plt_yz_zone = plot(
    df.y, df.z,
    xlabel="y (m)",
    ylabel="z (m)",
    title="Side View with Strike Zone",
    lw=2,
    legend=false
)

# Strike zone rectangle
plot!(
    plt_yz_zone,
    [y_plate, y_plate],
    [zone_z_low, zone_z_high],
    lw=4
)

savefig(plt_yz_zone, "traj_yz_zone.png")

println("Ranges:")
println("x: ", (minimum(df.x), maximum(df.x)))
println("y: ", (minimum(df.y), maximum(df.y)))
println("z: ", (minimum(df.z), maximum(df.z)))
println("t: ", (minimum(df.t), maximum(df.t)))
