using LinearAlgebra
using Plots

# ----------------------------
# USER SETTINGS (customize)
# ----------------------------
g = 9.81

r0 = [0.0, 1.5]          # initial coordinate-vector [x0, y0] (meters)

speed0 = 22.0            # launch speed (m/s)
theta_deg = 45.0         # launch angle in degrees (from +x axis)

# Drag model:
#   a_drag = -(k1 * v) - (k2 * |v| * v)
# Choose k1 (linear) and/or k2 (quadratic). Set to 0.0 to disable.
k1 = 0.05                # 1/s (linear drag)
k2 = 0.01                # 1/m (quadratic drag)

# Bounce model at y = 0:
e = 0.75                 # coefficient of restitution (0..1): vy_after = -e*vy_before
ground_friction = 0.15   # reduces vx on each bounce: vx_after = (1-ground_friction)*vx_before

# Simulation controls:
dt = 0.005
tmax = 30.0
max_bounces = 50

# "negligible motion" stopping criteria:
speed_eps = 0.25         # m/s
bounce_speed_eps = 0.35  # if impact speed is below this, stop after that bounce


# ----------------------------
# Helper functions
# ----------------------------
deg2rad(θ) = θ * (pi / 180)

function accel(v::Vector{Float64}, g::Float64, k1::Float64, k2::Float64)
    # Gravity + drag
    vmag = norm(v)
    a_drag = -(k1 .* v) - (k2 * vmag .* v)
    return [0.0, -g] .+ a_drag
end

function init_velocity(speed0, theta_deg)
    θ = deg2rad(theta_deg)
    return [speed0*cos(θ), speed0*sin(θ)]
end


# ----------------------------
# Simulate with bounces (Euler-Cromer / semi-implicit Euler)
# ----------------------------
function simulate(r0, speed0, theta_deg; g=9.81, k1=0.0, k2=0.0,
                  e=0.8, ground_friction=0.1, dt=0.01, tmax=20.0,
                  max_bounces=25, speed_eps=0.2, bounce_speed_eps=0.3)

    r = Float64.(r0)
    v = init_velocity(speed0, theta_deg)

    Rs = Vector{Vector{Float64}}()
    Vs = Vector{Vector{Float64}}()
    Ts = Float64[]
    push!(Rs, copy(r)); push!(Vs, copy(v)); push!(Ts, 0.0)

    bounces = 0
    t = 0.0

    while t < tmax
        # update velocity first (semi-implicit Euler)
        a = accel(v, g, k1, k2)
        v .= v .+ a .* dt

        # update position
        r .= r .+ v .* dt
        t += dt

        # ground collision at y=0 (simple plane)
        if r[2] < 0.0
            # approximate impact handling: snap to ground
            r[2] = 0.0

            impact_speed = abs(v[2])
            # bounce: reverse vy with restitution
            v[2] = -e * v[2]
            # lose some horizontal speed on bounce (crude friction/energy loss)
            v[1] = (1 - ground_friction) * v[1]

            bounces += 1
            if bounces >= max_bounces || impact_speed < bounce_speed_eps
                push!(Rs, copy(r)); push!(Vs, copy(v)); push!(Ts, t)
                break
            end
        end

        push!(Rs, copy(r)); push!(Vs, copy(v)); push!(Ts, t)

        # stop if basically not moving (and on/near ground)
        if norm(v) < speed_eps && r[2] <= 1e-6
            break
        end
    end

    return Ts, Rs, Vs
end

T, R, V = simulate(r0, speed0, theta_deg;
    g=g, k1=k1, k2=k2, e=e, ground_friction=ground_friction,
    dt=dt, tmax=tmax, max_bounces=max_bounces,
    speed_eps=speed_eps, bounce_speed_eps=bounce_speed_eps
)

# Convert R to x,y arrays for plotting
x = [ri[1] for ri in R]
y = [ri[2] for ri in R]

# ----------------------------
# Static plot (trajectory)
# ----------------------------
p_traj = plot(x, y, xlabel="x (m)", ylabel="y (m)", title="Projectile (drag + bounce)",
              legend=false, grid=true)
hline!(p_traj, [0.0])  # ground
display(p_traj)

# ----------------------------
# Animation (trail + point)
# ----------------------------
xmin, xmax = minimum(x), maximum(x)
ymax = maximum(y)

padx = 0.05 * (xmax - xmin + 1e-9)
pady = 0.10 * (ymax + 1e-9)

anim = @animate for i in 1:length(x)
    plot(x[1:i], y[1:i],
         xlim=(xmin - padx, xmax + padx),
         ylim=(0.0, ymax + pady),
         xlabel="x (m)", ylabel="y (m)",
         title="t = $(round(T[i], digits=2)) s",
         legend=false, grid=true)
    hline!([0.0])
    scatter!([x[i]], [y[i]], markersize=5)
end

gif(anim, "projectile_bounce.gif", fps=60)
