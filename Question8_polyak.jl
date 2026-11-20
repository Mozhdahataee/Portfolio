using JuMP, GLPK, LinearAlgebra
import MathOptInterface as MOI
using Plots

m = Model(GLPK.Optimizer)
@variable(m, x[1:15], Bin)
@variable(m, y[1:15, 1:3], Bin)

a = [12, 24, 20, 28, 10, 11, 30, 27, 22, 25, 40, 38, 32, 34, 36]
w = [14, 8, 7, 10, 13, 12, 9, 8, 7, 8, 12, 11, 10, 9, 10] 
c = [28, 32, 35]

# Remaining constraints
@constraint(m, [i in 1:15], sum(y[i, j] for j in 1:3) == x[i])
@constraint(m, x[2] <= x[1])
@constraint(m, x[3] <= x[1])
@constraint(m, x[4] <= x[1])
@constraint(m, x[7] <= x[5])
@constraint(m, x[8] <= x[5])
@constraint(m, x[9] <= x[6])
@constraint(m, x[10] <= x[6])
@constraint(m, x[11] <= x[7])
@constraint(m, x[12] <= x[8])
@constraint(m, x[15] <= x[10])
@constraint(m, x[11] + x[12] <= 1)
@constraint(m, x[2] + x[3] + x[4] + x[7] + x[8] == 3)

# Initial multipliers
u = [0.0, 0.0, 0.0]

# Parameters for Polyak-type step size
epsilon = 1.0
LB = 244.0

# Keep track of our best upper bound and its corresponding solution
best_bound = Inf
best_u = copy(u)
best_iteration = 0
max_iterations = 1000

bounds = Float64[]
best_bounds = Float64[]
best_x = zeros(15)
best_y = zeros(15, 3)
best_gamma = zeros(3)
best_capacity = zeros(3)

# Iterating over k's
for k in 1:max_iterations

    # Step 2: Update objective using current multipliers
    @objective(m, Max,
        sum(a[i] * x[i] for i in 1:15) +
        sum(u[j] * (c[j] - sum(w[i] * y[i,j] for i in 1:15))
            for j in 1:3)
    )

    optimize!(m)

    if termination_status(m) != MOI.OPTIMAL
        error("Optimization was not successful: ", termination_status(m))
    end

    # Save objective value
    bound = objective_value(m)
    push!(bounds, bound)

    # Step 3: Calculate used capacity and subgradients
    used_capacity = [
    sum(w[i] * value(y[i,j]) for i in 1:15)
    for j in 1:3 ]

    gamma = [
        c[j] - sum(w[i] * value(y[i,j]) for i in 1:15)
        for j in 1:3 ]

    # Save the best upper bound and its corresponding solution
    if bound < best_bound
        global best_bound = bound
        global best_iteration = k
        best_x .= value.(x)
        best_y .= value.(y)
        best_u .= u
        best_gamma .= gamma
        best_capacity .= used_capacity
    end

    push!(best_bounds, best_bound)

    # Step 4: Polyak-type step size
    denominator = sum(gamma.^2)

    if denominator > 0
        mu = epsilon * (bound - LB) / denominator
    else
        mu = 0.0
    end

    # Step 5: Update multipliers
    for j in 1:3
        u[j] = max(0.0, u[j] - mu * gamma[j])
    end

end

# Print solution corresponding to the best upper bound
println("Best Lagrangian upper bound: ", best_bound)
println("Found at iteration: ", best_iteration)
println("Best multipliers: ", best_u)

println("\nCapacity usage at best iteration :")
for j in 1:3
    println("Team ", j, ": ", best_capacity[j], " / ", c[j])
end

println("\nSubgradients at best iteration:")
for j in 1:3
    println("Team ", j, ": ", best_gamma[j])
end

println("\nSelected initiatives at best iteration:")

for i in 1:15
    if best_x[i] > 0.5
        for j in 1:3
            if best_y[i,j] > 0.5
                println(
                    "Initiative ", i,
                    " | Benefit: ", a[i],
                    " | Resources: ", w[i],
                    " | Team: ", j
                )
            end
        end
    end
end


#Plot the development of the upper bound through k iterations
p1 = plot(
    1:max_iterations, bounds,
    label="Lagrangian bound",
    xlabel="Iteration k",
    ylabel="Upper bound",
    linewidth=1.5,
    ylims=(240, 400)
)

plot!(
    p1, 1:max_iterations, best_bounds,
    label="Best upper bound",
    linewidth=2.5
)

hline!(
    p1, [254.8918918918919],
    label="LP relaxation",
    linestyle=:dash
)

display(p1)