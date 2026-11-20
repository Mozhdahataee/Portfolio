using JuMP, GLPK

model = Model(GLPK.Optimizer)

# Profit and weight matrices (groups 1 to 4, items 1 to 3)
P = [10 14 20; 
      8 15 18; 
      5 12 16; 
      7 11 19]

W = [ 2  3  5; 
      2  4  5; 
      1  3  4; 
      2  3  5]

# Defining the binary decision variables
@variable(model, x[1:4, 1:3], Bin)

# Defining the objective function
@objective(model, Max, sum(P[r, j] * x[r, j] for r in 1:4, j in 1:3))

# Defining the constraints
@constraint(model, capacity, sum(W[r, j] * x[r, j] for r in 1:4, j in 1:3) <= 13)
for r in 1:4
    @constraint(model, sum(x[r, j] for j in 1:3) == 1)
end

# Solving the model
optimize!(model)

# Printing the results
println("Objective: ", objective_value(model))
for r in 1:4
    for j in 1:3
        println("x[$r, $j] = ", value(x[r, j]))
    end
end