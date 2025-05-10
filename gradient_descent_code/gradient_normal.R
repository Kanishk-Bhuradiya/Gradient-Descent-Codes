# Define A and b
A <- matrix(c(2, 0, 0, 4), nrow = 2, byrow = TRUE)
b <- c(-4, -8)

# Objective function
f <- function(x) {
  t(x) %*% A %*% x + sum(b * x)
}

# Gradient
grad <- function(x) {
  2 * A %*% x + b
}

# Parameters
x <- c(1, 1)
eta <- 0.1
epsilon <- 1e-6
max_iter <- 10000
loss_history <- c()

# Gradient Descent
for (i in 1:max_iter) {
  g <- grad(x)
  loss_history <- c(loss_history, f(x))
  if (sqrt(sum(g^2)) < epsilon) break
  x <- x - eta * g
}

# Output
cat("Final x:", x, "\n")
cat("Iterations:", i, "\n")

# Plot
plot(loss_history, type = "l", xlab = "Iteration", ylab = "Loss", main = "Quadratic Function Convergence")
