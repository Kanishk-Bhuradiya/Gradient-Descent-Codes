set.seed(123)

# Parameters
n <- 100
eta <- 0.008   # Slightly slower step size
epsilon <- 1e-6
max_iter <- 1000

# Generate data
x_raw <- rnorm(n, mean = 1, sd = sqrt(2))
x_scaled <- scale(x_raw)
X1 <- rep(1, n)
X2 <- as.numeric(x_scaled)
y <- rnorm(n, mean = 2 + 3 * x_raw, sd = sqrt(5))

# Initialize parameters
beta0 <- 0
beta1 <- 0
loss_history <- c()

# Gradient Descent Loop (unvectorized)
for (iter in 1:max_iter) {
  grad0 <- 0
  grad1 <- 0
  loss <- 0
  
  for (i in 1:n) {
    xi1 <- X1[i]
    xi2 <- X2[i]
    yi <- y[i]
    pred <- beta0 * xi1 + beta1 * xi2
    error <- yi - pred
    loss <- loss + (error^2) / (2 * n)
    grad0 <- grad0 - error * xi1
    grad1 <- grad1 - error * xi2
  }
  
  grad0 <- grad0 / n
  grad1 <- grad1 / n
  
  # Update
  beta0 <- beta0 - eta * grad0
  beta1 <- beta1 - eta * grad1
  
  loss_history <- c(loss_history, loss)
  
  # Stopping criterion
  if (sqrt(grad0^2 + grad1^2) < epsilon) break
}

# Convert to original scale
beta1_orig <- beta1 / sd(x_raw)
beta0_orig <- beta0 - beta1 * mean(x_raw) / sd(x_raw)

# Output
cat("Final beta (scaled):", c(beta0, beta1), "\n")
cat("Final beta (original scale):", c(beta0_orig, beta1_orig), "\n")
cat("Iterations:", iter, "\n")

# Plot: Loss vs Iteration with abline and color
plot(loss_history, type = "l", col = "darkorange", lwd = 2,
     xlab = "Iteration", ylab = "Loss",
     main = "Unvectorized Gradient Descent (Linear Regression)")
abline(h = min(loss_history), col = "steelblue", lty = 2)
legend("topright", legend = c("Loss", "Convergence limit"),
       col = c("darkorange", "steelblue"), lwd = 2, lty = c(1, 2))
