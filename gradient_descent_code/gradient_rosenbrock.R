
#answer5
# Custom Rosenbrock function
rb_fn <- function(theta) {
  term1 <- (1 - theta[1])^2
  term2 <- 100 * (theta[2] - theta[1]^2)^2
  return(term1 + term2)
}

# Gradient of Rosenbrock
rb_grad <- function(theta) {
  g1 <- -2 * (1 - theta[1]) - 400 * theta[1] * (theta[2] - theta[1]^2)
  g2 <- 200 * (theta[2] - theta[1]^2)
  return(c(g1, g2))
}

# Gradient descent loop
rosen_gd <- function(init_theta, lr = 0.0015, tol = 1e-6) {
  theta <- init_theta
  prev_loss <- rb_fn(theta)
  loss_trend <- c(prev_loss)
  steps <- 0
  
  repeat {
    grad_vec <- rb_grad(theta)
    theta_new <- theta - lr * grad_vec
    current_loss <- rb_fn(theta_new)
    loss_trend <- c(loss_trend, current_loss)
    steps <- steps + 1
    
    if (abs(current_loss - prev_loss) < tol || steps > 10000) break
    
    theta <- theta_new
    prev_loss <- current_loss
  }
  
  return(list(opt_theta = theta, iter = steps, loss_curve = loss_trend))
}

# Run the algorithm
output <- rosen_gd(c(-1, 1), lr = 0.0015, tol = 1e-6)

# Display results
cat("Optimal solution:", round(output$opt_theta, 5), "\n")
cat("Total iterations:", output$iter, "\n")

# Plotting
plot(output$loss_curve, type = "l", col = "darkorange", lwd = 2,
     xlab = "Iteration", ylab = "Loss",
     main = "Rosenbrock Loss Curve (Custom GD)")
abline(h = min(output$loss_curve), col = "steelblue", lty = 2)
legend("topright", legend = c("Loss", "Minimum Loss"),
       col = c("darkorange", "steelblue"), lty = c(1, 2), lwd = 2)
