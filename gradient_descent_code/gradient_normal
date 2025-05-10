# Load data from local CSV
data <- read.csv("C:/Users/kanis/Downloads/synthetic data - problem-4.csv")  # Update path as needed
x <- data$x
n <- length(x)

# Negative log-likelihood function
nll <- function(params) {
  mu <- params[1]
  sigma <- params[2]
  if (sigma <= 0) return(Inf)  # prevent invalid σ
  sum(log(sigma) + (x - mu)^2 / (2 * sigma^2))
}

# Gradient of NLL
nll_grad <- function(params) {
  mu <- params[1]
  sigma <- params[2]
  if (sigma <= 0) return(c(NA, NA))
  
  grad_mu <- sum(mu - x) / (sigma^2)
  grad_sigma <- sum(1 / sigma - ((x - mu)^2) / (sigma^3))
  return(c(grad_mu, grad_sigma))
}

# Gradient descent function
gd_normal_mle <- function(start_params, lr = 0.01, threshold = 1e-6, max_iter = 10000) {
  theta <- start_params
  loss_history <- c(nll(theta))
  iter <- 0
  
  repeat {
    grad <- nll_grad(theta)
    if (any(is.na(grad))) break
    
    theta_new <- theta - lr * grad
    theta_new[2] <- max(theta_new[2], 1e-4)  # enforce σ > 0
    
    loss_new <- nll(theta_new)
    loss_history <- c(loss_history, loss_new)
    
    iter <- iter + 1
    if (abs(loss_new - loss_history[iter]) < threshold || iter >= max_iter) break
    
    theta <- theta_new
  }
  
  return(list(mu = theta[1], sigma = theta[2], loss = loss_history, iterations = iter))
}

# Run optimization
result <- gd_normal_mle(c(0, 1), lr = 0.01)

# Output results
cat("Estimated mu   :", round(result$mu, 4), "\n")
cat("Estimated sigma:", round(result$sigma, 4), "\n")
cat("Iterations     :", result$iterations, "\n")

# Plot loss curve
plot(result$loss, type = "l", col = "darkred", lwd = 2,
     xlab = "Iteration", ylab = "Negative Log-Likelihood",
     main = "MLE for Normal Distribution (Gradient Descent)")
abline(h = min(result$loss), col = "blue", lty = 2)
legend("topright", legend = c("Loss", "Minimum Loss"),
       col = c("darkred", "blue"), lty = c(1, 2), lwd = 2)


