# Load data
data <- read.csv("C:/Users/kanis/Downloads/logistic_sheet.csv.csv")

# Inputs
x1 <- data$x1
x2 <- data$x2
y <- as.numeric(data$y)  # Assuming y ∈ {0, 1}

# Parameters
n <- length(y)
beta0 <- 0  # Intercept
beta1 <- 0  # Coefficient for x1
beta2 <- 0  # Coefficient for x2
eta <- 0.05
threshold <- 1e-5
max_iter <- 10000
loss_history <- c()

# Sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Training loop (Unvectorized GD)
for (iter in 1:max_iter) {
  grad0 <- 0
  grad1 <- 0
  grad2 <- 0
  loss <- 0
  
  for (i in 1:n) {
    xi1 <- x1[i]
    xi2 <- x2[i]
    yi <- y[i]
    
    z <- beta0 + beta1 * xi1 + beta2 * xi2
    p <- sigmoid(z)
    
    # Accumulate gradients
    grad0 <- grad0 + (p - yi) * 1
    grad1 <- grad1 + (p - yi) * xi1
    grad2 <- grad2 + (p - yi) * xi2
    
    # Accumulate loss
    loss <- loss - (yi * log(p + 1e-10) + (1 - yi) * log(1 - p + 1e-10))
  }
  
  # Average gradients
  grad0 <- grad0 / n
  grad1 <- grad1 / n
  grad2 <- grad2 / n
  loss <- loss / n
  loss_history <- c(loss_history, loss)
  
  # Update parameters
  beta0_new <- beta0 - eta * grad0
  beta1_new <- beta1 - eta * grad1
  beta2_new <- beta2 - eta * grad2
  
  # Check convergence
  if (sqrt((beta0_new - beta0)^2 + (beta1_new - beta1)^2 + (beta2_new - beta2)^2) < threshold) {
    break
  }
  
  # Update parameters
  beta0 <- beta0_new
  beta1 <- beta1_new
  beta2 <- beta2_new
}

# Output
cat("Final Coefficients:\n")
cat("Intercept:", round(beta0, 4), "\n")
cat("Beta x1  :", round(beta1, 4), "\n")
cat("Beta x2  :", round(beta2, 4), "\n")
cat("Iterations:", iter, "\n")

# Plot convergence
plot(loss_history, type = "l", col = "purple", lwd = 2,
     xlab = "Iteration", ylab = "Log Loss",
     main = "Unvectorized Logistic Regression Convergence")
abline(h = min(loss_history), col = "blue", lty = 2)
legend("topright", legend = c("Loss", "Convergence Limit"),
       col = c("purple", "blue"), lty = c(1, 2), lwd = 2)
