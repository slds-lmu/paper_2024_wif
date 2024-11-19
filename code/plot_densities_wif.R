# Load necessary libraries
library(ggplot2)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Parameters for the distributions
size1 <- 10  # Dispersion parameter for first distribution
prob1 <- 0.3 # Probability parameter for first distribution

size2 <- 15  # Dispersion parameter for second distribution
prob2 <- 0.5 # Probability parameter for second distribution

# Generate data
x_vals <- 0:40

data <- data.frame(
  X_p = rep(x_vals, 2),
  pmf = c(
    dnbinom(x_vals, size = size1, prob = prob1),  # PMF values for first distribution
    dnbinom(x_vals, size = size2, prob = prob2)   # PMF values for second distribution
  ),
  Distribution = rep(c("Non-White", "White"), each = length(x_vals))  # Labels for distributions
)

# Calculate the 10% and 90% quantiles for Non-White and White
quantile_10_dist2 <- qnbinom(0.10, size = size2, prob = prob2)
quantile_10_dist1 <- qnbinom(0.10, size = size1, prob = prob1)

quantile_90_dist2 <- qnbinom(0.90, size = size2, prob = prob2)
quantile_90_dist1 <- qnbinom(0.90, size = size1, prob = prob1)

# Add flag for bars based on quantile ranges for each distribution
data <- data %>%
  mutate(quantile_range = case_when(
    Distribution == "Non-White" & X_p <= quantile_10_dist1 ~ "Below 10%",
    Distribution == "Non-White" & X_p > quantile_10_dist1 & X_p <= quantile_90_dist1 ~ "Between 10% and 90%",
    Distribution == "Non-White" & X_p > quantile_90_dist1 ~ "Above 90%",
    Distribution == "White" & X_p <= quantile_10_dist2 ~ "Below 10%",
    Distribution == "White" & X_p > quantile_10_dist2 & X_p <= quantile_90_dist2 ~ "Between 10% and 90%",
    Distribution == "White" & X_p > quantile_90_dist2 ~ "Above 90%"
  ))

# Create the bar plot with quantile annotations
pmf_plot <- ggplot(data, aes(x = X_p, y = pmf, fill = quantile_range)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +  # Plot the PMF as bar plots
  labs(x = "X_P - Number of Priors", y = "Probability Mass Function p(X_P|C=c)") +  # Add axis labels
  theme_minimal() +
  theme(
    axis.title = element_text(size = 18),  # Increase size of axis titles
    axis.text = element_text(size = 18),  # Increase size of axis text labels
    strip.text = element_text(size = 18)  # Increase size of facet labels
  ) +  # Use a minimal theme for a clean look
  facet_wrap(~ Distribution, ncol = 1, scales = "free_y") +
  geom_vline(data = filter(data, Distribution == "Non-White"), aes(xintercept = quantile_10_dist1), linetype = "dashed", color = "black", linewidth = 1) +
  geom_vline(data = filter(data, Distribution == "Non-White"), aes(xintercept = quantile_90_dist1), linetype = "dashed", color = "black", linewidth = 1) +
  geom_text(data = filter(data, Distribution == "Non-White"), aes(x = quantile_10_dist1, y = max(pmf) * 0.8), label = "10% Quantile", color = "black", angle = 90, vjust = -0.5) +
  geom_text(data = filter(data, Distribution == "Non-White"), aes(x = quantile_90_dist1, y = max(pmf) * 0.6), label = "90% Quantile", color = "black", angle = 90, vjust = -0.5) +
  geom_vline(data = filter(data, Distribution == "White"), aes(xintercept = quantile_10_dist2), linetype = "dashed", color = "black", linewidth = 1) +
  geom_vline(data = filter(data, Distribution == "White"), aes(xintercept = quantile_90_dist2), linetype = "dashed", color = "black", linewidth = 1) +
  geom_text(data = filter(data, Distribution == "White"), aes(x = quantile_10_dist2, y = max(pmf) * 0.75), label = "10% Quantile", color = "black", angle = 90, vjust = -0.5) +
  geom_text(data = filter(data, Distribution == "White"), aes(x = quantile_90_dist2, y = max(pmf) * 0.65), label = "90% Quantile", color = "black", angle = 90, vjust = -0.5) +  # Create separate plots for each distribution, vertically stacked
  scale_fill_manual(values = c("Below 10%" = "#FF9999", "Between 10% and 90%" = "#FFD700", "Above 90%" = "#4CAF50")) +
  theme(legend.position = "none")  # Remove legend for clarity

# Save the plot as a PDF
pdf("plots/rpid_priors.pdf", width = 16, height = 8)
print(pmf_plot)
dev.off()

