## Read in the Lobatto support point results from run_simulations
#  and plot the convergence curves.

library("ggplot2") # Plotting
library("RColorBrewer") # Color scale
library("grDevices") # Output to pdf
library("latex2exp") # Use LaTeX for labels

# width and height of each pdf page
w <- 11
h <- 5
# Labels for x and y axis
xlab <- "#DoFs space-time"
ylab <- TeX(r"($L^2$ error)")

# The steps for constructing a plot are the same for each of the refinements
# 1. Read in the data
# 2. Construct a data frame for plotting
#    This consists of named columns that can be accessed by name later.
#    Therefore, we give the vectors for x and y axis, i.e. DoFs and L2 errors
#    as well as a vector FE for grouping the inputs by the type of finite
#    read in data from spatial h-refinement
# 3. Build a plot, which consist of a few steps again
#  a. Construct a plot object containing the data and basic mapping of axes
#  b. Add a line plot with colour and linetype based on FE
#  c. Add labels to the axes
#  d. Set both axes to logarithmic scale
#  e. Change the style by choosing a theme and colourbar


# h-refinement step 1
h_cG1dG0 <- read.csv("results/results_Lobatto_cG1_dG0_h.csv")
h_cG1dG1 <- read.csv("results/results_Lobatto_cG1_dG1_h.csv")
h_cG1dG2 <- read.csv("results/results_Lobatto_cG1_dG2_h.csv")
h_cG2dG2 <- read.csv("results/results_Lobatto_cG2_dG2_h.csv")

# h-refinement step 2
lineData_h <- data.frame(
  dofs = c(h_cG1dG0$dofs, h_cG1dG1$dofs, h_cG1dG2$dofs, h_cG2dG2$dofs),
  err = c(h_cG1dG0$errors, h_cG1dG1$errors, h_cG1dG2$errors, h_cG2dG2$errors),
  FE = c(
    rep(paste("dG(0)xQ1\nM =", h_cG1dG0$M[1]), times = length(h_cG1dG0$N)),
    rep(paste("dG(1)xQ1\nM =", h_cG1dG1$M[1]), times = length(h_cG1dG1$N)),
    rep(paste("dG(2)xQ1\nM =", h_cG1dG2$M[1]), times = length(h_cG1dG2$N)),
    rep(paste("dG(2)xQ2\nM =", h_cG2dG2$M[1]), times = length(h_cG2dG2$N))
  )
)

# h-refinement step 3
plt_h <- ggplot(lineData_h, mapping = aes(x = dofs, y = err))
plt_h <- plt_h + geom_line(mapping = aes(colour = FE, linetype = FE), linewidth = 1.5)
plt_h <- plt_h + labs(x = xlab, y = ylab)
plt_h <- plt_h + scale_x_log10()
plt_h <- plt_h + scale_y_log10()
plt_h <- plt_h + theme_bw()
plt_h <- plt_h + scale_colour_brewer(palette = "Set1")


# k-refinement step 1
k_cG1dG0 <- read.csv("results/results_Lobatto_cG1_dG0_k.csv")
k_cG1dG1 <- read.csv("results/results_Lobatto_cG1_dG1_k.csv")
k_cG1dG2 <- read.csv("results/results_Lobatto_cG1_dG2_k.csv")
k_cG2dG2 <- read.csv("results/results_Lobatto_cG2_dG2_k.csv")

# k-refinement step 2
lineData_k <- data.frame(
  dofs = c(k_cG1dG0$dofs, k_cG1dG1$dofs, k_cG1dG2$dofs, k_cG2dG2$dofs),
  err = c(k_cG1dG0$errors, k_cG1dG1$errors, k_cG1dG2$errors, k_cG2dG2$errors),
  FE = c(
    rep(paste("dG(0)xQ1\nh = 0.5^", k_cG1dG0$N[1], sep = ""), times = length(k_cG1dG0$M)),
    rep(paste("dG(1)xQ1\nh = 0.5^", k_cG1dG1$N[1], sep = ""), times = length(k_cG1dG1$M)),
    rep(paste("dG(2)xQ1\nh = 0.5^", k_cG1dG2$N[1], sep = ""), times = length(k_cG1dG2$M)),
    rep(paste("dG(2)xQ2\nh = 0.5^", k_cG2dG2$N[1], sep = ""), times = length(k_cG2dG2$M))
  )
)

# k-refinement step 3
plt_k <- ggplot(lineData_k, mapping = aes(x = dofs, y = err))
plt_k <- plt_k + geom_line(mapping = aes(colour = FE, linetype = FE), linewidth = 1.5)
plt_k <- plt_k + labs(x = xlab, y = ylab)
plt_k <- plt_k + scale_x_log10()
plt_k <- plt_k + scale_y_log10()
plt_k <- plt_k + theme_bw()
plt_k <- plt_k + scale_colour_brewer(palette = "Set1")


# kh-refinement step 1
kh_cG1dG0 <- read.csv("results/results_Lobatto_cG1_dG0_kh.csv")
kh_cG1dG1 <- read.csv("results/results_Lobatto_cG1_dG1_kh.csv")
kh_cG1dG2 <- read.csv("results/results_Lobatto_cG1_dG2_kh.csv")
kh_cG2dG2 <- read.csv("results/results_Lobatto_cG2_dG2_kh.csv")


# kh-refinement step 2
lineData_kh <- data.frame(
  dofs = c(kh_cG1dG0$dofs, kh_cG1dG1$dofs, kh_cG1dG2$dofs, kh_cG2dG2$dofs),
  err = c(kh_cG1dG0$errors, kh_cG1dG1$errors, kh_cG1dG2$errors, kh_cG2dG2$errors),
  FE = c(
    rep(paste("dG(0)xQ1\nM_init = ", kh_cG1dG0$M[1], sep = ""), times = length(kh_cG1dG0$M)),
    rep(paste("dG(1)xQ1\nM_init = ", kh_cG1dG1$M[1], sep = ""), times = length(kh_cG1dG1$M)),
    rep(paste("dG(2)xQ1\nM_init = ", kh_cG1dG2$M[1], sep = ""), times = length(kh_cG1dG2$M)),
    rep(paste("dG(2)xQ2\nM_init = ", kh_cG2dG2$M[1], sep = ""), times = length(kh_cG2dG2$M))
  )
)

# kh-refinement step 3
plt_kh <- ggplot(lineData_kh, mapping = aes(x = dofs, y = err))
plt_kh <- plt_kh + geom_line(mapping = aes(colour = FE, linetype = FE), linewidth = 1.5)
plt_kh <- plt_kh + labs(x = xlab, y = ylab)
plt_kh <- plt_kh + scale_x_log10()
plt_kh <- plt_kh + scale_y_log10()
plt_kh <- plt_kh + theme_bw()
plt_kh <- plt_kh + scale_colour_brewer(palette = "Set1")


# Finally, we open the PDF file for output, print all plots to it and close it.
pdf("convergence_idealii_step-3.pdf", width = w, height = h)
print(plt_h)
print(plt_k)
print(plt_kh)

dev.off()
