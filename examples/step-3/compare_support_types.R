## Read in the results for the different support points
#  and compare the errors.
#  The output filenams from run_simulations are all identical
#  apart from the support type string in between.
#  Therefore, we construct both a filename prefix and suffix
prefix <- "results/results"

# The finite element combinations we have calculated
fes <- c("cG1_dG1", "cG1_dG2", "cG2_dG2")

# Go over each finite element combination and compare
for (i in seq(1, length(fes))) {
  fe <- fes[i]

  suffix <- paste(fe, "_k.csv", sep = "")

  # Read in each support point type
  Lobatto <- read.csv(paste(prefix, "Lobatto", suffix, sep = "_"))
  Legendre <- read.csv(paste(prefix, "Legendre", suffix, sep = "_"))
  RadauLeft <- read.csv(paste(prefix, "RadauLeft", suffix, sep = "_"))
  RadauRight <- read.csv(paste(prefix, "RadauRight", suffix, sep = "_"))

  # Since the meshes and finite elements are the same all support points 
  # have the same number of DoFs, so we need to read them in just once
  dofs <- Lobatto$dofs

  # Calculate the quotient between each support point and Lobatto 
  percentages_Leg <- 100 * Legendre$errors / Lobatto$errors
  percentages_RL <- 100 * RadauLeft$errors / Lobatto$errors
  percentages_RR <- 100 * RadauRight$errors / Lobatto$errors

  df <- data.frame(
    dofs,
    percentages_RL,
    percentages_Leg,
    percentages_RR
  )
  colnames(df) <- c("DoFs", "RadauLeft", "Legendre", "RadauRight")
  print(fe)
  print(df)
}
