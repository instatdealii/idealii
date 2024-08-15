## Program to run all simulations needed to reproduce the results in the
#  example step. Note that this may run for a few hours!


# This funcion constructs the command line call we pass to system()
# The options are as defined in the main() function of step-3.cc
simulationcall <- function(n_ref_space, M, s, r, support_type) {
  num_mpi_cores <- 8
  res <- paste("mpirun -n ", num_mpi_cores, " build/step-3", sep = "")
  res <- paste(res, " --s=", s, sep = "")
  res <- paste(res, " --r=", r, sep = "")
  res <- paste(res, " --n-ref-space=", n_ref_space, sep = "")
  res <- paste(res, " --M=", M, sep = "")
  res <- paste(res, " --support-type=", support_type, sep = "")
  res <- paste(res, " --no-vtu")
}

# This function goes through the given vectors of temporal element numbers M and
# spatial refinement steps N together to run a simulation for each.
# The orders s and r as well as the support type of the finite element are
# also given as parameters.
# The suffix is added to the filename to append information.
runsim <- function(M, N, s, r, support_type, suffix) {
  n <- length(M)
  # allocate storage for the resulting errors and number of space-time DoFs
  errors <- rep(0., times = n)
  dofs <- rep(0, times = n)
  # Go through M (and N) and call each simulation
  for (i in 1:length(M)) {
    # Save the output of the system call to res
    res <- system(simulationcall(s = s, r = r, M = M[i], n_ref_space = N[i], support_type = support_type), intern = TRUE)
    # The number of space-time DoFs is output in the third to last line of each
    # simulation and the error is printed in the final line.
    dofs[i] <- as.integer(tail(res, 3)[1])
    errors[i] <- as.double(tail(res, 1))
    # Print the results so we have an idea on simulation progress
    print(paste("Type: ", support_type, ", M=", M[i], ", N=", N[i], " -> ", dofs[i], " ", errors[i], sep = ""))
  }
  # Combine the data to output, construct a filename and finally write a CSV
  df <- data.frame(M, N, dofs, errors)
  filesuffix <- paste(support_type, "_cG", s, "_dG", r, suffix, ".csv", sep = "")
  write.csv(df, paste("results/results_", filesuffix, sep = ""))
  0.
}

# The following three functions construct vectors of length n matching the
# refinement we want to simulate, i.e. temporal k-refinement,
# spatial h-refinement or spatio-temporal kh-refinement

k_refinement <- function(s, r, M_init, N_init, support_type, n) {
  N <- rep(N_init, times = n)
  M <- M_init * (2^seq(0, n - 1))
  runsim(s = s, r = r, support_type = support_type, M = M, N = N, suffix = "_k")
}

h_refinement <- function(s, r, M_init, N_init, support_type, n) {
  N <- seq(N_init, N_init + n - 1)
  M <- rep(M_init, times = n)
  runsim(s = s, r = r, support_type = support_type, M = M, N = N, suffix = "_h")
}
kh_refinement <- function(s, r, M_init, N_init, support_type, n) {
  N <- seq(N_init, N_init + n - 1)
  M <- M_init * (2^seq(0, n - 1))
  runsim(s = s, r = r, support_type = support_type, M = M, N = N, suffix = "_kh")
}


# Make sure we have a results folder to write into
system("mkdir -p results")

# Run all refinements with Gauss-Lobatto support points
st <- "Lobatto"
h_refinement(support_type = st, s = 1, r = 0, M_init = 1600, N_init = 4, n = 4)
h_refinement(support_type = st, s = 1, r = 1, M_init = 400, N_init = 4, n = 4)
h_refinement(support_type = st, s = 1, r = 2, M_init = 400, N_init = 4, n = 4)
h_refinement(support_type = st, s = 2, r = 2, M_init = 200, N_init = 4, n = 4)

k_refinement(support_type = st, s = 1, r = 0, M_init = 10, N_init = 7, n = 7)
k_refinement(support_type = st, s = 1, r = 1, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 1, r = 2, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 2, r = 2, M_init = 10, N_init = 6, n = 5)

kh_refinement(support_type = st, s = 1, r = 0, M_init = 200, N_init = 4, n = 4)
kh_refinement(support_type = st, s = 1, r = 1, M_init = 50, N_init = 4, n = 4)
kh_refinement(support_type = st, s = 1, r = 2, M_init = 50, N_init = 4, n = 4)
kh_refinement(support_type = st, s = 2, r = 2, M_init = 25, N_init = 4, n = 4)

# Run k-refinement with all other types of support points too
st <- "Legendre"
k_refinement(support_type = st, s = 1, r = 1, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 1, r = 2, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 2, r = 2, M_init = 10, N_init = 6, n = 5)

st <- "RadauRight"
k_refinement(support_type = st, s = 1, r = 1, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 1, r = 2, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 2, r = 2, M_init = 10, N_init = 6, n = 5)

st <- "RadauLeft"
k_refinement(support_type = st, s = 1, r = 1, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 1, r = 2, M_init = 10, N_init = 7, n = 6)
k_refinement(support_type = st, s = 2, r = 2, M_init = 10, N_init = 6, n = 5)
