## Programm to run all simulations needed to reproduce the results in the
#  example step. Note that this might run for an hour or two.

# This function constructs the command line call and uses system()
# to execute it.
# The options are as defined in the main() function of step-4.cc.
runsim <- function(r = 0, do_output = FALSE) {
  num_mpi_cores <- 8
  call <- paste("mpirun -n ", num_mpi_cores, " build/step-4", sep = "")
  call <- paste(call, " --r=", r, sep = "")
  if (do_output) {
    call <- paste(call, " --write-vtu")
  } else {
    call <- paste(call, " --no-vtu")
  }
  system(call)
}

# Run the simulation for dG(0) to dG(2) temporal discretizations
# Only do the VTU ouptut of the velocity and pressure fields for dG(0)
runsim(r = 0, do_output = TRUE)
runsim(r = 1, do_output = FALSE)
runsim(r = 2, do_output = FALSE)
