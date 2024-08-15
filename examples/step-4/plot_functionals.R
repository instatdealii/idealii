## Plot the curves for the different functionals over time
#  Optionally, this compares the data to results from the software FEATFLOW
#  https://wwwold.mathematik.tu-dortmund.de/~featflow/en/software/featflow.html
#
library("ggplot2")
library("RColorBrewer")
library("grDevices")

w <- 9
h <- 5

# To compare with the featflow results you can download and unpack the files
# pressure_q2_cn_lv1-6_dt4.zip and draglift_q2_cn_lv1-6_dt4.zip
# from the FEATFLOW 2D-3 benchmark website
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html
# If downloaded, specify the folders the zip files have been extracted into
# Otherwise leave as empty string and only the ideal.II results will be plotted.
feat_draglift_path <- ""
feat_pressure_path <- ""
# Example:
# feat_draglift_path <- "/home/<user>/Downloads/draglift_q2_cn_lv1-6_dt4"
# feat_pressure_path <- "/home/<user>/Downloads/pressure_q2_cn_lv1-6_dt4"


# If a proper path was specified and read in the FEATFLOW data
if (feat_draglift_path != "") {
  # Spatial refinement level 2
  feat2_path <- paste(feat_draglift_path, "/bdforces_lv2", sep = "")
  feat2_forces <- # we ignore the first comment line
    read.csv(feat2_path, header = FALSE, sep = " ", comment.char = "#")
  # Without a header the columns are numbered
  feat2_drag <- feat2_forces$V4 # horiz
  feat2_lift <- feat2_forces$V5 # vert
  feat2_dltime <- feat2_forces$V2 # time

  # Do the same for spatial refinement level 6
  feat6_path <- paste(feat_draglift_path, "/bdforces_lv6", sep = "")
  feat6_forces <-
    read.csv(feat6_path, header = FALSE, sep = " ", comment.char = "#")
  feat6_drag <- feat6_forces$V4 # horiz
  feat6_lift <- feat6_forces$V5 # vert
  feat6_dltime <- feat6_forces$V2 # time
} else {
  # No input specified, set this data to NULL so we don't plot anything for
  # FEATFLOW
  feat2_drag <- NULL
  feat2_lift <- NULL
  feat2_dltime <- NULL

  feat6_drag <- NULL
  feat6_lift <- NULL
  feat6_dltime <- NULL
}


# Repeat the read in for the point data (pressure)
if (feat_pressure_path != "") {
  feat2_path <- paste(feat_pressure_path, "/pointvalues_lv2", sep = "")
  feat2_pressure <-
    read.csv(feat2_path, header = FALSE, sep = " ", comment.char = "#")
  feat2_front <- feat2_pressure$V7 # value
  feat2_back <- feat2_pressure$V12 # value.1
  # The pressure difference is not output by FEATFLOW so we calculate it here
  feat2_pdiff <- feat2_front - feat2_back
  feat2_ptime <- feat2_pressure$V2 # time

  feat6_path <- paste(feat_pressure_path, "/pointvalues_lv6", sep = "")
  feat6_pressure <-
    read.csv(feat6_path, header = FALSE, sep = " ", comment.char = "#")
  feat6_front <- feat6_pressure$V7 # value
  feat6_back <- feat6_pressure$V12 # value.1
  feat6_pdiff <- feat6_front - feat6_back
  feat6_ptime <- feat6_pressure$V2 # time
} else {
  feat2_ptime <- NULL
  feat2_pdiff <- NULL
  feat6_ptime <- NULL
  feat6_pdiff <- NULL
}

# Read in the ideal.II results for the different temporal elements
idealiiDG0 <- read.csv("functional_log_dG0_M256_lvl2.csv", header = TRUE)
idealiiDG0_diff <- idealiiDG0$pdiff
idealiiDG0_drag <- idealiiDG0$drag
idealiiDG0_lift <- idealiiDG0$lift
idealiiDG0_time <- idealiiDG0$t

idealiiDG1 <- read.csv("functional_log_dG1_M256_lvl2.csv", header = TRUE)
idealiiDG1_diff <- idealiiDG1$pdiff
idealiiDG1_drag <- idealiiDG1$drag
idealiiDG1_lift <- idealiiDG1$lift
idealiiDG1_time <- idealiiDG1$t

idealiiDG2 <- read.csv("functional_log_dG2_M256_lvl2.csv", header = TRUE)
idealiiDG2_diff <- idealiiDG2$pdiff
idealiiDG2_drag <- idealiiDG2$drag
idealiiDG2_lift <- idealiiDG2$lift
idealiiDG2_time <- idealiiDG2$t

# Construct a data frame with named columns for plotting drag and lift curves.
# The column code is for grouping results with the same discretization together.
dragliftData <- data.frame(
  t = c(
    feat2_dltime, feat6_dltime,
    idealiiDG0_time, idealiiDG1_time, idealiiDG2_time
  ),
  drag = c(
    feat2_drag, feat6_drag,
    idealiiDG0_drag, idealiiDG1_drag, idealiiDG2_drag
  ),
  lift = c(
    feat2_lift, feat6_lift,
    idealiiDG0_lift, idealiiDG1_lift, idealiiDG2_lift
  ),
  code = c(
    rep("FEATFLOW (lvl2)", time = length(feat2_drag)),
    rep("FEATFLOW (lvl6)", time = length(feat6_drag)),
    rep("ideal.II dG0 (lvl2)", time = length(idealiiDG0_drag)),
    rep("ideal.II dG1 (lvl2)", time = length(idealiiDG1_drag)),
    rep("ideal.II dG2 (lvl2)", time = length(idealiiDG2_drag))
  )
)

# Start building a plot by constructing a plot object with the basic mapping
plt_drag <- ggplot(dragliftData, mapping = aes(x = t, y = drag))
# Add a point cloud to better distinguish between the different results
plt_drag <- plt_drag + geom_point(mapping = aes(colour = code), size = 0.25)
# Add axes labels, styling and a different colorbar
plt_drag <- plt_drag + labs(x = "time", y = "drag")
plt_drag <- plt_drag + theme_bw()
plt_drag <- plt_drag + scale_colour_brewer(palette = "Set1")

# Do the same for the lift values
plt_lift <- ggplot(dragliftData, mapping = aes(x = t, y = lift))
plt_lift <- plt_lift + geom_point(mapping = aes(colour = code), size = 0.25)
plt_lift <- plt_lift + labs(x = "time", y = "lift")
plt_lift <- plt_lift + theme_bw()
plt_lift <- plt_lift + scale_colour_brewer(palette = "Set1")


# Construct another data frame for the pressure data
pressureData <- data.frame(
  t = c(
    feat2_ptime, feat6_ptime,
    idealiiDG0_time, idealiiDG1_time, idealiiDG2_time
  ),
  p = c(
    feat2_pdiff, feat6_pdiff,
    idealiiDG0_diff, idealiiDG1_diff, idealiiDG2_diff
  ),
  code = c(
    rep("FEATFLOW (lvl2)", time = length(feat2_pdiff)),
    rep("FEATFLOW (lvl6)", time = length(feat6_pdiff)),
    rep("ideal.II dG0 (lvl2)", time = length(idealiiDG0_diff)),
    rep("ideal.II dG1 (lvl2)", time = length(idealiiDG1_diff)),
    rep("ideal.II dG2 (lvl2)", time = length(idealiiDG2_diff))
  )
)

# Build a plot for the pressure as before
plt_press <- ggplot(pressureData, mapping = aes(x = t, y = p))
plt_press <- plt_press + geom_point(mapping = aes(colour = code), size = 0.25)
plt_press <- plt_press + labs(x = "time", y = "pressure difference")
plt_press <- plt_press + theme_bw()
plt_press <- plt_press + scale_colour_brewer(palette = "Set1")


# Open a PDF file depending on whether we have the FEATFLOW data or not
if (feat_draglift_path != "") {
  pdf("NSE_results_idealii_vs_FEATFLOW.pdf", width = w, height = h)
} else {
  pdf("NSE_results_idealii.pdf", width = w, height = h)
}
# Add the plots to the PDF
print(plt_drag)
print(plt_lift)
print(plt_press)

dev.off() #close the PDF file
