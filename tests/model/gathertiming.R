library(acepack)


# Reads the file produced by the Python translation.
# The Python is necessary because Python code saved the JSON
# in a specific format that rjson can't read.
timing.data <- function(timings.file) {
  df <- read.table(timings.file, header=TRUE, sep=",")
  for (make.cat in categorical.x) {
    df[[make.cat]] <- as.integer(df[[make.cat]])
  }
  df
}

timings.file <- "/home/adolgert/dev/cascade/tests/model/timings.csv"

possible.x <- c(
  "data_cnt", "data_extent_cohort", "data_extent_primary",
  "data_point_cohort", "data_point_primary",
  "effect", "derivative_test_fixed", "iterations",
  "n_children", "quasi_fixed", "random_effect_points",
  "rate_count", "smooth_count", "topology_choice",
  "variables", "zero_sum_random"
)

categorical.x <- c(
  "derivative_test_fixed", "effect", "quasi_fixed", 
  "topology_choice", "zero_sum_random"
)

possible.y <- c(
  "max_rss_kb", "user"
)


# df is the dataframe with all variables
# x.variables is a list of string variable names
# y is the name of the y variable
# returns the rsquare.
compare.n <- function(df, x.variables, y.variable) {
  categorical <- numeric(0)
  for (cat.idx in 1:length(x.variables)) {
    is.categorical <- x.variables[cat.idx]
    if (is.categorical %in% categorical.x) {
      categorical <- c(categorical, cat.idx)
    }
  }
  # cat("the x", x.variables, "the y", y.variable, "cat", categorical, "\n")
  ace.out <- ace(df[x.variables], df[[y.variable]],
                 cat=categorical)
  ace.out$rsq
}


# Show the best r-squared values from chosen combinations.
show.best <- function(rsq, combinations) {
  comb.cnt <- dim(combinations)[2]
  reorder <- order(rsq, 1:comb.cnt, decreasing=TRUE)
  show.cnt <- min(length(rsq), 10)
  for (show.idx in 1:show.cnt) {
    comb.idx = reorder[show.idx]
    cat(rsq[comb.idx], combinations[, comb.idx], "\n")
  }
}


# Look at all combinations of x values to see which
# combinations maximize r-squared.
check.through.level <- function(df, level=2) {
  for (x.count in 1:level) {
    combinations <- combn(possible.x, x.count)
    comb.cnt <- dim(combinations)[2]
    resource <- list(max_rss_kb=array(0, comb.cnt),
                     user=array(0, comb.cnt))
    for (comb.idx in 1:comb.cnt) {
      for (y.variable in possible.y) {
        rsq <- compare.n(df, combinations[,comb.idx], y.variable)
        resource[[y.variable]][comb.idx] <- rsq
      }
    }
    
    for (show.y in possible.y) {
      cat("Best values for level", x.count, "resource", show.y, "\n")
      show.best(resource[[show.y]], combinations)
    }
  }
}
