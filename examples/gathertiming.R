library(acepack)


modify.incoming.data <- function(df) {
  df[["data_cohort"]] <- df[["data_extent_cohort"]] + df[["data_point_cohort"]]
  df[["data_expensive"]] <- df[["prevalence"]] + df[["relrisk"]] + df[["Tincidence"]]
      + df[["withC"]] + df[["susceptible"]]
  # only larger data
  df.large<-df[df$max_rss_kb > 5e+4,]
  # fit both, not fit fixed or fit random
  only.both <- df.large[df.large$effect == 1,]
  only.both
}


# Reads the file produced by the Python translation.
# The Python is necessary because Python code saved the JSON
# in a specific format that rjson can't read.
timing.data <- function(timings.file) {
  df <- read.table(timings.file, header=TRUE, sep=",")
  for (make.cat in categorical.x) {
    df[[make.cat]] <- as.integer(df[[make.cat]])
  }
  modify.incoming.data(df)
}

timings.file <- "/home/adolgert/dev/cascade/tests/model/timings.csv"

possible.x <- c(
  "data_cnt", "data_extent_cohort", "data_extent_primary",
  "data_point_cohort", "data_point_primary",
  "effect", "derivative_test_fixed", "iterations",
  "n_children", "quasi_fixed", "random_effect_points",
  "rate_count", "smooth_count", "topology_choice",
  "variables", "zero_sum_random", "Sincidence",
  "Tincidence", "mtall", "mtexcess", "mtother",
  "mtspecific", "mtstandard", "prevalence",
  "relrisk", "remission", "susceptible", "withC",
  "data_cohort", "data_expensive"
)

categorical.x <- c(
  "derivative_test_fixed", "effect", "quasi_fixed", 
  "topology_choice", "zero_sum_random"
)

possible.y <- c("max_rss_kb", "user")


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
  ace(df[x.variables], df[[y.variable]],cat=categorical)
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


# Use the top choices from the last level for
# the next level.
top.choices <- function(rsq, combinations, keep=100) {
  comb.cnt <- dim(combinations)[2]
  reorder <- order(rsq, 1:comb.cnt, decreasing=TRUE)
  top.cnt <- min(length(rsq), keep)
  top.x <- numeric(0)
  for (top.idx in 1:top.cnt) {
    comb.idx = reorder[top.idx]
    top.x <- union(top.x, combinations[, comb.idx])
  }
  top.x
}


# Look at all combinations of x values to see which
# combinations maximize r-squared.
check.through.level <- function(df, level=2) {
  for (y.variable in possible.y) {
    check.through.level.single(df, y.variable, level)
  }
}


check.through.level.single <- function(df, y.variable, level=2) {
  allowed.x <- possible.x
  for (x.count in 1:level) {
    if (x.count > length(allowed.x)) break
    combinations <- combn(allowed.x, x.count)
    comb.cnt <- dim(combinations)[2]
    resource.rsq <- array(0, comb.cnt)
    for (comb.idx in 1:comb.cnt) {
      rsq <- compare.n(df, combinations[,comb.idx], y.variable)$rsq
      resource.rsq[comb.idx] <- rsq
    }
    # Limit the allowed x for the next round.
    allowed.x <- top.choices(resource.rsq, combinations)
    
    cat("===== level", x.count, "resource", y.variable, "\n")
    show.best(resource.rsq, combinations)
    if (comb.cnt == 1) break
  }
}


run <- function(levels=2) {
  df <- timing.data("timings.csv")
  check.through.level(df, levels)
}
