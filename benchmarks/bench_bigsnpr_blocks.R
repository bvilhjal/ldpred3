#!/usr/bin/env Rscript
# bigsnpr side, block-diagonal LD -- built the way bigsnpr is meant to be used:
# the SFBM is assembled INCREMENTALLY block-by-block with $add_columns(), so the
# full block-diagonal correlation never sits in RAM (only one k x k block at a
# time). This mirrors the LDpred2 vignette's per-chromosome loop and is the fair
# memory comparison against pyLDpred2's streaming sampler.
suppressMessages({library(bigsnpr); library(Matrix); library(bigsparser)})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(1); RhpcBLASctl::omp_set_num_threads(1)
}
options(bigstatsr.check.parallel.blas = FALSE)

args <- commandArgs(trailingOnly = TRUE)
h2 <- as.numeric(args[1]); p <- as.numeric(args[2])
burn_in <- as.integer(args[3]); num_iter <- as.integer(args[4]); workdir <- args[5]
# Optional auto initialisation (args 6-7); default = oracle (h2, p).
h2_init <- if (length(args) >= 6) as.numeric(args[6]) else h2
p_init  <- if (length(args) >= 7) as.numeric(args[7]) else p

sizes <- scan(file.path(workdir, "sizes.txt"), quiet = TRUE)
sfbm_file <- file.path(workdir, "corr_sfbm")
if (file.exists(paste0(sfbm_file, ".sbk"))) file.remove(paste0(sfbm_file, ".sbk"))

# Stream blocks from disk one at a time, appending each as a diagonal block.
con <- file(file.path(workdir, "blocks.bin"), "rb")
sfbm <- NULL
for (k in sizes) {
  M <- matrix(readBin(con, "double", n = k * k), k, k)
  block <- as(as(M, "CsparseMatrix"), "generalMatrix")
  if (is.null(sfbm)) {
    sfbm <- as_SFBM(block, backingfile = sfbm_file, compact = TRUE)
  } else {
    sfbm$add_columns(block, nrow(sfbm))   # shift row indices by current nrow
  }
}
close(con)
cat(sprintf("SFBM ncol %d  nval %d\n", ncol(sfbm), length(sfbm$p) - 1))

df_beta <- read.csv(file.path(workdir, "df_beta.csv"))

t_inf <- system.time(b_inf <- snp_ldpred2_inf(sfbm, df_beta, h2))[["elapsed"]]
gp <- data.frame(p = p, h2 = h2, sparse = FALSE)
t_grid <- system.time(
  b_grid <- snp_ldpred2_grid(sfbm, df_beta, gp, burn_in = burn_in,
                             num_iter = num_iter, ncores = 1))[["elapsed"]]
b_grid <- as.vector(b_grid[, 1])
t_auto <- system.time(
  res <- snp_ldpred2_auto(sfbm, df_beta, h2_init = h2_init, vec_p_init = p_init,
                          burn_in = burn_in, num_iter = num_iter, ncores = 1))[["elapsed"]]
b_auto <- res[[1]]$beta_est

write.csv(data.frame(inf = b_inf, grid = b_grid, auto = b_auto),
          file.path(workdir, "r_betas.csv"), row.names = FALSE)
cat(sprintf("TIME inf %.4f\n", t_inf))
cat(sprintf("TIME grid %.4f\n", t_grid))
cat(sprintf("TIME auto %.4f\n", t_auto))
