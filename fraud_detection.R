# =============================================================================
# CSP-554: Credit Card Fraud Detection using SparkR on AWS EMR
# Dataset : Kaggle Credit Card Fraud Detection (train.csv)
# Models  : Logistic Regression · Random Forest (full + trimmed)
# Authors : Pragati Khekale · Mounika (et al.)
# =============================================================================

# ── 0. Install & Load Packages ────────────────────────────────────────────────
required_pkgs <- c("sparklyr", "readr", "dplyr", "ggplot2",
                   "randomForest", "caret", "MLmetrics", "Hmisc")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "http://cran.rstudio.com")
}
invisible(lapply(required_pkgs, install_if_missing))
invisible(lapply(required_pkgs, library, character.only = TRUE))


# ── 1. Spark Connection (AWS EMR / local) ─────────────────────────────────────
sc <- spark_connect(master = Sys.getenv("SPARK_MASTER", unset = "local"))
cat("Spark version:", spark_version(sc), "\n")


# ── 2. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH <- "data/train.csv"   # adjust if running on EMR HDFS

raw <- read_csv(DATA_PATH, show_col_types = FALSE)

# Class must be a factor for classification
raw$Class <- factor(raw$Class, levels = c(0, 1), labels = c("Legit", "Fraud"))

cat("Dataset dimensions:", nrow(raw), "rows x", ncol(raw), "cols\n")
cat("Class distribution:\n")
print(table(raw$Class))
cat("Fraud rate:", round(mean(raw$Class == "Fraud") * 100, 3), "%\n")


# ── 3. Write to HDFS as Parquet (Spark) ───────────────────────────────────────
sdf <- copy_to(sc, raw, "credit_card", overwrite = TRUE)
spark_write_parquet(sdf, path = "hdfs:///user/rstudio-user/credit_fraud",
                    mode = "overwrite")


# ── 4. Train / Validation / Test Split (70 / 15 / 15) ────────────────────────
set.seed(1234)
n      <- nrow(raw)
idx    <- sample(seq_len(n))
n_tr   <- floor(0.70 * n)
n_val  <- floor(0.15 * n)

df_train <- raw[idx[1:n_tr], ]
df_val   <- raw[idx[(n_tr + 1):(n_tr + n_val)], ]
df_test  <- raw[idx[(n_tr + n_val + 1):n], ]

cat("\nSplit sizes — Train:", nrow(df_train),
    "| Val:", nrow(df_val), "| Test:", nrow(df_test), "\n")

cat("Train class distribution:\n"); print(table(df_train$Class))
cat("Test  class distribution:\n"); print(table(df_test$Class))


# ── 5. Logistic Regression ────────────────────────────────────────────────────
lr_model <- glm(Class ~ ., data = df_train, family = binomial())
cat("\n--- Logistic Regression Summary ---\n")
print(summary(lr_model))

lr_probs <- predict(lr_model, newdata = df_test, type = "response")
lr_preds <- factor(ifelse(lr_probs > 0.5, "Fraud", "Legit"),
                   levels = c("Legit", "Fraud"))

cat("\nLogistic Regression — Confusion Matrix:\n")
print(confusionMatrix(lr_preds, df_test$Class, positive = "Fraud"))
cat("Logistic Regression F1 Score:",
    round(F1_Score(df_test$Class, lr_preds, positive = "Fraud"), 4), "\n")


# ── 6. Random Forest — Full Model (all variables) ─────────────────────────────
set.seed(1234)
rf_full <- randomForest(Class ~ ., data = df_train,
                        ntree = 500, importance = TRUE, keep.forest = TRUE)
cat("\n--- Random Forest (Full) ---\n")
print(rf_full)

df_test$pred_full <- predict(rf_full, newdata = df_test)
cm_full <- confusionMatrix(df_test$pred_full, df_test$Class, positive = "Fraud")
f1_full <- F1_Score(df_test$Class, df_test$pred_full, positive = "Fraud")
cat("Full RF — F1 Score:", round(f1_full, 4), "\n")
print(cm_full)

# Variable Importance Plot
png("output/varImpPlot.png", width = 800, height = 600)
varImpPlot(rf_full, sort = TRUE, n.var = 10,
           main = "Top 10 Most Important Variables")
dev.off()


# ── 7. Random Forest — Trimmed Models ────────────────────────────────────────
# Variables ordered by importance (MeanDecreaseGini)
top_vars  <- c("V17", "V12", "V14", "V10", "V16", "V11", "V9", "V4", "V18", "V26")
trim_sets <- list(
  trim1  = top_vars[1],
  trim2  = top_vars[1:2],
  trim3  = top_vars[1:3],
  trim4  = top_vars[1:4],
  trim5  = top_vars[1:5],
  trim10 = top_vars[1:10]
)

f1_results <- list()
for (nm in names(trim_sets)) {
  vars   <- trim_sets[[nm]]
  fml    <- as.formula(paste("Class ~", paste(vars, collapse = " + ")))
  set.seed(1234)
  rf_k   <- randomForest(fml, data = df_train, ntree = 500)
  preds  <- predict(rf_k, newdata = df_test)
  f1_k   <- F1_Score(df_test$Class, preds, positive = "Fraud")
  f1_results[[nm]] <- list(k = length(vars), f1 = f1_k)
  cat(sprintf("  %-8s  k=%2d  F1=%.4f\n", nm, length(vars), f1_k))
}

# Compile F1 performance table
perf_df <- data.frame(
  num_vars = c(sapply(f1_results, `[[`, "k"), ncol(df_train) - 1),
  f1_score = c(sapply(f1_results, `[[`, "f1"), f1_full),
  model    = c(names(f1_results), "full")
)
print(perf_df)

# Plot: F1 Score vs Number of Variables
p_f1 <- ggplot(perf_df, aes(x = num_vars, y = f1_score)) +
  geom_line(color = "#f0883e", linewidth = 1.2) +
  geom_point(color = "#ff7b72", size = 3.5) +
  geom_text(aes(label = round(f1_score, 3)), vjust = -0.8, size = 3.2) +
  labs(title = "Random Forest — F1 Score vs. Number of Variables",
       x = "Number of Variables", y = "F1 Score") +
  theme_minimal(base_size = 13)
ggsave("output/f1_vs_vars.png", p_f1, width = 7, height = 4.5)


# ── 8. Final Optimised Model (top-10 vars, 1000 trees) ───────────────────────
set.seed(1234)
rf_final <- randomForest(
  Class ~ V17 + V12 + V14 + V10 + V16 + V11 + V9 + V4 + V18 + V26,
  data  = df_train,
  ntree = 1000,
  importance = TRUE
)

# Error rate vs trees plot
png("output/rf_error_curve.png", width = 800, height = 500)
plot(rf_final,
     main  = "RF Final Model — Error Rate vs. Number of Trees",
     col   = c("black", "#3fb950", "#f85149"),
     lwd   = 2)
legend("topright",
       legend = c("OOB Error", "Legit Error", "Fraud Error"),
       col    = c("black", "#3fb950", "#f85149"),
       lty    = 1, lwd = 2, bty = "n")
dev.off()

df_test$pred_final <- predict(rf_final, newdata = df_test)
cm_final  <- confusionMatrix(df_test$pred_final, df_test$Class, positive = "Fraud")
f1_final  <- F1_Score(df_test$Class, df_test$pred_final, positive = "Fraud")
cat("\nFinal RF Model (top-10 vars, 1000 trees)\n")
cat("F1 Score:", round(f1_final, 4), "\n")
print(cm_final)


# ── 9. Model Comparison Summary ───────────────────────────────────────────────
f1_lr    <- F1_Score(df_test$Class, lr_preds, positive = "Fraud")
summary_df <- data.frame(
  Model    = c("Logistic Regression", "Random Forest (Full)",
               "Random Forest (Top-10, 1000 trees)"),
  F1_Score = round(c(f1_lr, f1_full, f1_final), 4),
  Variables = c(ncol(df_train) - 1, ncol(df_train) - 1, 10)
)
cat("\n=== Model Comparison ===\n")
print(summary_df)

p_comp <- ggplot(summary_df, aes(x = reorder(Model, F1_Score), y = F1_Score,
                                  fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = F1_Score), hjust = -0.2, size = 4) +
  coord_flip() +
  scale_fill_manual(values = c("#58a6ff", "#3fb950", "#ffa657")) +
  labs(title = "Model Comparison — F1 Score",
       x = NULL, y = "F1 Score") +
  theme_minimal(base_size = 13) +
  xlim(0, 1.05)
ggsave("output/model_comparison.png", p_comp, width = 8, height = 4)

cat("\nAll outputs saved to output/\n")
spark_disconnect(sc)
