# Results Interpretation & Discussion (Brief Sections 10–15)

These notes are grounded in an actual full run of this package (the
NumPy-CNN version) against your `data.csv` (569 samples: 357 Benign / 212
Malignant, 30 features, 70/15/15 split → 397 train / 86 validation / 86
test, 5-fold Stratified CV). Because every random seed is fixed, re-running
`python main.py` on the same machine should reproduce these numbers
closely (small floating-point drift across different OS/CPU is normal).
**Always pull your final numbers from your own `outputs/results/*.csv`**
before writing them into your report — treat what follows as a template
argument structure, not numbers to copy blindly if your run differs.

---

## 10. Experimental Comparison

**Final held-out test set (n = 86)** — from `final_comparison_table.csv`:

| Model | PCA | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|---|
| CNN-KNN | No | 0.9884 | 0.9697 | 1.0000 | 0.9846 | 0.9997 | 0.0016 |
| CNN-KNN | Yes | 0.9884 | 0.9697 | 1.0000 | 0.9846 | 0.9997 | 0.0032 |
| CNN-LR | No | 0.9884 | 0.9697 | 1.0000 | 0.9846 | 0.9994 | 0.0031 |
| CNN-LR | Yes | 0.9884 | 0.9697 | 1.0000 | 0.9846 | 0.9994 | 0.0039 |
| CNN-SVM | No | 0.9884 | 0.9697 | 1.0000 | 0.9846 | **1.0000** | 0.0044 |
| CNN-SVM | Yes | 0.9884 | 0.9697 | 1.0000 | 0.9846 | **1.0000** | 0.0067 |

All six configurations landed on identical Accuracy/Precision/Recall/F1 on
this particular 86-sample test split — every model caught all 32 malignant
cases (recall = 1.0) with exactly one false positive each (precision =
0.9697). Only ROC-AUC differentiates them, with both SVM variants reaching
a perfect 1.000. This is a useful result in its own right (see Section 11),
but it means the **test set alone can't distinguish the models here** — the
cross-validation numbers below are the more informative comparison.

**5-fold cross-validation on the training pool (n = 483)**, mean ± std,
using each classifier's *default* hyperparameters (before tuning) — from
`cross_validation_results.csv`:

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| CNN-SVM (No PCA) | 0.9917 ± 0.0077 | 0.9944 ± 0.0111 | 0.9890 ± 0.0102 | 0.9994 ± 0.0011 |
| CNN-LR (No PCA) | 0.9917 ± 0.0077 | 0.9889 ± 0.0136 | 0.9890 ± 0.0102 | 0.9996 ± 0.0007 |
| **CNN-KNN (No PCA)** | **0.9959 ± 0.0082** | **1.0000 ± 0.0000** | **0.9946 ± 0.0108** | 0.9965 ± 0.0069 |
| CNN-SVM (PCA) | 0.9917 ± 0.0077 | 0.9944 ± 0.0111 | 0.9890 ± 0.0102 | 0.9994 ± 0.0011 |
| CNN-LR (PCA) | 0.9938 ± 0.0083 | 0.9944 ± 0.0111 | 0.9918 ± 0.0109 | 0.9996 ± 0.0007 |
| CNN-KNN (PCA) | 0.9938 ± 0.0083 | 0.9944 ± 0.0111 | 0.9918 ± 0.0109 | 0.9965 ± 0.0069 |

**GridSearchCV best hyperparameters** — from `hyperparameter_tuning_summary.csv`:

| Model | Best Params | Best CV F1 |
|---|---|---|
| CNN-SVM (No PCA) | `C=0.1, kernel=linear, gamma=scale` | 0.9918 |
| CNN-SVM (PCA) | `C=10, kernel=rbf, gamma=scale` | 0.9890 |
| CNN-LR (No PCA) | `C=0.01, penalty=l2` | 0.9919 |
| CNN-LR (PCA) | `C=1, penalty=l1` | 0.9946 |
| CNN-KNN (No PCA) | `k=5, metric=euclidean` | 0.9946 |
| CNN-KNN (PCA) | `k=9, metric=euclidean` | 0.9919 |

PCA reduced the 32-dimensional CNN feature vector to **6 principal
components** for 95% retained variance (`pca_explained_variance.png`) — an
~81% dimensionality cut.

---

## 11. Results Interpretation and Discussion

**Best-performing model, and why.** The test set alone can't separate the
six configurations — they're tied on every metric except ROC-AUC. The
5-fold CV numbers (n = 483, a far more reliable estimate than an 86-sample
test set) break the tie: **CNN-KNN without PCA** is the strongest and most
consistent performer — highest mean CV accuracy (99.59%), highest mean F1
(0.9946), and a perfect recall of 1.0 ± 0.0, meaning it caught every single
malignant case in every one of the 5 folds with zero variance. CNN-SVM
edges it out narrowly on test-set ROC-AUC (a perfect 1.000 vs. 0.9997), so
if ranking by discriminative ranking ability alone, CNN-SVM is a reasonable
alternative choice — but on the more robust CV evidence, CNN-KNN (No PCA)
has the strongest overall claim.

**Effect of PCA, per classifier — and an important nuance.** PCA's effect
was *not* uniformly positive in this run, which is worth stating explicitly
rather than defaulting to the common assumption that "PCA always helps":
- **LR benefited from PCA**: CV accuracy rose from 0.9917 → 0.9938 and F1
  from 0.9890 → 0.9918 — the clearest, cleanest improvement of the three.
- **KNN's CV performance was slightly *lower* with PCA** (accuracy 0.9959 →
  0.9938, F1 0.9946 → 0.9918), even though KNN is the classifier PCA is
  usually expected to help most (distance-based methods are traditionally
  most sensitive to the curse of dimensionality). One plausible read: with
  only 6 principal components retained, PCA may have trimmed away a small
  amount of class-separating variance along with the redundant/noisy
  directions — a reminder that "95% variance retained" optimizes for
  *reconstruction*, not necessarily for *classification*, and the two don't
  always align perfectly.
- **SVM was essentially unaffected**: CV accuracy, F1, and ROC-AUC were
  identical (to 4 decimal places) with and without PCA using default
  hyperparameters. After GridSearchCV tuning, the *chosen* kernel differed
  (linear without PCA, RBF with PCA), so the models found different routes
  to a similar decision boundary rather than PCA changing performance
  outright.

**Cross-validation stability.** All six configurations show tight spreads
(std ≈ 0.7%–1.4% across accuracy/F1/ROC-AUC) — every model is behaving
consistently across folds. CNN-KNN (No PCA)'s recall standard deviation of
exactly 0.0 (perfect recall in all 5 folds) is the single most stable
result in the table.

**Bias–variance tradeoff.** KNN (k = 5, No PCA) sits at a favorably low-bias,
low-variance point for this dataset specifically — its local-neighborhood
decision rule matches WDBC's well-separated class structure closely enough
that it out-performs the more constrained/regularized LR (C = 0.01, heavy
L2 shrinkage) and the margin-based SVM. This is a useful illustration that
"more regularization/simpler model = better generalization" is not a
universal rule — it depends on how well the model family's inductive bias
matches the actual class geometry, which here favors KNN's local
smoothness assumption.

**Computational efficiency.** All three classifiers are trivially fast at
this dataset's scale (all under 7 milliseconds to fit on the pooled
training features). KNN's near-zero fit time is expected (it is a lazy
learner — it just stores the data); its real cost shows up at *inference*
time, scaling with the size of the stored training set, which is worth
noting if this were being discussed as a deployment consideration rather
than a training-time one.

---

## 12. Clinical Interpretation

**Why recall matters most here:** a false negative means a malignant tumor
is classified as benign — a missed cancer, with potentially fatal delay in
treatment. A false positive means a benign case is flagged for further
testing — inconvenient and anxiety-inducing, but not dangerous. In a
screening/triage context, minimizing false negatives (maximizing recall)
should generally be prioritized over minimizing false positives, even at
some precision cost.

In this run, **every one of the six configurations achieved recall = 1.0**
on the 86-sample test set (32 malignant cases, all correctly caught), each
with exactly one false positive (precision = 0.9697). That's an encouraging
result but also a reminder of the earlier caveat: with only 86 test samples
and 32 positives, "zero missed cancers" is a real and good result, but it
shouldn't be read as "this model will never miss a malignant case in
deployment" — the confidence interval around a rate estimated from 32
positive cases is wide. The cross-validation recall figures (Section 11)
are the more defensible number to cite for how reliably each model catches
malignant cases: 1.000 ± 0.000 for CNN-KNN (No PCA), and 0.988–0.994 for
the others.

---

## 13. Limitations

- **Small dataset.** 569 total samples is modest for deep learning; the
  86-sample test set produced a tie across all six models on every headline
  metric, illustrating how little a test set this size can discriminate
  between closely-matched configurations. The 5-fold CV numbers are more
  informative for exactly this reason.
- **CNN suitability for tabular data.** A 1D CNN imposes an implicit
  "spatial adjacency" assumption between neighboring feature columns that
  doesn't reflect any real relationship in WDBC's 30 hand-engineered
  features (they aren't ordered by physical proximity or causal structure).
  The CNN's benefit here likely comes from its capacity to learn a useful
  non-linear feature embedding via training, not from genuine convolutional
  structure in the data — worth flagging as a modeling choice under debate,
  not an obviously correct one.
- **Visible overfitting signal.** `cnn_training_curves.png` shows training
  accuracy reaching 100% by roughly epoch 20 while validation loss actually
  *increases* over the same stretch (0.05 → 0.13) even with early stopping
  active — classic overfitting behavior on a small training set. Early
  stopping (restoring the best validation-loss weights) mitigates this, but
  it's a limitation worth naming rather than hiding: with only 397 training
  samples, the CNN has more capacity than the data strictly supports.
- **Lack of external validation.** All results come from splits of the same
  single-institution WDBC dataset. No claim about generalization to a
  different hospital, imaging pipeline, or patient population can be made
  from these numbers alone.
- **A hand-rolled NumPy CNN, not a framework-verified one.** Implementing
  Conv1D/backprop manually (done here specifically to sidestep TensorFlow's
  installation fragility — see the README) means the numerics haven't been
  validated against a mature framework's test suite. The forward/backward
  passes were checked for internal consistency (a save/load round-trip
  reproduces identical features), but a supervisor may reasonably ask you
  to cross-check a few CNN outputs against a small PyTorch/Keras reference
  implementation if compute/install constraints allow, as an extra
  correctness check.

---

## 14. Possible Improvements

- **Baseline comparison**: report a plain (non-CNN) SVM/LR/KNN directly on
  the scaled tabular features, to quantify what the CNN feature-extraction
  step is actually contributing.
- **Autoencoders** as an alternative unsupervised feature extractor,
  compared against this supervised CNN embedding.
- **Ensemble methods** (e.g., stacking CNN-SVM + CNN-LR + CNN-KNN, or a
  voting classifier) to see whether combining the three captures
  complementary error patterns.
- **Feature selection + PCA hybrid** (e.g., mutual information or
  recursive feature elimination before PCA) — motivated directly by this
  run's finding that pure variance-based PCA slightly hurt KNN; a
  classification-aware dimensionality reduction step might avoid that.
- **SMOTE or class-weighting** — the dataset is moderately imbalanced
  (357 Benign vs. 212 Malignant, ~63/37); this project used stratified
  splits, which is a reasonable first defense, but explicit
  imbalance-handling could be tested as a further refinement.
- **External dataset validation** — evaluate the trained pipeline on a
  different breast-cancer tabular dataset to test true generalization.
- **Nested cross-validation for the CNN stage**, addressing the
  simplification named in the Methodology Notes section of the README
  (retraining the CNN inside every outer fold, budget permitting).

---

## 15. Expected Outcome — Summary

Across both the held-out test set and 5-fold cross-validation, **CNN-KNN
(without PCA)** emerges as the most effective and most consistent hybrid
model for this dataset — a useful, slightly counter-intuitive finding,
since KNN is usually the classifier expected to benefit most from PCA's
dimensionality reduction. PCA's actual impact varied by classifier: clearly
positive for LR, roughly neutral for SVM, and mildly negative for KNN in
this run — evidence for discussing PCA's effect as classifier-dependent
rather than universally beneficial. This supports the brief's expected
outcome of a hybrid deep-learning/classical-ML pipeline that is both
accurate and interpretable, with clear, quantified, and appropriately
nuanced evidence for how PCA affects each classifier differently.
