# CS363M Final Project: Shared Context & Tracker

**Predicting high-selling video games**, CS363M Machine Learning, UT Austin, Fall 2026.
This file is the team's shared source of truth: what the project is, what's decided, and what's left. Teammates and AI assistants should read it before working on the notebook.

| | |
|---|---|
| **Notebook + Process Writeup due** | **Nov 19, 2026, midnight** (no slip days) |
| **Q&A session** | Dec 1, 2026 (sign up for a slot) |
| **Assignment** | [`Project_Expectations.pdf`](Project_Expectations.pdf) |
| **Our proposal** | [`ML_Project_Video_Games.pdf`](ML_Project_Video_Games.pdf) |
| **Notebook** | [`../MLProject.ipynb`](../MLProject.ipynb) |

---

## How to use this tracker

1. **Pull first.** Pick an unchecked item and mark it `(in progress)` so nobody duplicates it.
2. **One person edits `MLProject.ipynb` at a time**, because notebooks merge badly in git. Say so in the group chat before you start.
3. Before pushing, do **Kernel → Restart & Run All**. It must run with no errors.
4. In the same commit, check the item off (`- [x]`). Add any new decision to the [Decisions](#decisions) table.

---

## Rules

- **AI is NOT allowed for the Process Writeup.** The team writes every word of it. AI is allowed for notebook code, debugging and explanations.
- **We are graded on methodology, not accuracy.** Even 60% accuracy is fine. Show a careful process: cleaning, exploration, features, tuning, comparison and interpretation.
- **Everyone must be able to explain every cell at the Q&A.** Keep the code plain and readable, using standard scikit-learn.
- **Use established libraries** (pandas, scikit-learn, matplotlib, seaborn). Never implement an algorithm from scratch.
- **All code lives in `MLProject.ipynb`.** Never edit the raw CSV.

### Rubric
| Category | Points |
|---|---|
| Data Cleaning | 10 |
| Data Exploration | 10 |
| **Feature Engineering** | **20** |
| **Modeling** (correct techniques, comparison, tuning) | **40** |
| Code & Notebook Quality (narration, visuals; writeup clarity) | 10 |
| Outcome / Results (complete, correctly interpreted) | 10 |

The notebook needs these sections: **Introduction → Data Exploration & Preprocessing → Modeling → Results & Analysis**. Markdown should *briefly* explain the code and interpret the outputs. The "why" goes in the writeup.

---

## Problem

- **Goal:** predict whether a game is **high-selling**, meaning it sold **at least 1,000,000 copies** worldwide.
  - `label_highselling = 1` if `total_sales >= 1.0` (the column is in millions of copies), else `0`.
- **Type:** binary classification.
- **Required models:** Decision Tree and K-Nearest Neighbors. Extra models, such as ensembles, earn modeling points.
- **Features:** `console`, `genre`, `publisher`, `developer`, `critic_score`, and `release_date` (plus features engineered from these).
- **Never used as features:**
  - `total_sales` and the regional sales columns: they leak the answer.
  - `img`, `last_update`: no predictive meaning.
  - `title`: kept only as `game_titles`, for grouping the split.
- **Data:** Kaggle, [yasrebi/video-game-sales](https://www.kaggle.com/datasets/yasrebi/video-game-sales) (VGChartz). It has 64,016 rows × 14 columns, and each row is **one game on one console**.

## Decisions

| Date | Decision | Why |
|---|---|---|
| Oct | High-selling = `total_sales >= 1.0` (≥ 1M copies) | Definition from the proposal |
| Oct | Drop rows with no `total_sales` | They can't be labeled |
| Oct | Drop `img`, `total_sales`, `na/jp/pal/other_sales` after creating the label | Leakage / not predictive |
| Nov 3 | Save `game_titles` before dropping `title` | Needed to split by game (see the data facts below) |
| Nov 3 | Remove exact duplicate rows (11) | Redundant records |
| Nov 3 | **Keep only complete rows (`dropna()`), leaving 4,122 games** | Keeps only records with every feature filled, especially `critic_score`. Trade-off: 78% of labeled rows are dropped, and the model applies to games that were *reviewed*. Mention this in the writeup and the Q&A |
| Nov 7 | The current cleaning code gives **4,123 games (762 high-sellers, 18.5%)**: 21 duplicates removed, 18,901 → 4,123 rows after `dropna()`, hit rate 8.0% → 18.5% | These are the numbers the notebook prints and the markdown uses. The 1-row difference from 4,122 is not caused by dropping `developer` (keeping it also gives 4,123) |
| Nov 7 | Part 3 features: `platform_family` (PlayStation/Xbox/Nintendo/PC/Other; Sega → Other), `handheld` (GB, GBC, GBA, DS, 3DS, PSP, PSV; Switch counts as a home console), `num_platforms` (distinct consoles per title), `release_quarter`, `holiday_release` (Oct–Dec), `critic_score_bin` (≤6 / 6–8 / >8 → 0/1/2). Still one-hot encoded with `get_dummies` | None of them uses the label, so they are safe to compute before the split. `get_dummies` was kept so the existing Part 4 cells keep running; the Part 4 setup can move the encoding into a `Pipeline` |

## Data facts (after cleaning: the data we model)

| Fact | What it means for us |
|---|---|
| **4,122 games**; **762 high-sellers (18.5%)** | Imbalanced: a model that always says "no" gets 81.5% accuracy. Report balanced accuracy, F1, precision/recall and PR-AUC, and compare against a dummy baseline. |
| Before `dropna()`: 18,647 games, 8.0% high-sellers | Reviewed games sell better. This explains the jump to 18.5% and belongs in the analysis. |
| **1,958 rows (47%) share a title with another row**: the same game on several consoles | **The train/test split must be grouped by `game_titles`.** With a random split, the same game lands in both train and test, which inflates scores. |
| `console` has 28 values, `genre` 20, `publisher` 214, `developer` 1,050 | Publisher and developer have many values: 122 publishers and 834 developers have fewer than 5 games. Group rare ones into "Other" or use frequency encoding. |
| `critic_score` ranges 1.0–10.0 (median 7.3); release years 1989–2020 | No missing values remain after `dropna()`. |

---

## To-do list (by notebook section)

Checked items were reviewed on Nov 7 against commit `18dfcdb` ("added Decision Trees method"). The notebook runs cleanly. Items marked *(partial)* have been started but don't yet meet the description.

### Introduction
- [x] Title and team names
- [x] Problem statement and target definition (≥ 1M copies; 1 = high-selling, 0 = not)
- [x] **Dataset description**: source (Kaggle/VGChartz link), 64,016 rows × 14 columns, one row = one game on one console, years covered
- [x] Tidy the headers: cell 1 has both "The Problem trying to solve" and "1. Introduction"
- [x] Imports cell: remove the duplicated `numpy`/`pandas`/`matplotlib` imports and add `SEED = 42` (use `random_state=SEED` everywhere)

### Part 1: Data Exploration
- [x] Shape, `.info()`, `.describe()`
- [x] Missing values table and bar chart
- [x] Number of categories in genre, console, publisher and developer
- [x] Class distribution bar chart
- [x] Finish the markdown under the class chart (currently: "The graph because it shows the 92% vs. 8% imbalance")
- [x] **Explore the cleaned 4,122 games against the label.** These plots can go under a new "Exploring the cleaned data" subheading at the end of Part 2.
  - [x] Hit rate by genre (bar chart)
  - [x] Hit rate by console
  - [x] Number of games and hit rate by release year
  - [x] `critic_score` vs label (box plot or histogram per class)
  - [x] Top publishers by game count and by hit rate
  - [x] How many games appear on several consoles (motivates the grouped split)
- [x] A 1–2 sentence interpretation under every plot

### Part 2: Data Cleaning
- [x] Create the label before dropping `total_sales`
- [x] Drop `img` and all sales columns (leakage)
- [x] Save `game_titles` for the grouped split
- [x] Keep only the feature columns
- [x] Remove duplicates (11)
- [x] `dropna()` keeps 4,122 complete games (bar chart before/after)
- [x] **Markdown explaining the `dropna()` choice and its effect**: 18,647 → 4,122 rows, and the hit rate goes from 8.0% to 18.5%
- [x] Show the class balance *after* cleaning (762 high-sellers, 18.5%)
- [x] Re-align titles to the final rows: `game_titles = game_titles.loc[videogame_df.index]`
- [x] Parse dates: `videogame_df["release_date"] = pd.to_datetime(videogame_df["release_date"])` (done at the start of Part 3)
- [x] Sanity checks: category strings (spaces, capitalization, "Unknown") and the `critic_score` range
- [x] Fix the small text issues: the "Feartures" typo, and make the duplicates markdown match the output

### Part 3: Feature Engineering (20 pts, the biggest preparation item)
- [x] Date features: `release_year`, `release_month`, `release_quarter`, `holiday_release` (Oct–Dec); then drop raw `release_date`
- [x] Console features: `platform_family` (PlayStation / Xbox / Nintendo / PC / Other) and a `handheld` flag
- [x] `num_platforms`: how many consoles each title was released on, computed from `game_titles` (signals a big multi-platform release)
- [x] Publisher/developer: group rare values into "Other" (e.g. fewer than 5 games), and/or add a `publisher_game_count` frequency feature (done: publishers outside the top 50 → "Other", covering 87.9% of games; `developer` was dropped from the features)
- [x] Optional, from the proposal: bin `critic_score` into categories (e.g. low / mid / high)
- [x] Any feature that uses the **label** (e.g. a publisher's hit rate) must be computed **on training data only**; skip it if unsure
- [x] Show the engineered features (`.head()`) and plot at least one new feature against the label
- [x] Brief markdown on each feature

### Part 4: Modeling (40 pts)
**Setup**
- [ ] `X` / `y`. Grouped train/test split: `GroupShuffleSplit(test_size=0.2, random_state=SEED)` on `game_titles`. Add an `assert` that no title is in both sets, and show the class balance in each.
- [ ] Preprocessing `ColumnTransformer`: `OneHotEncoder(handle_unknown="ignore")` for categorical columns, `StandardScaler` for numeric ones (needed for KNN). Wrap it in a `Pipeline` with each model.
- [ ] An `evaluate()` helper so every model is scored the same way: balanced accuracy, precision, recall, F1, PR-AUC, confusion matrix
- [ ] Baseline: `DummyClassifier`

**Decision Tree**
- [ ] Untuned tree, cross-validated on train
- [ ] `GridSearchCV` over `max_depth`, `min_samples_leaf`, `criterion`, `class_weight`, using `StratifiedGroupKFold` CV (pass `groups=`) and `scoring="f1"` or `"average_precision"`
- [ ] Plot score vs `max_depth` (train vs CV, to show overfitting) and report the best parameters *(partial: train-vs-validation accuracy curves for `max_depth` and `min_samples_leaf` exist, but they score on the test set, use a random split, and the chosen values are still "[blank]")*
- [ ] Feature importances, or a plot of the tree's top levels

**K-Nearest Neighbors**
- [ ] Untuned KNN (with scaling) cross-validated on train
- [ ] Tune `n_neighbors`, `weights`, `p` the same way
- [ ] Plot score vs k
- [ ] Short note on why KNN needs scaling, and on the effect of many one-hot columns

**Improvements (extra modeling points)**
- [ ] At least one ensemble (Random Forest and/or Gradient Boosting), tuned the same way
- [ ] Compare `class_weight="balanced"` with no class weighting (optionally decision-threshold tuning)
- [ ] Comparison table of the cross-validation scores of every model

### Part 5: Results & Analysis
- [ ] Pick the final models from the CV results, then score the **test set once**
- [ ] Table of test metrics for every model vs the dummy baseline
- [ ] Confusion matrices and precision-recall curves
- [ ] Feature importance (e.g. permutation importance) with an interpretation
- [ ] Error analysis: which genres, consoles or years get misclassified?
- [ ] Markdown interpreting the results, including limitations (e.g. the model only covers reviewed games, because of `dropna()`)

### Final polish
- [ ] Section headers cover Introduction / Data Exploration & Preprocessing / Modeling / Results & Analysis
- [ ] Restart & Run All passes with outputs saved; no dead or debug cells
- [ ] Every code cell has a short markdown lead-in; every plot has a title, axis labels and an interpretation
- [ ] **Submit the notebook by Nov 19**

### Process Writeup (written by the team only, no AI; PDF, max 5 pages)
- [ ] Data exploration: key observations
- [ ] Cleaning and preprocessing: what was fixed and why (including `dropna()`)
- [ ] Feature engineering: what was created and why
- [ ] Algorithm selection: what was tried and why
- [ ] Hyperparameter tuning: which parameters were tuned and why
- [ ] Results and analysis
- [ ] **Work split** section at the end (who did what)
- [ ] **Submit the PDF by Nov 19**

### Q&A prep (Dec 1): everyone should be able to answer
- [ ] Why ≥ 1M copies? Why `dropna()`, and what did it change?
- [ ] Why is accuracy misleading here? Which metric did we optimize?
- [ ] Why split by title? What goes wrong with a random split?
- [ ] How did we encode publisher and developer? What features did we engineer, and why?
- [ ] Why does KNN need scaling when a tree doesn't? What did tuning show?
- [ ] Which model won, by how much over the baseline, and what does it get wrong?

---

## ML rules (common ways to lose points)

1. **No leakage.** Sales columns never become features. Anything computed from the label uses training data only.
2. **Grouped split.** Split train/test **by `game_titles`** and cross-validate with `StratifiedGroupKFold`, passing `groups=`.
3. **Use the test set once,** at the very end. Choose everything else with cross-validation on train.
4. **Preprocessing goes inside a `Pipeline`,** so encoding and scaling are fit on training data only.
5. **Imbalance-aware metrics.** Never report accuracy alone. Always compare to a `DummyClassifier`.
6. **Reproducible.** Use `SEED = 42` and `random_state=SEED` everywhere.
7. **Show the tuning, not just the winner:** plots or tables of scores across hyperparameters.

Split pattern to reuse:
```python
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV

X = videogame_df.drop(columns=["label_highselling"])
y = videogame_df["label_highselling"]
groups = game_titles.loc[videogame_df.index]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
assert set(groups_train).isdisjoint(groups.iloc[test_idx]), "same game in train and test"

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", return_train_score=True, n_jobs=-1)
search.fit(X_train, y_train, groups=groups_train)
```

## Course topics (use techniques from class)

The assignment says to "apply techniques learned in class". Algorithms from these lectures are the easiest to defend at the Q&A, so prefer them over methods we didn't cover.

| # | Topic | Use in this project |
|---|---|---|
| 1 | Data Preprocessing | Cleaning, encoding, scaling |
| 2 | Decision Trees | Required model |
| 3 | Linear Regression | n/a (we're classifying) |
| 4 | Cross Validation & Overfitting | Grouped CV, tuning curves, train vs CV gap |
| 5 | Nearest Neighbor | Required model (KNN) |
| 6 | Naive Bayes | Candidate extra model / baseline |
| 7 | Evaluating Classifiers | Precision, recall, F1, PR-AUC, confusion matrix |
| 8 | Support Vector Machines | Candidate extra model |
| 9 | Non-linear Regression | n/a |
| 10 | Neural Nets | Candidate extra model |
| 12 | Ensemble Methods | Bagging, Random Forest, boosting, voting |
| 13–14 | Clustering: K-Means, DBSCAN, Hierarchical | Optional exploration |

## Resources (from the assignment)
- [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering): mutual information, creating features, target encoding
- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html): pipelines and cross-validation
- [scikit-learn: Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html): GridSearchCV, RandomizedSearchCV, scoring
- [Handling imbalanced datasets](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7c81ee86d922): class imbalance strategies
- [Nine Rules Not to Follow](https://www.kdnuggets.com/2017/04/nine-rules-not-to-follow-data-mining.html): common data-mining pitfalls
- [ML Cheat Sheets](https://github.com/amitpappalkar/ML-Cheat-Sheets): performance improvement ideas

Environment: Python with pandas, numpy, matplotlib, seaborn and scikit-learn (`pip install scikit-learn` if it's missing).
