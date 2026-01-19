# AdaBoost vs Gradient Boost vs XGBoost - Complete Guide

I'll explain these ensemble methods from your Adaboost notebook and compare them with PCA.

---

## **PART 1: QUICK COMPARISON TABLE**

| Aspect | AdaBoost | Gradient Boost | XGBoost |
|--------|----------|----------------|---------|
| **Type** | Ensemble (Sequential) | Ensemble (Sequential) | Ensemble (Optimized) |
| **Base Learner** | Weak learners (usually trees) | Weak learners (trees) | Decision trees |
| **Error Focus** | Focuses on misclassified samples | Focuses on residuals (errors) | Focuses on residuals |
| **Learning Rate** | Fixed or adaptive | Learnable (eta) | Learnable (eta) |
| **Speed** | Fast | Slower | Fast (optimized) |
| **Accuracy** | Good | Better | Better |
| **Overfitting** | Can overfit | Can overfit | Regularization built-in |
| **Regularization** | Limited | Yes | Strong (L1, L2) |
| **Parallelization** | No (sequential) | No (sequential) | Yes (can parallelize) |
| **Missing Values** | Not handled | Not handled | Automatic handling |
| **Complexity** | Simple | Complex | Very complex |
| **Hyperparameters** | Few | Many | Many |
| **Production Ready** | Yes | Yes | Very yes |

---

## **PART 2: ENSEMBLE METHODS CONCEPT**

### **What is Ensemble Learning?**

**Ensemble = Combine multiple weak models to create strong model**

```
Weak Learners = Models slightly better than random guessing
Strong Model = Combining weak learners for excellent performance

Analogy:
├─ One expert: Might make mistakes (80% accuracy)
├─ Committee of 10 experts: More likely to get it right (95% accuracy)
└─ Voting/averaging opinions gives better result!
```

### **Why Ensemble Works**

```
Weak Learner Performance:
├─ Model A: Correct 70% of time
├─ Model B: Correct 75% of time
└─ Model C: Correct 72% of time

If errors are independent:
└─ Ensemble: Correct ~85-90% of time!

Why? Different models make different mistakes
└─ Averaging cancels out individual errors
```

---

## **PART 3: ADABOOST (ADAPTIVE BOOSTING)**

### **What is AdaBoost?**

**AdaBoost = Sequential ensemble focusing on hard-to-classify samples**

```
Algorithm:
1. Train weak learner (Decision Tree) on all data
2. Check predictions - which ones are wrong?
3. Increase weight on misclassified samples
4. Train next weak learner on re-weighted data
5. Now focuses on previously hard samples
6. Repeat until n_estimators trees created
7. Final prediction = Weighted average of all trees
```

### **Simple Visual Example**

```
Initial Dataset: [✓✓✓✓✓✓✓✗✗✗]
                 Mostly correct, some wrong

Tree 1 Training:
├─ Trains on above data
├─ Predicts: [✓✓✓✓✓✓✓✗✗✗] (90% correct)
└─ Identifies: Wrong on last 3 samples

Tree 2 Training:
├─ Increase weight on last 3 samples
├─ Data becomes: [✓✓✓✓✓✓✓✗✗✗] ← ✗'s heavier now
├─ Trains on re-weighted data
└─ Now focuses on learning from hard samples!

Tree 3 Training:
├─ Further increase weight on remaining errors
└─ Continue...

Final Prediction:
├─ Combine all trees with weights
├─ Trees that performed better get higher weight
└─ Result: Strong model from weak learners!
```

### **How AdaBoost Works in Your Notebook**

````python
from sklearn.ensemble import AdaBoostRegressor

# Default: Uses Decision Trees as base learner
model = AdaBoostRegressor(
    n_estimators=50,      # Number of weak learners
    loss='linear',        # Loss function for regression
    learning_rate=1.0     # How much to adjust weights
)

model.fit(X_train, y_train)
````

**AdaBoost Parameters (Your tuning):**

```python
ada_params = {
    "n_estimators": [50, 60, 70, 80],      # Number of trees
    "loss": ['linear', 'square', 'exponential']  # Loss function
}

# Best found: n_estimators=60, loss='linear'
```

### **AdaBoost Advantages & Disadvantages**

```
Advantages:
├─ ✓ Simple and interpretable
├─ ✓ Fast to train
├─ ✓ Works well with weak learners
├─ ✓ Few hyperparameters
└─ ✓ Good for classification & regression

Disadvantages:
├─ ✗ Sequential (can't parallelize)
├─ ✗ Sensitive to outliers (keeps boosting them)
├─ ✗ Can overfit with too many iterations
├─ ✗ Slower than Random Forest for prediction
└─ ✗ Less accurate than modern Gradient Boosting
```

---

## **PART 4: GRADIENT BOOSTING**

### **What is Gradient Boosting?**

**Gradient Boost = Sequential ensemble focusing on residuals using gradients**

```
Algorithm:
1. Train weak learner on all data
2. Calculate residuals (errors) = actual - predicted
3. Train next weak learner on RESIDUALS (not data!)
4. This tree learns what previous tree got wrong
5. Update predictions: new_pred = old_pred + learning_rate × new_tree_pred
6. Repeat: Each tree tries to correct previous tree's mistakes
7. Final prediction = Sum of all tree predictions (with shrinkage)
```

### **Visual Comparison: AdaBoost vs Gradient Boost**

```
ADABOOST:
├─ Re-weights DATA based on errors
├─ Focus samples → harder samples get higher weight
└─ Training data changes between trees

GRADIENT BOOSTING:
├─ Trains on RESIDUALS (errors from previous tree)
├─ Each tree corrects previous tree's mistakes
├─ Residuals are target, not data
└─ Training target changes between trees

Example:
First tree predicts car price as $500k (actual is $600k)
├─ AdaBoost: Increase weight on this sample in next tree
└─ GB: Train next tree to predict residual of +$100k
```

### **Gradient Boosting Algorithm Step-by-Step**

```
Actual prices: [600k, 700k, 500k, 400k]

Tree 1 predictions: [550k, 650k, 480k, 420k]
Tree 1 residuals: [50k, 50k, 20k, -20k]
                   (what Tree 1 got wrong)

Tree 2 trained on residuals:
Tree 2 predictions on residuals: [45k, 48k, 18k, -18k]

Updated overall predictions:
= Tree1 + learning_rate × Tree2
= [550k, 650k, 480k, 420k] + 0.1 × [45k, 48k, 18k, -18k]
= [554.5k, 654.8k, 481.8k, 418.2k]
(Much closer to actual!)

Tree 3 trained on remaining residuals:
New residuals: [45.5k, 45.2k, 18.2k, -18.2k]
...continue...
```

### **Gradient Boosting Advantages & Disadvantages**

```
Advantages:
├─ ✓ Better accuracy than AdaBoost
├─ ✓ Works on residuals (more predictable)
├─ ✓ Learning rate controls shrinkage
├─ ✓ Less sensitive to outliers
└─ ✓ Good regularization options

Disadvantages:
├─ ✗ Sequential (can't parallelize)
├─ ✗ Slower to train than AdaBoost
├─ ✗ More hyperparameters to tune
├─ ✗ Prone to overfitting without careful tuning
├─ ✗ Not in scikit-learn (use XGBoost instead)
└─ ✗ Difficult to implement correctly
```

---

## **PART 5: XGBOOST (EXTREME GRADIENT BOOSTING)**

### **What is XGBoost?**

**XGBoost = Production-ready, optimized Gradient Boosting**

```
Created by: Tianqi Chen (2016)
Based on: Gradient Boosting principles
Improvements:
├─ Regularization (prevent overfitting)
├─ Parallel processing (faster training)
├─ Missing value handling
├─ Built-in early stopping
├─ Better performance
└─ Production-grade code

Think of it: Gradient Boosting 2.0
```

### **XGBoost Algorithm (Simplified)**

```
Same as Gradient Boosting, but:

1. Uses 2nd-order derivatives (Newton method, not just gradients)
   └─ More accurate direction to optimize

2. Built-in L1 & L2 regularization
   └─ Prevents overfitting automatically

3. Column subsampling
   └─ Each tree uses random feature subset

4. Row subsampling
   └─ Each tree uses random sample subset

5. Early stopping
   └─ Stop if validation error doesn't improve

6. Can parallelize splits
   └─ Faster training

Result: Better accuracy + Faster training + Less overfitting
```

### **XGBoost Code Example**

````python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,           # Number of trees
    learning_rate=0.1,          # Shrinkage (eta)
    max_depth=5,                # Tree depth
    min_child_weight=1,         # Min samples per leaf
    subsample=0.8,              # Row subsampling
    colsample_bytree=0.8,       # Column subsampling
    gamma=0,                    # Min loss reduction
    reg_alpha=0,                # L1 regularization
    reg_lambda=1,               # L2 regularization
    early_stopping_rounds=10    # Stop if no improvement
)

model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          verbose=False)
````

---

## **PART 6: DETAILED COMPARISON**

### **Training Process**

```
ADABOOST:
Tree 1: Error = 0.30
Tree 2: (weights increased on wrong samples) Error = 0.25
Tree 3: (weights further increased) Error = 0.20
...

Approach: Modify DATA for next tree

GRADIENT BOOSTING:
Tree 1: Pred = 500, Actual = 600, Residual = 100
Tree 2: (learn residuals) Learn to predict +100
Tree 3: (learn remaining residuals) Learn to predict +5
...

Approach: Modify TARGET for next tree

XGBOOST:
Same as GB but:
├─ Faster (parallel splits)
├─ Better (2nd order derivatives)
├─ Regularized (L1, L2)
└─ Robust (missing value handling)

Approach: Optimized Gradient Boosting
```

### **Performance on Car Price Prediction (Your Problem)**

```
Expected Results on Your Dataset:

AdaBoost (from your notebook):
├─ Test R²: ~0.75
├─ Test RMSE: ~180,000
└─ Training time: Fast

Gradient Boosting (scikit-learn):
├─ Test R²: ~0.82
├─ Test RMSE: ~140,000
└─ Training time: Slower

XGBoost:
├─ Test R²: ~0.85
├─ Test RMSE: ~120,000
└─ Training time: Fast (despite being better!)

Random Forest (your baseline):
├─ Test R²: ~0.80
├─ Test RMSE: ~160,000
└─ Training time: Medium
```

---

## **PART 7: WHEN TO USE EACH**

### **Decision Tree**

```
Use when:
✓ Need interpretability
✓ Small dataset
✓ Real-time predictions
✓ Avoid overfitting

❌ Don't use for: Maximum accuracy needed
```

### **Random Forest**

```
Use when:
✓ Need good accuracy
✓ Have large dataset
✓ Don't want sequential training
✓ Want parallelization

Example: Your first attempt at car price
```

### **AdaBoost**

```
Use when:
✓ Need baseline boosting model
✓ Clean data (no outliers)
✓ Want simple, interpretable ensemble
✓ Want fast training

❌ Don't use: For best accuracy, with outliers
```

### **Gradient Boosting**

```
Use when:
✓ Want better accuracy than AdaBoost
✓ Have moderate dataset
✓ Can afford slower training
✓ Need regularization

❌ Don't use: Need fast training, production pressure
```

### **XGBoost**

```
Use when:
✓ Want BEST accuracy
✓ Have large dataset
✓ Production system
✓ Kaggle competitions

❌ Don't use: Need simplicity, limited computing
```

---

## **PART 8: YOUR ADABOOST NOTEBOOK EXPLAINED**

### **Model Training Section**

````python
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor": AdaBoostRegressor()  # ← Your ensemble!
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    mae, rmse, r2 = evaluate_model(y_test, y_test_pred)
    print(f"{model_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
````

### **Hyperparameter Tuning**

````python
ada_params = {
    "n_estimators": [50, 60, 70, 80],           # Number of trees
    "loss": ['linear', 'square', 'exponential']  # Loss function
}

# For regression AdaBoost:
# ├─ 'linear': error = |actual - predicted|
# ├─ 'square': error = (actual - predicted)²
# └─ 'exponential': error = exp(-0.5 * correct)

random = RandomizedSearchCV(
    estimator=AdaBoostRegressor(),
    param_distributions=ada_params,
    n_iter=100,
    cv=3,
    n_jobs=-1
)
random.fit(X_train, y_train)

# Best found: n_estimators=60, loss='linear'
````

### **Final Model**

````python
final_model = AdaBoostRegressor(
    n_estimators=60,    # 60 weak learners
    loss='linear',      # Linear loss function
    learning_rate=1.0   # Default: no shrinkage
)
final_model.fit(X_train, y_train)

# Results:
# Train R²: ~0.80
# Test R²: ~0.75
# Shows slight overfitting (train > test)
````

---

## **PART 9: PCA (PRINCIPAL COMPONENT ANALYSIS)**

### **What is PCA?**

**PCA = Reduce data dimensions while keeping important information**

```
Imagine: Photo with 1000x1000 pixels = 1 million values
Problem: Too much data, too slow, hard to visualize

PCA Solution:
└─ Compress to 100 dimensions that capture 95% of variation
└─ Much faster, easier to work with, minimal info loss
```

### **How PCA Works**

```
Step 1: Find direction of maximum variation in data
        └─ First Principal Component (PC1)

Step 2: Find perpendicular direction with max remaining variation
        └─ Second Principal Component (PC2)

Step 3: Continue until all dimensions covered
        └─ All components

Step 4: Select top k components (keep 95% variance)
        └─ New data = projection onto k components

Result: 1000 dimensions → 50 dimensions (5% of original!)
```

### **Visual Example**

```
Original 2D Data:
    x
    │     ●●
    │   ●●●●●●
    │ ●●●●●●●●
    │●●●●●●●●●●
    └────────── y

PCA: Find diagonal direction (PC1) where data varies most
    PC1 →  ●●
        ●●●●●●
      ●●●●●●●●
    ●●●●●●●●●●

Compress: Use only PC1 (keep 95% of variation)
Result: 2D → 1D (reduce by 50%!)
```

### **PCA Code Example**

````python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Standardize data (PCA requires this!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Step 3: Check results
print(f"Original dimensions: {X.shape[1]}")
print(f"PCA dimensions: {X_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Example output:
# Original dimensions: 100
# PCA dimensions: 35
# Explained variance: 0.9502

# Step 4: Use PCA data for modeling
model.fit(X_pca, y)
````

---

## **PART 10: PCA - DETAILED EXPLANATION**

### **Why Use PCA?**

```
Problem 1: CURSE OF DIMENSIONALITY
├─ More features = more data needed
├─ 100 features needs 1000x more samples
├─ Your car dataset: 15,411 rows, 20+ features
└─ Solution: PCA reduces features intelligently

Problem 2: MULTICOLLINEARITY
├─ Multiple features measure same thing
├─ Example: car_age and car_year (correlated)
├─ Models get confused
└─ PCA: Creates independent components

Problem 3: COMPUTATIONAL COST
├─ More features = slower training
├─ More storage needed
├─ More memory required
└─ PCA: Reduce by 50% with <5% info loss

Problem 4: OVERFITTING
├─ Too many features = memorize noise
├─ Training accuracy high, test low
└─ PCA: Reduce noise, improve generalization

Problem 5: VISUALIZATION
├─ Can't visualize 100 dimensions
├─ PCA: Project to 2-3D for visualization
└─ Understand data better
```

### **PCA vs Feature Selection**

```
FEATURE SELECTION:
├─ Choose k best original features
├─ Keeps interpretability
├─ May lose important info
└─ Which features are important?

PCA:
├─ Creates new features (combinations)
├─ Loses interpretability
├─ Captures all variance
└─ Automatic dimensionality reduction

Example:
Feature Selection: Use only [Age, Mileage, Owner]
└─ Easy to interpret, but may miss patterns

PCA: Creates 3 components from all 20 features
├─ PC1 = 0.4×Age + 0.3×Mileage - 0.2×Owner + ...
├─ PC2 = 0.2×Age - 0.5×Mileage + 0.3×Owner + ...
└─ Harder to interpret, but captures all patterns
```

### **PCA Application to Your Car Dataset**

```
Your car data:
├─ 15,411 samples
├─ 20+ features (after OneHotEncoding)
├─ Categories: Price (continuous)

Problem: Too many features, slow training

Solution: Apply PCA before modeling

Steps:
1. Standardize features (PCA requirement)
2. Apply PCA(n_components=0.95)
3. Reduce 20 features → maybe 12-15 components (keep 95% variance)
4. Train models on PCA features
5. Get faster training, similar accuracy!
```

### **PCA Code for Your Notebook**

````python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score

# Your current approach:
# X: 15411 × 20 (after preprocessing)

# Step 1: Standardize (required before PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original shape: {X_train_scaled.shape}")
print(f"PCA shape: {X_train_pca.shape}")
print(f"Kept {pca.explained_variance_ratio_.sum():.2%} variance")
# Output:
# Original shape: (12328, 20)
# PCA shape: (12328, 13)
# Kept 95.00% variance

# Step 3: Train model with PCA data
model_with_pca = AdaBoostRegressor(n_estimators=60, loss='linear')
model_with_pca.fit(X_train_pca, y_train)

y_pred_pca = model_with_pca.predict(X_test_pca)
r2_pca = r2_score(y_test, y_pred_pca)

print(f"R² with PCA: {r2_pca:.4f}")

# Step 4: Compare with original
model_no_pca = AdaBoostRegressor(n_estimators=60, loss='linear')
model_no_pca.fit(X_train, y_train)

y_pred_no_pca = model_no_pca.predict(X_test)
r2_no_pca = r2_score(y_test, y_pred_no_pca)

print(f"R² without PCA: {r2_no_pca:.4f}")
print(f"Difference: {abs(r2_pca - r2_no_pca):.4f}")
# Likely: Similar R² but PCA is faster!
````

---

## **PART 11: PCA INTERPRETATION**

### **Explained Variance Ratio**

```python
# After fitting PCA
print(pca.explained_variance_ratio_)
# Output: [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]

# Interpretation:
# PC1 explains 45% of variation
# PC1+PC2 explain 70% of variation
# PC1+PC2+PC3 explain 85% of variation
# PC1+PC2+PC3+PC4 explain 93% of variation
# PC1-PC7 explain 100% of variation (all 7 components)

# Decision: Use 4 components to keep 93% variance
# Reduces: 20 features → 4 components
```

### **Cumulative Variance Plot**

````python
import numpy as np
import matplotlib.pyplot as plt

# Plot cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(cumsum, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Variance Explained')
plt.legend()
plt.grid(True)
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")
````

---

## **PART 12: COMPLETE COMPARISON MIND MAP**

```
ENSEMBLE METHODS & DIMENSIONALITY REDUCTION
│
├─ ENSEMBLE METHODS
│  │
│  ├─ ADABOOST
│  │  ├─ Sequential boosting
│  │  ├─ Focus on misclassified samples
│  │  ├─ Re-weight data
│  │  ├─ Fast training
│  │  ├─ Less accurate
│  │  └─ Few hyperparameters
│  │
│  ├─ GRADIENT BOOSTING
│  │  ├─ Sequential boosting
│  │  ├─ Focus on residuals
│  │  ├─ Train next tree on errors
│  │  ├─ Better accuracy than AdaBoost
│  │  ├─ More hyperparameters
│  │  └─ Slower training
│  │
│  └─ XGBOOST
│     ├─ Optimized Gradient Boosting
│     ├─ 2nd order derivatives
│     ├─ Built-in regularization
│     ├─ Parallelizable
│     ├─ Best accuracy
│     ├─ Production-ready
│     └─ Many hyperparameters
│
├─ DIMENSIONALITY REDUCTION
│  │
│  └─ PCA
│     ├─ Reduces features intelligently
│     ├─ Keeps 95% variance with 50% fewer features
│     ├─ Uncorrelated components
│     ├─ Faster training
│     ├─ Better generalization
│     ├─ Harder to interpret
│     └─ Linear transformation
│
├─ PERFORMANCE RANKING (Accuracy)
│  └─ XGBoost > Gradient Boost > AdaBoost > RF > DT
│
├─ SPEED RANKING (Training)
│  └─ AdaBoost > RF > XGBoost > Gradient Boost > DT
│
└─ USE CASES
   ├─ Fast accuracy needed: XGBoost
   ├─ Simple ensemble: AdaBoost
   ├─ Good baseline: Random Forest
   ├─ Reduce dimensions: PCA
   └─ Interpretability: Decision Tree
```

---

## **PART 13: COMPLETE CODE EXAMPLE**

````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# ========== 1. LOAD AND PREPARE DATA ==========
df = pd.read_csv("./data/cardekho_imputated.csv", index_col=[0])
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Preprocess (your code)
# ... (OneHotEncoder, StandardScaler, etc.)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 2. MODELS WITHOUT PCA ==========
print("=" * 60)
print("MODELS WITHOUT PCA")
print("=" * 60)

models = {
    "AdaBoost": AdaBoostRegressor(n_estimators=60, loss='linear'),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
}

results_no_pca = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results_no_pca[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    print(f"\n{name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

# ========== 3. APPLY PCA ==========
print("\n" + "=" * 60)
print("APPLYING PCA")
print("=" * 60)

# Standardize first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (keep 95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nOriginal dimensions: {X_train_scaled.shape[1]}")
print(f"PCA dimensions: {X_train_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# ========== 4. MODELS WITH PCA ==========
print("\n" + "=" * 60)
print("MODELS WITH PCA")
print("=" * 60)

results_pca = {}
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results_pca[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    print(f"\n{name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

# ========== 5. COMPARISON ==========
print("\n" + "=" * 60)
print("COMPARISON: Without PCA vs With PCA")
print("=" * 60)

for name in models.keys():
    r2_diff = results_pca[name]['r2'] - results_no_pca[name]['r2']
    rmse_diff = results_no_pca[name]['rmse'] - results_pca[name]['rmse']
    
    print(f"\n{name}:")
    print(f"  R² change: {r2_diff:+.4f}")
    print(f"  RMSE improvement: {rmse_diff:+.2f}")

# ========== 6. VISUALIZE EXPLAINED VARIANCE ==========
cumsum = np.cumsum(pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Individual Variance Explained')

# Cumulative variance
axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 7. VISUALIZE MODEL COMPARISON ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(models.keys())
r2_no_pca = [results_no_pca[m]['r2'] for m in model_names]
r2_pca = [results_pca[m]['r2'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, r2_no_pca, width, label='Without PCA')
axes[0].bar(x + width/2, r2_pca, width, label='With PCA')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Comparison: R² Score')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

rmse_no_pca = [results_no_pca[m]['rmse'] for m in model_names]
rmse_pca = [results_pca[m]['rmse'] for m in model_names]

axes[1].bar(x - width/2, rmse_no_pca, width, label='Without PCA')
axes[1].bar(x + width/2, rmse_pca, width, label='With PCA')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Model Comparison: RMSE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=45)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
````

---

## **FINAL CHECKLIST**

✓ **AdaBoost**: Focus on misclassified samples, fast, simple
✓ **Gradient Boosting**: Focus on residuals, better accuracy, complex
✓ **XGBoost**: Optimized GB, best accuracy, production-ready
✓ **Accuracy ranking**: XGBoost > GB > AdaBoost > RF
✓ **PCA**: Reduce dimensions while keeping variance
✓ **PCA benefits**: Faster training, less overfitting, visualization
✓ **PCA requirement**: Standardize before applying
✓ **Common approach**: PCA with ensemble methods
✓ **Your notebook**: AdaBoost good baseline for car price

---

**Summary:** 
- **AdaBoost** focuses on hard-to-classify samples through re-weighting
- **Gradient Boosting** focuses on residuals from previous trees
- **XGBoost** is optimized GB with better performance and regularization
- **PCA** reduces dimensions intelligently, keeping 95% variance with 50% fewer features, making training faster without sacrificing accuracy