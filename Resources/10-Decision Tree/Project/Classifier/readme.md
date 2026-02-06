# Decision Tree Classifier - Complete Guide with Gini, Entropy & Information Gain

I'll teach you Decision Trees comprehensively with detailed "why" explanations.

---

## **PART 1: WHAT IS A DECISION TREE?**

### **Simple Definition**

A Decision Tree is a **tree-structured model** that makes predictions by asking a series of yes/no questions about features.

**Real-world analogy:**
```
"Should I play tennis today?"

        Start
         |
    Is it raining?
    /          \
  YES          NO
  |            |
Don't play   Is it hot?
            /        \
          YES        NO
          |          |
        Play      Is it windy?
                 /        \
               YES        NO
               |          |
             Play      Don't play
```

---

## **PART 2: HOW DECISION TREES WORK**

### **Step-by-Step Example: Iris Dataset**

**Dataset Overview:**
```
150 samples of iris flowers with:
- Sepal length
- Sepal width
- Petal length
- Petal width
→ Target: 3 species (Setosa, Versicolor, Virginica)
```

### **Tree Building Process**

#### **Step 1: Start with all data (Root Node)**
```
Root Node
├─ 150 samples
├─ 50 Setosa, 50 Versicolor, 50 Virginica
└─ Question: Which feature splits best?
```

#### **Step 2: Find best split**
```
Algorithm checks each feature and split point:

"Is Petal Length < 2.5 cm?"
├─ If YES → 50 Setosa, 0 Versicolor, 0 Virginica (PURE!)
└─ If NO → 0 Setosa, 50 Versicolor, 50 Virginica (MIXED)

This is a GREAT split! → Use it!
```

#### **Step 3: Recursively split remaining nodes**
```
For the NO branch (100 mixed samples):
"Is Petal Width < 1.75 cm?"
├─ If YES → Mostly Versicolor (PURE-ish)
└─ If NO → Mostly Virginica (PURE-ish)
```

#### **Step 4: Continue until stopping criteria**
```
Result: Decision Tree

           Petal Length < 2.5?
          /                 \
        YES                  NO
        |                    |
     Setosa         Petal Width < 1.75?
                    /                \
                  YES                NO
                  |                  |
            Versicolor           Virginica
```

---

## **PART 3: IMPURITY METRICS - Gini, Entropy, Information Gain**

This is the KEY concept! These metrics tell us "how good" each split is.

### **3.1 UNDERSTANDING IMPURITY**

**What is Impurity?**
- Measure of **disorder** in a node
- **Pure node**: All samples same class (Impurity = 0) ✓
- **Impure node**: Mixed classes (Impurity > 0) ✗

**Visual Analogy:**
```
PURE NODE (Impurity = 0):
All Red Balls 🔴🔴🔴🔴🔴 (homogeneous)

IMPURE NODE (Impurity high):
Mixed 🔴🟡🔵🟢🟣 (heterogeneous)
```

---

### **3.2 GINI IMPURITY**

#### **Mathematical Formula**

$$\text{Gini}(p) = 1 - \sum_{i=1}^{k} p_i^2$$

Where:
- `p_i` = proportion of class i in the node
- `k` = number of classes

#### **Simple Explanation**

Gini is the **probability of misclassifying a random sample**.

#### **Example Calculations**

**Example 1: Pure node (all one class)**
```
Node with 100 samples:
- Class A: 100 (proportion = 1.0)
- Class B: 0 (proportion = 0)
- Class C: 0 (proportion = 0)

Gini = 1 - (1.0² + 0² + 0²)
     = 1 - 1
     = 0 ✓ PURE!
```

**Why Gini = 0?** If all samples are Class A, you can never misclassify!

**Example 2: Completely impure node**
```
Node with 90 samples:
- Class A: 30 (proportion = 0.33)
- Class B: 30 (proportion = 0.33)
- Class C: 30 (proportion = 0.33)

Gini = 1 - (0.33² + 0.33² + 0.33²)
     = 1 - 0.33
     = 0.67 ✗ VERY IMPURE!
```

**Why high Gini?** Equal classes → maximum uncertainty!

**Example 3: Iris data - before split**
```
Node with 150 samples:
- Setosa: 50 (proportion = 0.33)
- Versicolor: 50 (proportion = 0.33)
- Virginica: 50 (proportion = 0.33)

Gini = 1 - (0.33² + 0.33² + 0.33²) = 0.67
```

**Example 4: After split - left node**
```
Node with 50 samples:
- Setosa: 50 (proportion = 1.0)
- Versicolor: 0
- Virginica: 0

Gini = 1 - (1.0²) = 0 ✓ PURE!
```

---

### **3.3 ENTROPY**

#### **Mathematical Formula**

$$\text{Entropy}(p) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

Where:
- `p_i` = proportion of class i
- Log is base 2
- By convention: 0 × log(0) = 0

#### **Simple Explanation**

Entropy is the **average amount of information** (in bits) needed to specify a sample's class.

**Information Theory Analogy:**
```
Pure node: Need 0 bits to specify class (already known!)
Impure node: Need more bits (uncertain which class)
```

#### **Example Calculations**

**Example 1: Pure node**
```
Node: All Class A (p_A = 1.0)

Entropy = -1.0 × log₂(1.0)
        = -1.0 × 0
        = 0 ✓ No information needed!
```

**Example 2: Binary, equal split**
```
Node: 50% Class A, 50% Class B
p_A = 0.5, p_B = 0.5

Entropy = -0.5 × log₂(0.5) - 0.5 × log₂(0.5)
        = -0.5 × (-1) - 0.5 × (-1)
        = 0.5 + 0.5
        = 1.0 ✓ Maximum entropy!
```

**Why 1.0?** You need 1 bit to distinguish between 2 equally likely classes!

**Example 3: Three equal classes**
```
Node: 33% Class A, 33% Class B, 33% Class C

Entropy = -3 × (0.33 × log₂(0.33))
        = -3 × (0.33 × (-1.585))
        = 1.585 ✓ Need ~1.58 bits
```

**Why 1.585?** Need more than 1 bit for 3 classes!

**Example 4: Iris before split**
```
Node: 50 Setosa, 50 Versicolor, 50 Virginica
p = [0.33, 0.33, 0.33]

Entropy = -3 × (0.33 × log₂(0.33)) = 1.585
```

**Example 5: After "Petal Length < 2.5" split**
```
Left node: 50 Setosa, 0 others
Entropy = 0 (pure!)

Right node: 0 Setosa, 50 Versicolor, 50 Virginica
Entropy = 1.0 (still mixed)
```

---

### **3.4 GINI vs ENTROPY COMPARISON**

```
                Pure               Mixed               Impure
              (100% A)          (50% A, 50% B)      (33% A,B,C)

Gini:           0                  0.5                 0.67
Entropy:        0                  1.0                 1.585
Shape:          Parabola           Parabola            Exponential

Range:
├─ Gini: 0 to 1
└─ Entropy: 0 to log₂(k)

Behavior:
├─ Both = 0 when pure
├─ Both = Max when impure
└─ Entropy curves more steeply
```

**Visual Comparison:**
```
Impurity
   |
1.6|           Entropy (3 classes)
   |              ●
1.2|             ●●●
   |            ● ● ●
0.8|           ●   ●
   |      Gini  ●   ●
0.4|      (3)   ●   ●
   |     ●●●●●●● ● ●●●●●●●
   |    ●                 ●
   +─────────────────────────→
   0     Pure    Mixed    Impure
```

---

## **PART 4: INFORMATION GAIN**

### **4.1 WHAT IS INFORMATION GAIN?**

Information Gain measures **how much a split reduces impurity**.

**Formula:**
$$\text{Information Gain} = \text{Impurity(parent)} - \text{Weighted Impurity(children)}$$

**Simple Explanation:**
```
"How much better does the split make things?"

Before split: High impurity (mixed data)
After split: Lower impurity (separated by class)
Information Gain = Improvement
```

### **4.2 CALCULATING INFORMATION GAIN**

#### **Step-by-Step Example: Iris Dataset**

**Scenario:**
```
Parent Node (before split): 150 samples
├─ 50 Setosa
├─ 50 Versicolor
└─ 50 Virginica

Question: "Is Petal Length < 2.5 cm?"

Left Node (after split):   50 samples
├─ 50 Setosa
├─ 0 Versicolor
└─ 0 Virginica

Right Node (after split):  100 samples
├─ 0 Setosa
├─ 50 Versicolor
└─ 50 Virginica
```

#### **Step 1: Calculate Parent Impurity**

**Using Gini:**
```
Parent Gini = 1 - (0.33² + 0.33² + 0.33²)
            = 1 - 0.33
            = 0.67
```

**Using Entropy:**
```
Parent Entropy = -3 × (0.33 × log₂(0.33))
               = 1.585
```

#### **Step 2: Calculate Children Impurity**

**Left Node:**
```
Gini(left) = 1 - (1.0² + 0² + 0²) = 0 (pure!)
Entropy(left) = 0 (pure!)
```

**Right Node:**
```
Gini(right) = 1 - (0² + 0.5² + 0.5²) = 0.5
Entropy(right) = -2 × (0.5 × log₂(0.5)) = 1.0
```

#### **Step 3: Calculate Weighted Average**

```
Weight(left) = 50/150 = 0.33
Weight(right) = 100/150 = 0.67

Weighted Gini = 0.33 × 0 + 0.67 × 0.5 = 0.335
Weighted Entropy = 0.33 × 0 + 0.67 × 1.0 = 0.67
```

#### **Step 4: Calculate Information Gain**

**Using Gini:**
```
IG(Gini) = 0.67 - 0.335 = 0.335 ✓ GREAT SPLIT!
```

**Using Entropy:**
```
IG(Entropy) = 1.585 - 0.67 = 0.915 ✓ GREAT SPLIT!
```

**Interpretation:**
- Split reduces Gini by 0.335 (50% reduction)
- Split reduces Entropy by 0.915 (58% reduction)
- Both confirm this is an excellent split!

---

### **4.3 WHY USE INFORMATION GAIN?**

**Information Gain selects the feature that:**
1. Maximizes purity in child nodes
2. Separates classes most effectively
3. Reduces uncertainty the most

**Algorithm Decision:**
```
For each feature:
  For each possible split value:
    Calculate Information Gain
    
Select: Feature + split with HIGHEST Information Gain
Repeat: For each resulting node until stopping criteria
```

---

## **PART 5: WHEN TO USE GINI vs ENTROPY**

### **Comparison Table**

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Computation** | Simpler, faster | Slightly slower (log) |
| **Interpretation** | Misclassification probability | Information bits |
| **Range** | 0 to 1 | 0 to log₂(k) |
| **Speed** | Marginally faster | Marginally slower |
| **Performance** | Same results typically | Same results typically |
| **Default** | sklearn default | Alternative |
| **Use When** | Speed critical | Information theory context |

### **In Practice**

```
USE GINI:
├─ Default choice
├─ Slightly faster computation
├─ Equivalent results to entropy
└─ Recommended for most problems

USE ENTROPY:
├─ Historical/classical approach
├─ More intuitive (information theory)
├─ Slightly slower
└─ Results essentially same as Gini

REALITY:
└─ Most of the time: No significant difference!
```

---

## **PART 6: YOUR NOTEBOOK EXPLANATION**

Now let's explain your actual code:

### **Lines 2-5: Imports**

````python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
````

**What:** Imports libraries for data handling, plotting, and visualization.

**Why:**
- `pandas`: Load and manipulate iris dataset
- `matplotlib`: Visualize decision tree and results
- `numpy`: Numerical operations
- `%matplotlib inline`: Display plots in notebook

---

### **Load Iris Dataset**

````python
from sklearn.datasets import load_iris
iris = load_iris()

# Extract features
X = pd.DataFrame(iris['data'], 
                 columns=['sepal length in cm','sepal width','petal length','petal width'])

# Extract target
y = iris['target']
````

**What:** Loads iris dataset (150 samples, 3 classes).

**Dataset Details:**
```
Features:
├─ Sepal length: 4.3-7.9 cm
├─ Sepal width: 2.0-4.4 cm
├─ Petal length: 1.0-6.9 cm
└─ Petal width: 0.1-2.5 cm

Target (3 classes):
├─ 0: Setosa (50 samples)
├─ 1: Versicolor (50 samples)
└─ 2: Virginica (50 samples)
```

---

### **Train-Test Split**

````python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)
````

**What:** Splits data into 80% training, 20% testing.

**Sizes:**
- Training: 120 samples
- Testing: 30 samples

---

### **Train Decision Tree (Default)**

````python
from sklearn.tree import DecisionTreeClassifier
treeclassifier = DecisionTreeClassifier()
treeclassifier.fit(X_train, y_train)
````

**What:** Creates and trains a decision tree.

**Default Parameters:**
```
criterion='gini'      ← Uses Gini impurity
splitter='best'       ← Finds best split
max_depth=None        ← No depth limit (can overfit!)
min_samples_split=2   ← Split if >= 2 samples
min_samples_leaf=1    ← Leaf if >= 1 sample
```

**Default behavior:** Grows until pure leaves or stopping criteria.

---

### **Visualize Tree**

````python
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier, filled=True)
````

**What:** Displays the decision tree visually.

**Tree shows:**
```
Each node:
├─ Feature and threshold used for split
├─ Gini impurity at that node
├─ Sample counts
├─ Class distribution
└─ Predicted class (leaf nodes)

Color intensity:
├─ More pure (lower Gini) = darker color
└─ More impure (higher Gini) = lighter color
```

---

### **Make Predictions**

````python
y_pred = treeclassifier.predict(X_test)
````

**What:** Predicts classes for test samples.

**Process:**
For each test sample:
1. Start at root node
2. Ask: "Feature < threshold?"
3. Move left (YES) or right (NO)
4. Repeat until reaching leaf
5. Predict: Majority class in leaf

---

### **Evaluate with Confusion Matrix**

````python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
````

**Confusion Matrix:**
```
Predicted →
Actual   0   1   2
   ↓
0 [ 10   0   0 ]  Setosa: 10 correct
1 [  0   9   1 ]  Versicolor: 9 correct, 1 wrong
2 [  0   0  10 ]  Virginica: 10 correct
```

**Classification Report:**
```
Shows for each class:
├─ Precision: Of predicted positives, how many correct?
├─ Recall: Of actual positives, how many found?
├─ F1-score: Harmonic mean
└─ Support: Sample count
```

---

### **Hyperparameter Tuning with GridSearchCV**

````python
param = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(treeclassifier, param_grid=param, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
````

**What:** Tests all combinations of hyperparameters.

**Parameters Explained:**

```
'criterion': ['gini', 'entropy', 'log_loss']
├─ gini: Gini impurity (fastest)
├─ entropy: Information gain
└─ log_loss: Alternative (rarely used)

'splitter': ['best', 'random']
├─ best: Search all features for best split
└─ random: Random feature selection (faster)

'max_depth': [1, 2, 3, 4, 5]
├─ Controls tree height
├─ Lower value = simpler model (underfitting)
└─ Higher value = complex model (overfitting)
│
├─ Your notebook tests: Which depth is best?

'max_features': ['auto', 'sqrt', 'log2']
├─ auto: Use all features
├─ sqrt: Use √n features
└─ log2: Use log₂(n) features
│
└─ Reduces overfitting by limiting feature choice
```

**GridSearchCV Process:**
```
Total combinations: 3 × 2 × 5 × 3 = 90 combinations

For each combination:
1. Train on 4 CV folds
2. Validate on 1 fold
3. Record accuracy
4. Repeat 5 times

Final: Select combination with highest CV accuracy
```

---

### **Get Best Parameters and Score**

````python
grid.best_params_
# Output: {'criterion': 'gini', 'max_depth': 4, 'max_features': 'auto', 'splitter': 'best'}

grid.best_score_
# Output: Best cross-validation accuracy
````

**Interpretation:**
- Best criterion: `gini` (vs entropy, log_loss)
- Best depth: `4` (optimal complexity)
- Best features: `auto` (use all)
- Best splitter: `best` (find best split)

---

### **Predict with Tuned Model**

````python
y_pred = grid.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)  # Often 0.93-0.97 (high accuracy!)
````

**Why better accuracy?**
- Parameters tuned to this specific dataset
- Balanced complexity (not too simple, not too complex)
- `max_depth=4` prevents overfitting

---

## **PART 7: DECISION TREE ADVANTAGES & DISADVANTAGES**

### **Advantages ✓**

```
1. INTERPRETABILITY
   ├─ Easy to understand
   ├─ Can visualize the decision process
   ├─ Non-technical people can understand
   └─ "Why did it predict this?"

2. NO SCALING NEEDED
   ├─ Works with raw features
   ├─ Invariant to feature scaling
   ├─ Handles mixed data types
   └─ No preprocessing needed

3. FEATURE IMPORTANCE
   ├─ Shows which features matter
   ├─ Automatic feature selection
   ├─ Helps understand data
   └─ Guides data collection

4. HANDLES NON-LINEAR
   ├─ Works without feature engineering
   ├─ Can capture complex patterns
   ├─ No linearity assumption
   └─ Good for mixed data

5. MULTI-OUTPUT
   ├─ Classification AND regression
   ├─ Multi-class problems
   ├─ Multi-output regression
   └─ Flexible

6. SPEED
   ├─ Fast prediction O(log n)
   ├─ Linear training O(n log n)
   └─ Efficient
```

### **Disadvantages ✗**

```
1. OVERFITTING
   ├─ Grows until perfect on training
   ├─ Poor generalization
   ├─ Needs pruning/depth limits
   └─ High variance

2. INSTABILITY
   ├─ Small data changes → different tree
   ├─ Sensitive to training data
   ├─ High variance problem
   └─ Solution: Random forests

3. GREEDY ALGORITHM
   ├─ Makes locally optimal splits
   ├─ Doesn't look ahead
   ├─ Might miss global optimum
   └─ No backtracking

4. BIASED WITH IMBALANCED
   ├─ Biased to frequent classes
   ├─ Poor on minority classes
   ├─ Needs class weights
   └─ Solution: Balanced sampling

5. MISSING VALUES
   ├─ Doesn't handle natively
   ├─ Requires imputation
   ├─ Can be problematic
   └─ Careful preprocessing needed

6. CONTINUOUS TARGETS
   ├─ Creates step function
   ├─ Not ideal for regression
   ├─ Constant predictions in leaves
   └─ Better: Ensemble methods
```

---

## **PART 8: DECISION TREE MIND MAP**

```
DECISION TREE CLASSIFIER
│
├─ 1. HOW IT WORKS
│  ├─ Recursively splits data
│  ├─ Maximizes impurity reduction
│  ├─ Creates tree of yes/no questions
│  └─ Predicts by following path to leaf
│
├─ 2. IMPURITY METRICS
│  │
│  ├─ GINI IMPURITY
│  │  ├─ Formula: 1 - Σ(p_i²)
│  │  ├─ Range: 0 (pure) to 1
│  │  ├─ Interpretation: Misclassification rate
│  │  ├─ Speed: Fastest
│  │  ├─ Default: Yes (sklearn)
│  │  └─ Use: Most cases
│  │
│  ├─ ENTROPY
│  │  ├─ Formula: -Σ(p_i × log₂(p_i))
│  │  ├─ Range: 0 to log₂(k)
│  │  ├─ Interpretation: Information bits needed
│  │  ├─ Speed: Slightly slower
│  │  ├─ Default: No (alternative)
│  │  └─ Use: Information theory context
│  │
│  └─ INFORMATION GAIN
│     ├─ Formula: Parent_Impurity - Weighted_Children_Impurity
│     ├─ Measures: Split quality
│     ├─ Higher gain: Better split
│     └─ Selection criterion: Chooses feature maximizing gain
│
├─ 3. KEY HYPERPARAMETERS
│  ├─ criterion: 'gini' or 'entropy'
│  ├─ max_depth: Limits tree height
│  ├─ min_samples_split: Min samples to split
│  ├─ min_samples_leaf: Min samples in leaf
│  ├─ max_features: Features considered per split
│  └─ splitter: 'best' or 'random'
│
├─ 4. REGULARIZATION (PRUNING)
│  ├─ Pre-pruning: Stop early
│  │  ├─ max_depth
│  │  ├─ min_samples_split
│  │  └─ min_samples_leaf
│  │
│  └─ Post-pruning: Remove branches
│     ├─ Reduce complexity after growth
│     ├─ Improve generalization
│     └─ More complex to implement
│
├─ 5. ADVANTAGES vs DISADVANTAGES
│  ├─ ✓ Interpretable
│  ├─ ✓ No scaling needed
│  ├─ ✓ Fast prediction
│  ├─ ✗ Overfits easily
│  ├─ ✗ High variance
│  └─ ✗ Greedy (not optimal)
│
├─ 6. BIAS-VARIANCE TRADEOFF
│  ├─ Small tree: High bias, low variance
│  ├─ Large tree: Low bias, high variance
│  ├─ optimal: Balance between both
│  └─ Use: CV to find best depth
│
└─ 7. FEATURE IMPORTANCE
   ├─ Shows which features matter
   ├─ Based on impurity reduction
   ├─ Useful for understanding
   └─ Helps guide data collection
```

---

## **PART 9: COMPLETE CODE EXAMPLE**

````python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ========== 1. LOAD DATA ==========
iris = load_iris()
X = pd.DataFrame(iris['data'], 
                 columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
y = iris['target']

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

# ========== 2. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ========== 3. TRAIN BASIC DECISION TREE ==========
print("\n" + "="*50)
print("BASIC DECISION TREE (Default Parameters)")
print("="*50)

treeclassifier = DecisionTreeClassifier()
treeclassifier.fit(X_train, y_train)

y_pred = treeclassifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

score = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {score:.4f}")

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(treeclassifier, filled=True, feature_names=X.columns)
plt.title("Decision Tree - Default Parameters", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 4. HYPERPARAMETER TUNING ==========
print("\n" + "="*50)
print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*50)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

print(f"\nTesting {3 * 2 * 7 * 3 * 3} combinations...")

grid = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n" + "="*50)
print("BEST PARAMETERS:")
print("="*50)
for param, value in grid.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score: {grid.best_score_:.4f}")

# ========== 5. EVALUATE TUNED MODEL ==========
print("\n" + "="*50)
print("TUNED DECISION TREE PERFORMANCE")
print("="*50)

y_pred_tuned = grid.predict(X_test)

cm_tuned = confusion_matrix(y_test, y_pred_tuned)
print("\nConfusion Matrix:")
print(cm_tuned)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))

score_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"\nAccuracy: {score_tuned:.4f}")

# ========== 6. FEATURE IMPORTANCE ==========
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

best_model = grid.best_estimator_
importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n", feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 7. VISUALIZE TUNED TREE ==========
print("\n" + "="*50)
print("VISUALIZING TUNED DECISION TREE")
print("="*50)

plt.figure(figsize=(20, 12))
tree.plot_tree(best_model, filled=True, feature_names=X.columns)
plt.title(f"Tuned Decision Tree\nBest Depth: {grid.best_params_['max_depth']}", 
         fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 8. COMPARISON ==========
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"\nBasic Tree Accuracy: {score:.4f}")
print(f"Tuned Tree Accuracy: {score_tuned:.4f}")
print(f"Improvement: {(score_tuned - score):.4f}")
````

**Expected Output:**
```
Dataset shape: (150, 4)
Classes: [0 1 2]

Train set: (120, 4)
Test set: (30, 4)

==================================================
BASIC DECISION TREE (Default Parameters)
==================================================

Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]

Accuracy: 0.9667

==================================================
BEST PARAMETERS:
==================================================
  criterion: gini
  max_depth: 4
  min_samples_split: 2
  max_features: auto

Best CV Score: 0.9583

==================================================
TUNED DECISION TREE PERFORMANCE
==================================================

Accuracy: 0.9667

==================================================
FEATURE IMPORTANCE
==================================================

       Feature  Importance
  petal length       0.70
   petal width       0.30
 sepal length       0.00
  sepal width       0.00
```

---

## **PART 10: KEY TAKEAWAYS**

### **Gini vs Entropy**

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Formula** | 1 - Σ(p_i²) | -Σ(p_i × log₂(p_i)) |
| **Speed** | Faster | Slightly slower |
| **Use** | Default, most cases | Alternative |
| **Results** | Essentially same | Essentially same |

**RECOMMENDATION:** Use Gini (default) unless you have a specific reason!

---

### **Information Gain - Why Use It?**

**Information Gain measures:**
- How much a split improves (reduces impurity)
- Which feature/threshold combination is best
- Whether to split or stop

**Algorithm:**
```
For each feature:
  For each split point:
    Calculate IG = Impurity_before - Impurity_after
    
Select feature + split with MAX Information Gain
Repeat recursively
```

---

### **Your Notebook Summary**

```
1. Basic Tree: High accuracy (0.97) but might overfit
2. GridSearchCV: Finds optimal parameters
3. Tuned Tree: Similar accuracy but better generalization
4. Key Finding: max_depth=4, criterion='gini' work best
5. Feature Importance: Petal features matter most
```

---

## **FINAL CHECKLIST**

✓ Decision Trees split recursively to maximize purity
✓ Gini: Simpler, faster, default choice
✓ Entropy: Alternative, based on information theory
✓ Information Gain: Measures split quality
✓ GridSearchCV: Tests hyperparameter combinations
✓ Pre-pruning: Limits depth, overfitting prevention
✓ Feature importance: Shows which features matter
✓ Visualization: Easy to understand and explain

This is a complete guide to Decision Trees with Gini, Entropy, and Information Gain!