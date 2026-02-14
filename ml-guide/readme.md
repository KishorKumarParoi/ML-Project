# Complete Machine Learning Algorithms Guide

## 🗺️ Machine Learning Mindmap

```
MACHINE LEARNING
│
├── SUPERVISED LEARNING
│   ├── Regression (Continuous Output)
│   │   ├── Linear Regression ⭐⭐⭐
│   │   ├── Ridge/Lasso Regression ⭐⭐
│   │   ├── Polynomial Regression ⭐
│   │   ├── Support Vector Regression (SVR) ⭐⭐
│   │   ├── Decision Tree Regression ⭐⭐
│   │   ├── Random Forest Regression ⭐⭐⭐
│   │   ├── Gradient Boosting (XGBoost, LightGBM) ⭐⭐⭐⭐⭐
│   │   └── Neural Networks ⭐⭐⭐⭐
│   │
│   └── Classification (Categorical Output)
│       ├── Logistic Regression ⭐⭐⭐⭐
│       ├── Naive Bayes ⭐⭐⭐
│       ├── K-Nearest Neighbors (KNN) ⭐⭐
│       ├── Support Vector Machines (SVM) ⭐⭐⭐⭐
│       ├── Decision Trees ⭐⭐⭐
│       ├── Random Forest ⭐⭐⭐⭐⭐
│       ├── Gradient Boosting (XGBoost, LightGBM, CatBoost) ⭐⭐⭐⭐⭐
│       └── Neural Networks ⭐⭐⭐⭐
│
├── UNSUPERVISED LEARNING
│   ├── Clustering
│   │   ├── K-Means ⭐⭐⭐⭐⭐
│   │   ├── Hierarchical Clustering ⭐⭐⭐
│   │   ├── DBSCAN ⭐⭐⭐
│   │   ├── Gaussian Mixture Models (GMM) ⭐⭐
│   │   └── Mean Shift ⭐
│   │
│   └── Dimensionality Reduction
│       ├── PCA (Principal Component Analysis) ⭐⭐⭐⭐⭐
│       ├── t-SNE ⭐⭐⭐⭐
│       ├── UMAP ⭐⭐⭐
│       ├── LDA (Linear Discriminant Analysis) ⭐⭐
│       └── Autoencoders ⭐⭐⭐
│
└── REINFORCEMENT LEARNING
    ├── Q-Learning ⭐⭐⭐
    ├── Deep Q-Networks (DQN) ⭐⭐⭐⭐
    ├── Policy Gradient Methods ⭐⭐⭐
    └── Actor-Critic ⭐⭐⭐⭐
```

---

## 🏆 Most Used & Best Algorithms (Detailed)

### 1. **XGBoost/LightGBM/CatBoost** ⭐⭐⭐⭐⭐

**Why?** Winner of 70%+ Kaggle competitions. Best performance on structured/tabular data.

**When to Use:**
- Structured/tabular datasets
- Medium to large datasets (1K-10M rows)
- When accuracy is critical
- Competition-grade models

**Why It Works:**
- Handles missing values automatically
- Built-in regularization prevents overfitting
- Feature importance insights
- Handles non-linear relationships

**Use Case Example:**
```python
# Credit Card Fraud Detection
from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=99  # Handle imbalanced data
)
model.fit(X_train, y_train)
```
**Real-world:** PayPal fraud detection, Airbnb pricing

---

### 2. **Random Forest** ⭐⭐⭐⭐⭐

**Why?** Robust, versatile, minimal tuning needed.

**When to Use:**
- Feature importance needed
- Small to medium datasets
- When interpretability matters
- Baseline model

**Why It Works:**
- Reduces overfitting through averaging
- Handles outliers well
- Works with mixed data types

**Use Case Example:**
```python
# Customer Churn Prediction
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```
**Real-world:** Banking churn prediction, medical diagnosis

---

### 3. **Logistic Regression** ⭐⭐⭐⭐

**Why?** Fast, interpretable, great baseline.

**When to Use:**
- Binary classification
- Need probability scores
- Linear relationships exist
- Quick prototyping

**Why It Works:**
- Probabilistic output (0-1)
- Coefficient interpretation
- Low computational cost

**Use Case Example:**
```python
# Email Spam Detection
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=100)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
```
**Real-world:** Email spam filters, disease prediction

---

### 4. **K-Means Clustering** ⭐⭐⭐⭐⭐

**Why?** Simple, fast, scalable unsupervised learning.

**When to Use:**
- Customer segmentation
- Document clustering
- Image compression
- Unknown number of groups (use elbow method)

**Why It Works:**
- Efficient on large datasets
- Easy to interpret
- Scales well

**Use Case Example:**
```python
# Customer Segmentation
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
customer_segments = kmeans.fit_predict(customer_data)

# Analyze segments
for i in range(5):
    print(f"Segment {i}: {np.sum(customer_segments == i)} customers")
```
**Real-world:** Netflix user groups, retail customer segments

---

### 5. **Support Vector Machines (SVM)** ⭐⭐⭐⭐

**Why?** Powerful for high-dimensional data.

**When to Use:**
- Image classification
- Text classification
- Small to medium datasets
- Clear margin of separation

**Why It Works:**
- Effective in high dimensions
- Kernel trick for non-linear data
- Memory efficient

**Use Case Example:**
```python
# Image Classification (Handwritten Digits)
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
```
**Real-world:** Face detection, text categorization

---

### 6. **Neural Networks (Deep Learning)** ⭐⭐⭐⭐

**Why?** State-of-the-art for unstructured data.

**When to Use:**
- Image/video processing
- Natural language processing
- Speech recognition
- Large datasets (100K+ samples)

**Why It Works:**
- Learns hierarchical features
- Handles complex patterns
- Transfer learning available

**Use Case Example:**
```python
# Image Classification with CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```
**Real-world:** Tesla autopilot, ChatGPT, medical image analysis

---

### 7. **PCA (Dimensionality Reduction)** ⭐⭐⭐⭐⭐

**Why?** Essential for high-dimensional data preprocessing.

**When to Use:**
- 100+ features
- Visualization of high-dim data
- Noise reduction
- Before applying other algorithms

**Why It Works:**
- Removes multicollinearity
- Reduces computation time
- Captures maximum variance

**Use Case Example:**
```python
# Reduce 1000 features to 50
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_train)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```
**Real-world:** Gene expression analysis, image compression

---

## 📊 Quick Decision Guide

| Problem Type | Algorithm Choice | Why? |
|--------------|-----------------|------|
| **Structured Data + High Accuracy** | XGBoost/LightGBM | Best performance on tabular data |
| **Structured Data + Interpretability** | Random Forest | Feature importance + robust |
| **Binary Classification + Speed** | Logistic Regression | Fast + probabilistic |
| **High-Dimensional Data** | SVM | Kernel trick for non-linearity |
| **Image/Text/Audio** | Neural Networks | Learns hierarchical features |
| **Customer Segmentation** | K-Means | Simple + scalable |
| **Anomaly Detection** | Isolation Forest/One-Class SVM | Specialized for outliers |
| **Time Series Forecasting** | LSTM/ARIMA/Prophet | Handles temporal dependencies |

---

## 🎯 Algorithm Selection Flowchart

```
START
  ↓
Do you have labeled data?
  ├─ YES → SUPERVISED
  │   ↓
  │   Is output continuous or categorical?
  │   ├─ Continuous → REGRESSION
  │   │   ├─ Linear relationship? → Linear Regression
  │   │   ├─ Need regularization? → Ridge/Lasso
  │   │   └─ Complex patterns? → XGBoost/Random Forest
  │   │
  │   └─ Categorical → CLASSIFICATION
  │       ├─ Binary + Simple? → Logistic Regression
  │       ├─ Text/Image? → Neural Networks
  │       └─ Tabular + Accuracy? → XGBoost/Random Forest
  │
  └─ NO → UNSUPERVISED
      ↓
      What's your goal?
      ├─ Group similar items? → K-Means/DBSCAN
      ├─ Reduce dimensions? → PCA/t-SNE
      └─ Find patterns? → Association Rules
```

---

## 💡 Key Takeaways

1. **Start Simple:** Logistic Regression/Linear Regression for baseline
2. **Tree-Based for Tabular:** XGBoost/Random Forest dominate structured data
3. **Deep Learning for Unstructured:** CNN for images, RNN/LSTM for sequences
4. **Always Preprocess:** PCA for dimensionality, scaling for distance-based algorithms
5. **Ensemble Methods Win:** Combining models often beats single algorithms

Would you like me to dive deeper into any specific algorithm or create a comparison notebook?

# 🧠 Complete Machine Learning Algorithms - Deep Dive

---

## 📚 TABLE OF CONTENTS
1. Supervised Learning - Regression
2. Supervised Learning - Classification
3. Unsupervised Learning
4. Ensemble Methods
5. Deep Learning
6. Real-World Implementation Guide

---

# 🎯 REGRESSION ALGORITHMS

## 1. Linear Regression ⭐⭐⭐

### **What is it?**
Finds the best-fit straight line through data points to predict continuous values.

### **Mathematical Concept:**
```
y = mx + b
where:
- y = predicted value
- m = slope (coefficient)
- x = input feature
- b = intercept
```

### **Why Use It?**
✅ **Simple & Interpretable:** Easy to understand relationships  
✅ **Fast Training:** Works with large datasets  
✅ **Baseline Model:** Starting point for any regression task  
✅ **Feature Importance:** Coefficients show impact  

❌ **Limitations:**
- Assumes linear relationship
- Sensitive to outliers
- Can't capture complex patterns

### **When to Use:**
- **Relationship is linear** (scatter plot shows straight line)
- **Quick predictions needed** (milliseconds)
- **Interpretability required** (explain to stakeholders)
- **Simple datasets** (few features, clear trends)

### **Detailed Example: House Price Prediction**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
house_data = pd.DataFrame({
    'square_feet': np.random.randint(800, 3500, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'age': np.random.randint(0, 50, 1000)
})

# Price formula: 100 per sqft + 10000 per bedroom - 500 per year
house_data['price'] = (
    100 * house_data['square_feet'] + 
    10000 * house_data['bedrooms'] - 
    500 * house_data['age'] + 
    np.random.normal(0, 10000, 1000)  # Add noise
)

# Prepare data
X = house_data[['square_feet', 'bedrooms', 'age']]
y = house_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${mean_squared_error(y_test, y_pred, squared=False):,.2f}")
print("\nCoefficients (Feature Importance):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: ${coef:,.2f}")
print(f"  Intercept: ${model.intercept_:,.2f}")

# Interpretation Example
print("\n" + "=" * 50)
print("INTERPRETATION:")
print("=" * 50)
print(f"Each additional square foot adds ${model.coef_[0]:.2f}")
print(f"Each bedroom adds ${model.coef_[1]:,.2f}")
print(f"Each year of age reduces value by ${abs(model.coef_[2]):,.2f}")
```

**Real-World Applications:**
- 🏠 **Real Estate:** Zillow home price estimates
- 📈 **Stock Market:** Predicting stock prices based on indicators
- 🌡️ **Weather:** Temperature prediction
- 💰 **Sales Forecasting:** Revenue prediction

---

## 2. Ridge & Lasso Regression ⭐⭐⭐

### **What is it?**
Linear regression with **regularization** to prevent overfitting.

### **The Problem They Solve:**
Regular linear regression can overfit when:
- Too many features
- Features are correlated (multicollinearity)
- Model is too complex

### **Ridge (L2 Regularization):**
```
Cost = RSS + λ × (sum of squared coefficients)
```
- Shrinks coefficients but keeps all features
- Good when all features are somewhat useful

### **Lasso (L1 Regularization):**
```
Cost = RSS + λ × (sum of absolute coefficients)
```
- Can shrink coefficients to exactly zero
- Performs feature selection automatically

### **Why Use Them?**
✅ **Prevents Overfitting:** Especially with many features  
✅ **Handles Multicollinearity:** Ridge is excellent for correlated features  
✅ **Feature Selection:** Lasso automatically removes irrelevant features  
✅ **Better Generalization:** Works better on unseen data  

### **Detailed Example: Marketing Campaign ROI**

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Create marketing data with many correlated features
np.random.seed(42)
marketing_data = pd.DataFrame({
    'tv_ads': np.random.randint(1000, 50000, 500),
    'radio_ads': np.random.randint(500, 20000, 500),
    'newspaper_ads': np.random.randint(100, 10000, 500),
    'social_media_spend': np.random.randint(500, 30000, 500),
    'email_campaigns': np.random.randint(10, 100, 500),
    'influencer_marketing': np.random.randint(0, 15000, 500),
    'seo_spend': np.random.randint(500, 10000, 500),
    'content_marketing': np.random.randint(100, 5000, 500)
})

# Add correlated features (multicollinearity)
marketing_data['tv_ads_squared'] = marketing_data['tv_ads'] ** 2
marketing_data['radio_tv_interaction'] = marketing_data['tv_ads'] * marketing_data['radio_ads']

# ROI formula (only some features actually matter)
marketing_data['roi'] = (
    0.3 * marketing_data['tv_ads'] + 
    0.2 * marketing_data['social_media_spend'] +
    0.1 * marketing_data['email_campaigns'] +
    np.random.normal(0, 5000, 500)
)

X = marketing_data.drop('roi', axis=1)
y = marketing_data['roi']

# Scale features (important for regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1),
    'Ridge (α=10)': Ridge(alpha=10),
    'Lasso (α=1)': Lasso(alpha=1),
    'Lasso (α=10)': Lasso(alpha=10),
    'ElasticNet': ElasticNet(alpha=1, l1_ratio=0.5)
}

print("=" * 70)
print("REGULARIZATION COMPARISON")
print("=" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n{name}")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²:  {test_score:.4f}")
    print(f"  Overfitting: {(train_score - test_score):.4f}")
    
    # Show which features Lasso eliminated
    if 'Lasso' in name:
        non_zero = np.sum(model.coef_ != 0)
        print(f"  Features used: {non_zero}/{len(model.coef_)}")

# Feature importance with Lasso
lasso_best = Lasso(alpha=1)
lasso_best.fit(X_train, y_train)

print("\n" + "=" * 70)
print("LASSO FEATURE SELECTION")
print("=" * 70)
for feature, coef in zip(X.columns, lasso_best.coef_):
    if abs(coef) > 0.01:
        print(f"✓ {feature:25s}: {coef:10.2f} (KEPT)")
    else:
        print(f"✗ {feature:25s}: {coef:10.2f} (REMOVED)")
```

**When to Choose:**
- **Ridge:** All features potentially useful, multicollinearity present
- **Lasso:** Want automatic feature selection, sparse solutions
- **ElasticNet:** Combination of both (best of both worlds)

**Real-World Applications:**
- 📊 **Finance:** Portfolio optimization with many assets
- 🧬 **Genomics:** Gene expression with thousands of features
- 📱 **Marketing:** Multi-channel attribution
- 🏥 **Healthcare:** Disease prediction with many biomarkers

---

## 3. Polynomial Regression ⭐⭐

### **What is it?**
Extends linear regression to model non-linear relationships using polynomial features.

### **Mathematical Concept:**
```
Linear:     y = b₀ + b₁x
Quadratic:  y = b₀ + b₁x + b₂x²
Cubic:      y = b₀ + b₁x + b₂x² + b₃x³
```

### **Why Use It?**
✅ **Captures Curves:** Models U-shaped, S-shaped relationships  
✅ **Still Linear Model:** Uses linear regression under the hood  
✅ **Flexible:** Can approximate complex functions  

❌ **Limitations:**
- Can overfit easily (high degrees)
- Extrapolation is dangerous
- Computationally expensive

### **Detailed Example: Learning Curve**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Student learning data (non-linear relationship)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
test_scores = np.array([20, 35, 48, 58, 66, 72, 76, 78, 79, 80])  # Diminishing returns

# Try different polynomial degrees
degrees = [1, 2, 3, 5, 10]

plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees, 1):
    # Create polynomial features
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(study_hours, test_scores)
    
    # Generate smooth curve
    hours_range = np.linspace(0, 12, 100).reshape(-1, 1)
    predictions = poly_model.predict(hours_range)
    
    plt.subplot(2, 3, i)
    plt.scatter(study_hours, test_scores, color='blue', s=100, label='Actual')
    plt.plot(hours_range, predictions, color='red', linewidth=2, label=f'Degree {degree}')
    plt.xlabel('Study Hours')
    plt.ylabel('Test Score')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show overfitting
    train_score = poly_model.score(study_hours, test_scores)
    plt.text(1, 70, f'R² = {train_score:.3f}', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.savefig('polynomial_comparison.png', dpi=150, bbox_inches='tight')
print("Saved polynomial_comparison.png")

print("\n" + "=" * 70)
print("POLYNOMIAL REGRESSION ANALYSIS")
print("=" * 70)
print("\nKey Insights:")
print("• Degree 1 (Linear): Underfitting - too simple")
print("• Degree 2-3: Good fit - captures curve without overfitting")
print("• Degree 5+: Overfitting - memorizes noise")
print("\n💡 Rule of Thumb: Use degree 2-3 for most real-world data")
```

**Real-World Applications:**
- 📈 **Economics:** Marginal utility curves
- 🎯 **Marketing:** Product adoption curves
- 🏋️ **Sports:** Performance vs training intensity
- 🧪 **Chemistry:** Reaction rates

---

# 🎯 CLASSIFICATION ALGORITHMS

## 4. Logistic Regression ⭐⭐⭐⭐⭐

### **What is it?**
Despite the name, it's for **classification**! Predicts probability of belonging to a class.

### **Mathematical Concept:**
```
Sigmoid Function: σ(z) = 1 / (1 + e^(-z))

where z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ

Output: probability between 0 and 1
Decision: if P(y=1) > 0.5 → Class 1, else Class 0
```

### **Why Use It?**
✅ **Probability Output:** Get confidence scores (0.0 to 1.0)  
✅ **Fast & Scalable:** Works with millions of samples  
✅ **Interpretable:** Coefficients show feature impact  
✅ **Regularization Built-in:** L1/L2 available  
✅ **Multi-class Support:** One-vs-Rest or Multinomial  

### **When to Use:**
- Binary or multi-class classification
- Need probability scores
- Linear decision boundary exists
- Baseline model for classification

### **Detailed Example: Credit Card Fraud Detection**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
import seaborn as sns

# Simulate credit card transactions
np.random.seed(42)
n_samples = 10000

# Normal transactions
normal_transactions = pd.DataFrame({
    'amount': np.random.normal(50, 30, int(n_samples * 0.98)),
    'time_since_last': np.random.exponential(2, int(n_samples * 0.98)),
    'merchant_category': np.random.randint(1, 20, int(n_samples * 0.98)),
    'distance_from_home': np.random.gamma(2, 2, int(n_samples * 0.98)),
    'fraud': 0
})

# Fraudulent transactions (different pattern)
fraud_transactions = pd.DataFrame({
    'amount': np.random.normal(200, 100, int(n_samples * 0.02)),  # Higher amounts
    'time_since_last': np.random.exponential(0.5, int(n_samples * 0.02)),  # Rapid succession
    'merchant_category': np.random.randint(15, 20, int(n_samples * 0.02)),  # Specific categories
    'distance_from_home': np.random.gamma(10, 5, int(n_samples * 0.02)),  # Far from home
    'fraud': 1
})

# Combine
transactions = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
transactions = transactions.sample(frac=1).reset_index(drop=True)  # Shuffle

# Prepare data
X = transactions.drop('fraud', axis=1)
y = transactions['fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(
    C=0.1,  # Regularization strength (inverse)
    class_weight='balanced',  # Handle imbalanced data
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("=" * 70)
print("FRAUD DETECTION RESULTS")
print("=" * 70)

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved confusion_matrix.png")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

print(f"\n🎯 ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature Importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Coefficient Analysis)")
print("=" * 70)
for feature, coef in zip(X.columns, model.coef_[0]):
    direction = "increases" if coef > 0 else "decreases"
    print(f"{feature:20s}: {coef:8.4f} ({direction} fraud probability)")

# Probability Analysis
print("\n" + "=" * 70)
print("PROBABILITY THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    
    print(f"\nThreshold = {threshold}")
    print(f"  Precision: {precision:.3f} (How many predicted frauds are real)")
    print(f"  Recall:    {recall:.3f} (How many real frauds we catch)")
    print(f"  F1-Score:  {f1:.3f}")

# Business Impact
print("\n" + "=" * 70)
print("BUSINESS IMPACT")
print("=" * 70)
avg_fraud_amount = transactions[transactions['fraud'] == 1]['amount'].mean()
total_fraud_cases = y_test.sum()
detected_frauds = (y_pred == 1) & (y_test == 1)
detected_count = detected_frauds.sum()

savings = detected_count * avg_fraud_amount
print(f"💰 Average fraud amount: ${avg_fraud_amount:.2f}")
print(f"🚨 Total frauds in test set: {total_fraud_cases}")
print(f"✅ Frauds detected: {detected_count}")
print(f"💵 Estimated savings: ${savings:,.2f}")
```

**Why Probability Matters:**
```python
# Example: Setting custom threshold based on business needs
print("\n" + "=" * 70)
print("CUSTOM THRESHOLD STRATEGY")
print("=" * 70)

# Conservative (catch most frauds, more false alarms)
conservative_pred = (y_pred_proba >= 0.3).astype(int)
print(f"Conservative (threshold=0.3):")
print(f"  Frauds caught: {((conservative_pred == 1) & (y_test == 1)).sum()}/{y_test.sum()}")

# Balanced
balanced_pred = (y_pred_proba >= 0.5).astype(int)
print(f"\nBalanced (threshold=0.5):")
print(f"  Frauds caught: {((balanced_pred == 1) & (y_test == 1)).sum()}/{y_test.sum()}")

# Aggressive (only high-confidence frauds)
aggressive_pred = (y_pred_proba >= 0.8).astype(int)
print(f"\nAggressive (threshold=0.8):")
print(f"  Frauds caught: {((aggressive_pred == 1) & (y_test == 1)).sum()}/{y_test.sum()}")
```

**Real-World Applications:**
- 💳 **Finance:** Credit card fraud (PayPal, Stripe)
- 📧 **Email:** Spam detection (Gmail uses this)
- 🏥 **Healthcare:** Disease diagnosis
- 🎯 **Marketing:** Customer churn prediction
- 🔒 **Security:** Intrusion detection

---

## 5. Support Vector Machines (SVM) ⭐⭐⭐⭐⭐

### **What is it?**
Finds the **optimal hyperplane** that best separates classes with maximum margin.

### **Core Concept:**
```
Goal: Find decision boundary that:
1. Separates classes
2. Maximizes margin (distance to nearest points)
3. Minimizes classification errors

Support Vectors: Data points closest to decision boundary
```

### **The Kernel Trick:**
Transforms data to higher dimensions where linear separation is possible.

```
Linear Kernel:     K(x,y) = x·y
Polynomial:        K(x,y) = (γx·y + r)^d
RBF (Gaussian):    K(x,y) = exp(-γ||x-y||²)
Sigmoid:           K(x,y) = tanh(γx·y + r)
```

### **Why Use It?**
✅ **Powerful:** Works in high dimensions  
✅ **Kernel Trick:** Handles non-linear data  
✅ **Memory Efficient:** Only stores support vectors  
✅ **Robust:** Less prone to overfitting with right kernel  

❌ **Limitations:**
- Slow on large datasets (>100K samples)
- Sensitive to feature scaling
- No probability output (need calibration)
- Hard to interpret

### **When to Use:**
- Medium-sized datasets (1K-100K samples)
- High-dimensional data (text, images)
- Clear margin of separation
- Non-linear relationships

### **Detailed Example: Image Classification (Handwritten Digits)**

```python
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import time

# Load handwritten digits dataset (8x8 pixel images)
digits = load_digits()
X, y = digits.data, digits.target

print("=" * 70)
print("DIGIT CLASSIFICATION WITH SVM")
print("=" * 70)
print(f"Dataset: {X.shape[0]} images of {X.shape[1]} features (8x8 pixels)")
print(f"Classes: {len(np.unique(y))} digits (0-9)")

# Visualize sample digits
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f'Label: {digits.target[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight')
print("\nSaved sample_digits.png")

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('Digit Dataset in 2D (PCA)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('digits_pca.png', dpi=150, bbox_inches='tight')
print("Saved digits_pca.png")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (CRITICAL for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different kernels
kernels = {
    'Linear': SVC(kernel='linear', C=1.0, random_state=42),
    'Polynomial (degree=3)': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
    'RBF (Gaussian)': SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42),
    'Sigmoid': SVC(kernel='sigmoid', gamma='scale', C=1.0, random_state=42)
}

print("\n" + "=" * 70)
print("KERNEL COMPARISON")
print("=" * 70)

results = []
for name, model in kernels.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    n_support = len(model.support_vectors_)
    
    results.append({
        'Kernel': name,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Support Vectors': n_support,
        'Train Time': train_time
    })
    
    print(f"\n{name}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Support Vectors: {n_support}/{len(X_train)} ({n_support/len(X_train)*100:.1f}%)")
    print(f"  Training Time: {train_time:.2f}s")

# Best model: RBF
best_model = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Detailed evaluation
y_pred = best_model.predict(X_test_scaled)

print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT (RBF Kernel)")
print("=" * 70)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Digit Classification')
plt.ylabel('True Digit')
plt.xlabel('Predicted Digit')
plt.savefig('digits_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\nSaved digits_confusion_matrix.png")

# Misclassified examples
misclassified = X_test[y_test != y_pred]
misclassified_true = y_test[y_test != y_pred]
misclassified_pred = y_pred[y_test != y_pred]

print("\n" + "=" * 70)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 70)
print(f"Total misclassified: {len(misclassified)} / {len(y_test)}")

if len(misclassified) > 0:
    plt.figure(figsize=(15, 3))
    for i in range(min(10, len(misclassified))):
        plt.subplot(2, 5, i+1)
        plt.imshow(misclassified[i].reshape(8, 8), cmap='gray')
        plt.title(f'True: {misclassified_true.iloc[i]}\nPred: {misclassified_pred[i]}', 
                 color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_digits.png', dpi=150, bbox_inches='tight')
    print("Saved misclassified_digits.png")
```

### **Hyperparameter Tuning:**

```python
from sklearn.model_selection import GridSearchCV

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
    'kernel': ['rbf', 'poly']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Running grid search (this may take a while)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Best Cross-Validation Score: {grid_search.best_score_:.4f}")
print(f"✅ Test Set Score: {grid_search.score(X_test_scaled, y_test):.4f}")

# Visualize parameter impact
results_df = pd.DataFrame(grid_search.cv_results_)
pivot_table = results_df.pivot_table(
    values='mean_test_score', 
    index='param_C', 
    columns='param_gamma'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('SVM Performance: C vs Gamma')
plt.savefig('svm_heatmap.png', dpi=150, bbox_inches='tight')
print("\nSaved svm_heatmap.png")
```

**Real-World Applications:**
- 🖼️ **Computer Vision:** Face recognition (Facebook photo tagging)
- 📝 **Text Classification:** Sentiment analysis, spam detection
- 🧬 **Bioinformatics:** Protein classification
- 🔊 **Speech Recognition:** Phoneme classification
- 🏥 **Medical:** Cancer detection from images

---

## 6. Decision Trees ⭐⭐⭐⭐

### **What is it?**
Tree-like model that makes decisions by asking questions about features.

### **How It Works:**
```
Is Age > 30?
├─ YES → Is Income > 50K?
│         ├─ YES → High Risk
│         └─ NO  → Medium Risk
└─ NO  → Is Credit Score > 700?
          ├─ YES → Low Risk
          └─ NO  → Medium Risk
```

### **Splitting Criteria:**
**Gini Impurity:**
```
Gini = 1 - Σ(pi²)
where pi = probability of class i
```

**Entropy (Information Gain):**
```
Entropy = -Σ(pi × log2(pi))
```

### **Why Use It?**
✅ **Interpretable:** Easy to visualize and explain  
✅ **No Feature Scaling:** Works with raw data  
✅ **Handles Non-linear:** Naturally captures complex patterns  
✅ **Mixed Data Types:** Categorical + numerical  
✅ **Feature Importance:** Shows which features matter  

❌ **Limitations:**
- Prone to overfitting
- Unstable (small data changes = different tree)
- Biased toward dominant classes

### **When to Use:**
- Need explainable model
- Mixed data types
- Quick baseline
- Feature selection needed

### **Detailed Example: Customer Churn Prediction**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report

# Generate customer data
np.random.seed(42)
n_customers = 5000

customers = pd.DataFrame({
    'age': np.random.randint(18, 70, n_customers),
    'tenure_months': np.random.randint(1, 120, n_customers),
    'monthly_charges': np.random.uniform(20, 150, n_customers),
    'total_charges': np.random.uniform(100, 8000, n_customers),
    'contract_type': np.random.choice(['Month-to-Month', '1 Year', '2 Year'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No'], n_customers),
    'num_support_calls': np.random.randint(0, 10, n_customers)
})

# Churn logic (complex rules)
churn = np.zeros(n_customers)
churn[(customers['contract_type'] == 'Month-to-Month') & 
      (customers['monthly_charges'] > 70)] = 1
churn[(customers['num_support_calls'] > 5) & 
      (customers['tech_support'] == 'No')] = 1
churn[customers['tenure_months'] < 6] = 1
churn = np.random.binomial(1, churn * 0.7 + 0.1)  # Add noise

customers['churned'] = churn

# One-hot encode categorical
customers_encoded = pd.get_dummies(customers, 
                                   columns=['contract_type', 'internet_service', 'tech_support'],
                                   drop_first=True)

X = customers_encoded.drop('churned', axis=1)
y = customers_encoded['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Tree
tree = DecisionTreeClassifier(
    max_depth=5,           # Limit depth to prevent overfitting
    min_samples_split=100, # Minimum samples to split
    min_samples_leaf=50,   # Minimum samples in leaf
    random_state=42
)

tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - DECISION TREE")
print("=" * 70)
print(f"\nTest Accuracy: {tree.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=X.columns,
          class_names=['Stayed', 'Churned'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Customer Churn', fontsize=16)
plt.savefig('decision_tree_visual.png', dpi=150, bbox_inches='tight')
print("\nSaved decision_tree_visual.png")

# Text representation
tree_rules = export_text(tree, feature_names=list(X.columns))
print("\n" + "=" * 70)
print("DECISION RULES (Text Format)")
print("=" * 70)
print(tree_rules[:1000])  # Print first 1000 chars

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved feature_importance.png")

# Overfitting analysis
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Train Accuracy', marker='o')
plt.plot(depths, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Overfitting Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved overfitting_analysis.png")

print("\n" + "=" * 70)
print("OVERFITTING INSIGHT")
print("=" * 70)
print(f"Optimal depth (max test accuracy): {depths[np.argmax(test_scores)]}")
print(f"Max test accuracy: {max(test_scores):.4f}")
```

**Business Insights from Tree:**
```python
print("\n" + "=" * 70)
print("ACTIONABLE BUSINESS INSIGHTS")
print("=" * 70)

# Extract decision paths
from sklearn.tree import _tree

def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    paths = []
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_path = path + [f"{name} <= {threshold:.2f}"]
            right_path = path + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_left[node], left_path)
            recurse(tree_.children_right[node], right_path)
        else:
            paths.append((path, tree_.value[node]))
    
    recurse(0, [])
    return paths

rules = get_rules(tree, list(X.columns))

print("\nTop 3 Rules for Churn:")
churn_rules = [(path, val) for path, val in rules if val[0][1] > val[0][0]]
churn_rules.sort(key=lambda x: x[1][0][1], reverse=True)

for i, (path, value) in enumerate(churn_rules[:3], 1):
    churn_count = int(value[0][1])
    print(f"\n{i}. IF {' AND '.join(path[:3])}")
    print(f"   THEN: High churn risk ({churn_count} customers)")
```

**Real-World Applications:**
- 🏦 **Banking:** Loan approval decisions
- 🏥 **Healthcare:** Medical diagnosis trees
- 🎯 **Marketing:** Customer segmentation
- 💼 **HR:** Employee retention
- 🛒 **Retail:** Product recommendation

---

*This is Part 1 of the deep dive. Would you like me to continue with:*
- **Ensemble Methods (Random Forest, XGBoost, etc.)**
- **Unsupervised Learning (K-Means, DBSCAN, PCA)**
- **Deep Learning (Neural Networks, CNN, RNN)**
- **Complete comparison table with use cases**

Let me know which section you want next! 🚀

# 🚀 PART 2: Advanced ML Algorithms Deep Dive

---

# 🌲 ENSEMBLE METHODS (The Champions)

## 7. Random Forest ⭐⭐⭐⭐⭐ (Most Versatile)

### **What is it?**
**"Wisdom of the Crowd"** - Combines multiple decision trees to make better predictions.

### **How It Works:**
```
Training:
1. Create 100-1000 decision trees
2. Each tree trained on random subset of data (Bootstrap)
3. Each split uses random subset of features
4. Each tree votes on prediction

Prediction:
- Classification: Majority vote
- Regression: Average of all trees

Example:
Tree 1: Cat    Tree 2: Cat    Tree 3: Dog    Tree 4: Cat    Tree 5: Cat
Final Prediction: Cat (3 votes vs 2)
```

### **Why It's Amazing:**
✅ **Robust:** Resistant to overfitting (unlike single tree)  
✅ **Accurate:** Often best "out-of-the-box" algorithm  
✅ **Feature Importance:** Shows what matters  
✅ **Handles Missing Values:** Built-in mechanisms  
✅ **No Feature Scaling:** Works with raw data  
✅ **Parallel Training:** Fast on multi-core systems  
✅ **Versatile:** Works for classification & regression  

❌ **Limitations:**
- Large model size (many trees)
- Slower prediction than single tree
- Less interpretable than single tree

### **When to Use:**
- **Default choice** for structured data
- Need reliable baseline quickly
- Don't have time for extensive tuning
- Want feature importance
- Any tabular dataset

### **Detailed Example: Employee Attrition Prediction**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# Generate comprehensive employee dataset
np.random.seed(42)
n_employees = 3000

employees = pd.DataFrame({
    'age': np.random.randint(22, 65, n_employees),
    'years_at_company': np.random.randint(0, 40, n_employees),
    'salary': np.random.randint(30000, 200000, n_employees),
    'satisfaction_score': np.random.uniform(1, 10, n_employees),
    'performance_rating': np.random.uniform(1, 5, n_employees),
    'projects_completed': np.random.randint(0, 50, n_employees),
    'avg_monthly_hours': np.random.randint(120, 300, n_employees),
    'promotions_last_5years': np.random.randint(0, 5, n_employees),
    'department': np.random.choice(['Sales', 'Tech', 'HR', 'Marketing', 'Finance'], n_employees),
    'work_from_home_days': np.random.randint(0, 5, n_employees),
    'commute_distance': np.random.uniform(1, 50, n_employees),
    'has_mentor': np.random.choice([0, 1], n_employees),
    'training_hours': np.random.randint(0, 100, n_employees)
})

# Complex attrition logic
attrition = np.zeros(n_employees)

# Low satisfaction = high attrition
attrition[employees['satisfaction_score'] < 4] += 0.6

# Low salary for experience = attrition
low_paid_experienced = (employees['salary'] < 50000) & (employees['years_at_company'] > 5)
attrition[low_paid_experienced] += 0.5

# Overworked = attrition
attrition[employees['avg_monthly_hours'] > 250] += 0.4

# No promotions = attrition
attrition[employees['promotions_last_5years'] == 0] += 0.3

# Long commute without WFH = attrition
long_commute = (employees['commute_distance'] > 30) & (employees['work_from_home_days'] == 0)
attrition[long_commute] += 0.3

# Convert to binary with noise
employees['attrition'] = np.random.binomial(1, np.clip(attrition, 0, 0.8))

print("=" * 70)
print("EMPLOYEE ATTRITION DATASET")
print("=" * 70)
print(f"Total Employees: {len(employees)}")
print(f"Attrition Rate: {employees['attrition'].mean():.2%}")
print(f"\nFeatures: {employees.shape[1] - 1}")
print(f"Target Distribution:")
print(employees['attrition'].value_counts())

# Encode categorical variables
employees_encoded = pd.get_dummies(employees, columns=['department'], drop_first=True)

X = employees_encoded.drop('attrition', axis=1)
y = employees_encoded['attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Random Forest with default parameters
print("\n" + "=" * 70)
print("1. BASELINE RANDOM FOREST")
print("=" * 70)

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)

print(f"Train Accuracy: {rf_baseline.score(X_train, y_train):.4f}")
print(f"Test Accuracy:  {rf_baseline.score(X_test, y_test):.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-Val ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Detailed predictions
y_pred = rf_baseline.predict(X_test)
y_pred_proba = rf_baseline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
            xticklabels=['Stay', 'Leave'], 
            yticklabels=['Stay', 'Leave'])
plt.title('Confusion Matrix - Employee Attrition')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('rf_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved rf_confusion_matrix.png")

# ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Employee Attrition')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('rf_roc_curve.png', dpi=150, bbox_inches='tight')
print("Saved rf_roc_curve.png")

# Feature Importance Analysis
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['importance'], color='steelblue')
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Importance Score')
plt.title('Top 15 Features Predicting Employee Attrition')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved rf_feature_importance.png")

# Number of Trees Analysis
print("\n" + "=" * 70)
print("2. OPTIMAL NUMBER OF TREES")
print("=" * 70)

n_trees_range = [10, 50, 100, 200, 500, 1000]
train_scores = []
test_scores = []
train_times = []

for n_trees in n_trees_range:
    start = time.time()
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    train_times.append(train_time)
    
    print(f"Trees: {n_trees:4d} | Train: {train_scores[-1]:.4f} | "
          f"Test: {test_scores[-1]:.4f} | Time: {train_time:.2f}s")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(n_trees_range, train_scores, marker='o', label='Train Accuracy')
ax1.plot(n_trees_range, test_scores, marker='s', label='Test Accuracy')
ax1.set_xlabel('Number of Trees')
ax1.set_ylabel('Accuracy')
ax1.set_title('Random Forest: Trees vs Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(n_trees_range, train_times, marker='o', color='green')
ax2.set_xlabel('Number of Trees')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Random Forest: Trees vs Training Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_trees_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved rf_trees_analysis.png")

# Hyperparameter Tuning
print("\n" + "=" * 70)
print("3. HYPERPARAMETER TUNING (Random Search)")
print("=" * 70)

param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions,
    n_iter=20,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Running random search (this will take a moment)...")
random_search.fit(X_train, y_train)

print(f"\n✅ Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n✅ Best Cross-Val ROC-AUC: {random_search.best_score_:.4f}")
print(f"✅ Test Set ROC-AUC: {roc_auc_score(y_test, random_search.predict_proba(X_test)[:, 1]):.4f}")

# Best Model Analysis
best_rf = random_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

print("\n" + "=" * 70)
print("TUNED MODEL PERFORMANCE")
print("=" * 70)
print(classification_report(y_test, y_pred_tuned, target_names=['Stay', 'Leave']))

# Compare baseline vs tuned
print("\n" + "=" * 70)
print("BASELINE vs TUNED COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Metric': ['Train Accuracy', 'Test Accuracy', 'ROC-AUC', 'Precision', 'Recall'],
    'Baseline': [
        rf_baseline.score(X_train, y_train),
        rf_baseline.score(X_test, y_test),
        roc_auc_score(y_test, rf_baseline.predict_proba(X_test)[:, 1]),
        precision_score(y_test, rf_baseline.predict(X_test)),
        recall_score(y_test, rf_baseline.predict(X_test))
    ],
    'Tuned': [
        best_rf.score(X_train, y_train),
        best_rf.score(X_test, y_test),
        roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]),
        precision_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_tuned)
    ]
})
comparison['Improvement'] = comparison['Tuned'] - comparison['Baseline']
print(comparison.to_string(index=False))

# Business Application: Risk Scoring
print("\n" + "=" * 70)
print("BUSINESS APPLICATION: EMPLOYEE RISK SCORING")
print("=" * 70)

# Get probabilities
attrition_probabilities = best_rf.predict_proba(X_test)[:, 1]

# Create risk categories
risk_categories = pd.cut(attrition_probabilities, 
                         bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low Risk', 'Medium Risk', 'High Risk'])

risk_df = pd.DataFrame({
    'Employee_ID': range(len(attrition_probabilities)),
    'Attrition_Probability': attrition_probabilities,
    'Risk_Category': risk_categories,
    'Actual_Attrition': y_test.values
})

print("\nRisk Distribution:")
print(risk_df['Risk_Category'].value_counts().sort_index())

print("\nHigh Risk Employees (>60% probability):")
high_risk = risk_df[risk_df['Risk_Category'] == 'High Risk'].head(10)
print(high_risk.to_string(index=False))

# Action Plan
print("\n" + "=" * 70)
print("RECOMMENDED ACTIONS")
print("=" * 70)

# Get feature values for high-risk employees
high_risk_indices = risk_df[risk_df['Risk_Category'] == 'High Risk'].index
high_risk_features = X_test.iloc[high_risk_indices]

# Top issues
print("\n1. HIGH RISK EMPLOYEES - Common Characteristics:")
for feature in feature_importance.head(5)['feature']:
    avg_value = high_risk_features[feature].mean()
    overall_avg = X[feature].mean()
    diff_pct = ((avg_value - overall_avg) / overall_avg) * 100
    print(f"   • {feature}: {avg_value:.2f} (vs company avg {overall_avg:.2f}, "
          f"{diff_pct:+.1f}%)")

print("\n2. RETENTION STRATEGIES:")
print("   📊 Low Satisfaction → Conduct stay interviews, improve work conditions")
print("   💰 Low Salary → Review compensation packages for experienced staff")
print("   ⏰ Overwork → Enforce work-life balance policies")
print("   🚀 No Promotions → Create clear career progression paths")
print("   🏠 Long Commute → Offer remote work options")

# Model Explanation for Single Employee
print("\n" + "=" * 70)
print("INDIVIDUAL EMPLOYEE PREDICTION EXPLANATION")
print("=" * 70)

sample_employee_idx = high_risk_indices[0]
sample_employee = X_test.iloc[sample_employee_idx]
sample_proba = attrition_probabilities[sample_employee_idx]

print(f"\nEmployee #{sample_employee_idx}")
print(f"Attrition Probability: {sample_proba:.1%}")
print(f"Actual Outcome: {'LEFT' if y_test.iloc[sample_employee_idx] == 1 else 'STAYED'}")

print("\nKey Contributing Factors:")
# Get feature contributions (approximate using feature importance)
employee_features = sample_employee[feature_importance.head(10)['feature']]
print(employee_features.to_string())
```

### **Out-of-Bag (OOB) Score - Free Validation:**

```python
print("\n" + "=" * 70)
print("4. OUT-OF-BAG (OOB) EVALUATION")
print("=" * 70)

rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Enable OOB evaluation
    random_state=42,
    n_jobs=-1
)

rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.4f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.4f}")
print("\n💡 OOB provides validation without separate test set!")
print("   Each tree is tested on data it didn't see during training")
```

**Real-World Random Forest Applications:**
- 🏦 **Banking:** Loan default prediction (Capital One)
- 🏥 **Healthcare:** Disease diagnosis, patient readmission
- 🛒 **E-commerce:** Product recommendation (Amazon)
- 🚗 **Insurance:** Claim prediction, fraud detection
- 📱 **Tech:** User churn prediction (Netflix, Spotify)

---

## 8. XGBoost/LightGBM/CatBoost ⭐⭐⭐⭐⭐ (Kaggle Winners)

### **What is it?**
**Gradient Boosting Machines** - Build trees sequentially, each correcting previous errors.

### **The Difference from Random Forest:**
```
RANDOM FOREST (Parallel):
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┼──→ Average/Vote → Prediction
Tree 4 ──┤
Tree 5 ──┘

GRADIENT BOOSTING (Sequential):
Tree 1 → Errors → Tree 2 → Errors → Tree 3 → ... → Final Prediction
         (learn)         (learn)
```

### **How Gradient Boosting Works:**
```
Step 1: Make initial prediction (average)
Step 2: Calculate errors (residuals)
Step 3: Train tree to predict errors
Step 4: Update predictions: new = old + learning_rate × tree_prediction
Step 5: Repeat steps 2-4 for N trees

Example:
Initial: Predict all houses = $200k
Tree 1: Learns "add $50k if >2000 sqft"
Tree 2: Learns "add $20k if 3+ bedrooms"
Tree 3: Learns "subtract $10k if >20 years old"
Final: $200k + $50k + $20k - $10k = $260k
```

### **XGBoost vs LightGBM vs CatBoost:**

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | Medium | ⚡ Fastest | Medium |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Categorical** | Manual encode | Manual encode | 🎯 Built-in |
| **Memory** | High | 💾 Low | Medium |
| **Overfitting** | Prone | Prone | 🛡️ Resistant |
| **Best For** | General purpose | Large datasets | Categorical data |

### **Why They're Champions:**
✅ **Competition Winner:** 70%+ Kaggle competitions  
✅ **State-of-the-art:** Best for structured/tabular data  
✅ **Feature Engineering:** Handles interactions automatically  
✅ **Missing Values:** Built-in handling  
✅ **Regularization:** Multiple techniques to prevent overfitting  
✅ **Parallel Processing:** GPU support  

### **When to Use:**
- **XGBoost:** General-purpose, proven track record
- **LightGBM:** Large datasets (>100K rows), speed critical
- **CatBoost:** Many categorical features, less tuning

### **Detailed Example: Credit Risk Scoring**

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Generate realistic loan application data
np.random.seed(42)
n_applications = 50000

loans = pd.DataFrame({
    'age': np.random.randint(18, 70, n_applications),
    'income': np.random.lognormal(10.5, 0.5, n_applications),  # Log-normal distribution
    'loan_amount': np.random.randint(1000, 100000, n_applications),
    'credit_score': np.random.randint(300, 850, n_applications),
    'employment_length': np.random.randint(0, 30, n_applications),
    'num_credit_lines': np.random.randint(0, 20, n_applications),
    'debt_to_income': np.random.uniform(0, 1, n_applications),
    'num_delinquencies': np.random.poisson(0.5, n_applications),
    'num_inquiries': np.random.poisson(1, n_applications),
    'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_applications),
    'loan_purpose': np.random.choice(['Debt Consolidation', 'Credit Card', 
                                     'Home Improvement', 'Other'], n_applications),
    'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_applications)
})

# Default probability logic
default_prob = np.zeros(n_applications)
default_prob += (loans['credit_score'] < 600) * 0.4
default_prob += (loans['debt_to_income'] > 0.5) * 0.3
default_prob += (loans['num_delinquencies'] > 2) * 0.3
default_prob += (loans['income'] < 30000) * 0.2
default_prob += ((loans['loan_amount'] / loans['income']) > 3) * 0.25

loans['default'] = np.random.binomial(1, np.clip(default_prob, 0, 0.7))

print("=" * 70)
print("CREDIT RISK DATASET")
print("=" * 70)
print(f"Total Applications: {len(loans):,}")
print(f"Default Rate: {loans['default'].mean():.2%}")
print(f"\nFeatures: {loans.shape[1] - 1}")

# Split categorical and numerical features
cat_features = ['home_ownership', 'loan_purpose', 'state']
num_features = [col for col in loans.columns if col not in cat_features + ['default']]

# Prepare data for XGBoost/LightGBM (need encoding)
loans_encoded = pd.get_dummies(loans, columns=cat_features, drop_first=True)
X_encoded = loans_encoded.drop('default', axis=1)
y = loans_encoded['default']

X_train_enc, X_test_enc, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# Prepare data for CatBoost (handles categorical natively)
X_original = loans.drop('default', axis=1)
cat_indices = [X_original.columns.get_loc(col) for col in cat_features]

X_train_orig, X_test_orig, _, _ = train_test_split(
    X_original, y, test_size=0.25, random_state=42, stratify=y
)

# ===========================
# 1. XGBoost
# ===========================
print("\n" + "=" * 70)
print("1. XGBOOST")
print("=" * 70)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Train with early stopping
eval_set = [(X_train_enc, y_train), (X_test_enc, y_test)]
xgb_model.fit(
    X_train_enc, y_train,
    eval_set=eval_set,
    verbose=False
)

# Evaluate
xgb_pred = xgb_model.predict(X_test_enc)
xgb_proba = xgb_model.predict_proba(X_test_enc)[:, 1]

print(f"Train Accuracy: {xgb_model.score(X_train_enc, y_train):.4f}")
print(f"Test Accuracy:  {xgb_model.score(X_test_enc, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

# Learning curves
results = xgb_model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost: Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('xgb_learning_curves.png', dpi=150, bbox_inches='tight')
print("Saved xgb_learning_curves.png")

# ===========================
# 2. LightGBM
# ===========================
print("\n" + "=" * 70)
print("2. LIGHTGBM")
print("=" * 70)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
lgb_model.fit(
    X_train_enc, y_train,
    eval_set=[(X_test_enc, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)
lgb_time = time.time() - start

lgb_pred = lgb_model.predict(X_test_enc)
lgb_proba = lgb_model.predict_proba(X_test_enc)[:, 1]

print(f"Train Accuracy: {lgb_model.score(X_train_enc, y_train):.4f}")
print(f"Test Accuracy:  {lgb_model.score(X_test_enc, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, lgb_proba):.4f}")
print(f"Training Time: {lgb_time:.2f}s")

# ===========================
# 3. CatBoost
# ===========================
print("\n" + "=" * 70)
print("3. CATBOOST (handles categorical features natively)")
print("=" * 70)

cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False,
    cat_features=cat_indices  # 🎯 Magic happens here!
)

start = time.time()
cat_model.fit(
    X_train_orig, y_train,
    eval_set=(X_test_orig, y_test),
    early_stopping_rounds=10,
    verbose=False
)
cat_time = time.time() - start

cat_pred = cat_model.predict(X_test_orig).flatten()
cat_proba = cat_model.predict_proba(X_test_orig)[:, 1]

print(f"Train Accuracy: {cat_model.score(X_train_orig, y_train):.4f}")
print(f"Test Accuracy:  {cat_model.score(X_test_orig, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, cat_proba):.4f}")
print(f"Training Time: {cat_time:.2f}s")

# ===========================
# COMPARISON
# ===========================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'ROC-AUC': [
        roc_auc_score(y_test, xgb_proba),
        roc_auc_score(y_test, lgb_proba),
        roc_auc_score(y_test, cat_proba)
    ],
    'Precision': [
        precision_score(y_test, xgb_pred),
        precision_score(y_test, lgb_pred),
        precision_score(y_test, cat_pred.astype(int))
    ],
    'Recall': [
        recall_score(y_test, xgb_pred),
        recall_score(y_test, lgb_pred),
        recall_score(y_test, cat_pred.astype(int))
    ],
    'F1-Score': [
        f1_score(y_test, xgb_pred),
        f1_score(y_test, lgb_pred),
        f1_score(y_test, cat_pred.astype(int))
    ]
})

print(comparison.to_string(index=False))

# Feature Importance Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XGBoost
xgb_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[0].barh(range(len(xgb_importance)), xgb_importance['importance'])
axes[0].set_yticks(range(len(xgb_importance)))
axes[0].set_yticklabels(xgb_importance['feature'])
axes[0].set_title('XGBoost Feature Importance')
axes[0].invert_yaxis()

# LightGBM
lgb_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[1].barh(range(len(lgb_importance)), lgb_importance['importance'])
axes[1].set_yticks(range(len(lgb_importance)))
axes[1].set_yticklabels(lgb_importance['feature'])
axes[1].set_title('LightGBM Feature Importance')
axes[1].invert_yaxis()

# CatBoost
cat_importance = pd.DataFrame({
    'feature': X_original.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[2].barh(range(len(cat_importance)), cat_importance['importance'])
axes[2].set_yticks(range(len(cat_importance)))
axes[2].set_yticklabels(cat_importance['feature'])
axes[2].set_title('CatBoost Feature Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('boosting_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved boosting_comparison.png")

# Hyperparameter Tuning Example (XGBoost)
print("\n" + "=" * 70)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5]
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=10,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Running hyperparameter search...")
random_search.fit(X_train_enc, y_train)

print(f"\n✅ Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

best_xgb = random_search.best_estimator_
print(f"\n✅ Best CV ROC-AUC: {random_search.best_score_:.4f}")
print(f"✅ Test ROC-AUC: {roc_auc_score(y_test, best_xgb.predict_proba(X_test_enc)[:, 1]):.4f}")

# Business Impact: Risk-Based Pricing
print("\n" + "=" * 70)
print("BUSINESS APPLICATION: RISK-BASED PRICING")
print("=" * 70)

# Use best model for pricing
default_probabilities = best_xgb.predict_proba(X_test_enc)[:, 1]

# Calculate interest rates based on risk
base_rate = 0.05  # 5% base rate
risk_df = pd.DataFrame({
    'Loan_Amount': X_test_enc['loan_amount'],
    'Default_Probability': default_probabilities,
    'Risk_Category': pd.cut(default_probabilities, 
                           bins=[0, 0.1, 0.3, 0.5, 1.0],
                           labels=['Low', 'Medium', 'High', 'Very High'])
})

# Risk-based interest rates
rate_mapping = {'Low': 0.05, 'Medium': 0.10, 'High': 0.15, 'Very High': 0.25}
risk_df['Interest_Rate'] = risk_df['Risk_Category'].map(rate_mapping)
risk_df['Expected_Loss'] = risk_df['Loan_Amount'] * risk_df['Default_Probability']

print("\nRisk Distribution:")
print(risk_df.groupby('Risk_Category').agg({
    'Loan_Amount': 'mean',
    'Default_Probability': 'mean',
    'Interest_Rate': 'first',
    'Expected_Loss': 'mean'
}).round(2))

print("\n💰 Expected Loss vs Interest Income:")
total_loans = risk_df['Loan_Amount'].sum()
total_expected_loss = risk_df['Expected_Loss'].sum()
total_interest_income = (risk_df['Loan_Amount'] * risk_df['Interest_Rate']).sum()

print(f"   Total Loan Portfolio: ${total_loans:,.2f}")
print(f"   Expected Losses: ${total_expected_loss:,.2f} ({total_expected_loss/total_loans:.2%})")
print(f"   Interest Income: ${total_interest_income:,.2f} ({total_interest_income/total_loans:.2%})")
print(f"   Net Profit: ${total_interest_income - total_expected_loss:,.2f}")
```

**Key Hyperparameters Explained:**

```python
print("\n" + "=" * 70)
print("HYPERPARAMETER GUIDE")
print("=" * 70)

hyperparams = """
🎯 KEY PARAMETERS TO TUNE:

1. n_estimators (number of trees)
   • More trees = better performance BUT slower
   • Use early_stopping to find optimal number
   • Typical: 100-1000

2. learning_rate (eta)
   • Lower = better performance BUT need more trees
   • Typical: 0.01-0.3
   • Rule: learning_rate × n_estimators ≈ constant

3. max_depth
   • Deeper = more complex BUT overfitting risk
   • Typical: 3-10
   • Start with 6

4. subsample
   • Fraction of samples for each tree
   • Prevents overfitting
   • Typical: 0.6-1.0

5. colsample_bytree
   • Fraction of features for each tree
   • Prevents overfitting
   • Typical: 0.6-1.0

6. gamma (min_split_loss)
   • Minimum loss reduction for split
   • Higher = more conservative
   • Typical: 0-5

7. reg_alpha (L1 regularization)
   • Feature selection
   • Typical: 0-10

8. reg_lambda (L2 regularization)
   • Prevents overfitting
   • Typical: 0-10

💡 TUNING STRATEGY:
   1. Fix n_estimators=100, learning_rate=0.1
   2. Tune max_depth, min_child_weight
   3. Tune subsample, colsample_bytree
   4. Tune gamma
   5. Tune regularization (alpha, lambda)
   6. Lower learning_rate, increase n_estimators
"""
print(hyperparams)
```

**Real-World Applications:**
- 🏆 **Kaggle:** 70%+ competition winners use XGBoost/LightGBM
- 💳 **Finance:** Credit scoring (FICO), fraud detection
- 🛒 **E-commerce:** Click-through rate prediction (Alibaba)
- 🚕 **Ride-sharing:** Demand forecasting (Uber, Lyft)
- 📱 **Ad Tech:** Ad click prediction (Google, Facebook)

---

*Shall I continue with:*
- **Unsupervised Learning (K-Means, DBSCAN, PCA, t-SNE)**
- **Deep Learning (Neural Networks, CNN, RNN, LSTM)**
- **Time Series (ARIMA, Prophet, LSTM)**
- **Complete algorithm selection flowchart with code**

Which topic would you like next? 🚀

Similar code found with 1 license type

# 🎓 PART 3: Unsupervised Learning & Complete Guide

---

# 🔍 UNSUPERVISED LEARNING (No Labels!)

Unsupervised learning finds hidden patterns when you don't have target labels.

## 9. K-Means Clustering ⭐⭐⭐⭐⭐ (Most Popular)

### **What is it?**
Divides data into K clusters where each point belongs to nearest cluster center.

### **How It Works (Step-by-Step):**
```
Step 1: Randomly initialize K cluster centers
Step 2: Assign each point to nearest center
Step 3: Update centers as mean of assigned points
Step 4: Repeat steps 2-3 until convergence

Visual Example (K=3):
Initial:        After Iter 1:    After Iter 2:    Converged:
●●●             ●●              ●●               ●●
●●●      →      ●●       →      ●●        →      ●●
●●●             ●●              ●●               ●●

(● = data point, + = cluster center)
```

### **The Algorithm:**
```python
Initialize K random centers: μ₁, μ₂, ..., μₖ

For each iteration:
    # Assignment step
    For each point xᵢ:
        assign xᵢ to nearest center: cᵢ = argmin ||xᵢ - μⱼ||²
    
    # Update step
    For each center μⱼ:
        μⱼ = mean of all points assigned to j
    
    # Check convergence
    If centers don't change, STOP
```

### **Why Use It?**
✅ **Simple & Fast:** O(nkd) complexity  
✅ **Scalable:** Works with millions of points  
✅ **Interpretable:** Easy to understand clusters  
✅ **Versatile:** Works for many domains  
✅ **Foundation:** Base for many advanced algorithms  

❌ **Limitations:**
- Must specify K beforehand
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

### **When to Use:**
- Customer segmentation
- Image compression
- Document clustering
- Any exploratory data analysis

### **Detailed Example: Customer Segmentation**

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate customer purchase data
np.random.seed(42)
n_customers = 500

customers = pd.DataFrame({
    'annual_income': np.concatenate([
        np.random.normal(40000, 10000, 150),      # Segment 1: Low income
        np.random.normal(80000, 15000, 200),      # Segment 2: Middle income
        np.random.normal(150000, 20000, 150)      # Segment 3: High income
    ]),
    'annual_spending': np.concatenate([
        np.random.normal(5000, 2000, 150),        # Segment 1: Low spenders
        np.random.normal(15000, 3000, 200),       # Segment 2: Medium spenders
        np.random.normal(40000, 5000, 150)        # Segment 3: High spenders
    ]),
    'purchase_frequency': np.concatenate([
        np.random.normal(10, 3, 150),             # Segment 1: Infrequent
        np.random.normal(25, 5, 200),             # Segment 2: Regular
        np.random.normal(50, 8, 150)              # Segment 3: Frequent
    ]),
    'customer_lifetime_value': np.concatenate([
        np.random.normal(20000, 8000, 150),       # Segment 1: Low CLV
        np.random.normal(80000, 15000, 200),      # Segment 2: Medium CLV
        np.random.normal(250000, 30000, 150)      # Segment 3: High CLV
    ]),
    'avg_order_value': np.concatenate([
        np.random.normal(100, 30, 150),           # Segment 1: Budget shoppers
        np.random.normal(300, 50, 200),           # Segment 2: Standard
        np.random.normal(800, 100, 150)           # Segment 3: Premium
    ]),
    'website_visits_month': np.concatenate([
        np.random.normal(5, 2, 150),              # Segment 1: Rarely visit
        np.random.normal(15, 4, 200),             # Segment 2: Regular visitors
        np.random.normal(40, 8, 150)              # Segment 3: Frequent visitors
    ]),
    'cart_abandonment_rate': np.concatenate([
        np.random.uniform(0.4, 0.8, 150),         # Segment 1: High abandonment
        np.random.uniform(0.2, 0.5, 200),         # Segment 2: Medium abandonment
        np.random.uniform(0.05, 0.2, 150)         # Segment 3: Low abandonment
    ]),
    'months_since_purchase': np.concatenate([
        np.random.uniform(0, 6, 150),             # Segment 1: At-risk
        np.random.uniform(0, 2, 200),             # Segment 2: Active
        np.random.uniform(0, 1, 150)              # Segment 3: Very active
    ])
})

print("=" * 70)
print("CUSTOMER SEGMENTATION DATASET")
print("=" * 70)
print(f"Total Customers: {len(customers)}")
print(f"\nDataset Statistics:")
print(customers.describe().round(2))

# Visualize raw data (before clustering)
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(customers.columns):
    axes[idx].hist(customers[col], bins=30, alpha=0.7, color='steelblue')
    axes[idx].set_title(col)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('customer_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved customer_distribution.png")

# ===========================
# STEP 1: DETERMINE OPTIMAL K
# ===========================
print("\n" + "=" * 70)
print("STEP 1: FINDING OPTIMAL NUMBER OF CLUSTERS (K)")
print("=" * 70)

# Standardize features (CRITICAL for K-Means!)
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Method 1: Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(customers_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(customers_scaled, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve
ax1.plot(K_range, inertias, marker='o', color='steelblue', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
ax1.set_title('Elbow Method for Optimal K')
ax1.grid(True, alpha=0.3)
ax1.annotate('Elbow Point', xy=(3, inertias[1]), xytext=(4, inertias[2]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Silhouette scores
ax2.plot(K_range, silhouette_scores, marker='s', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score for Each K')
ax2.grid(True, alpha=0.3)
optimal_k = K_range[np.argmax(silhouette_scores)]
ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
ax2.legend()

plt.tight_layout()
plt.savefig('optimal_k_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved optimal_k_analysis.png")

print(f"\n✅ Optimal K (by Silhouette Score): {optimal_k}")

# ===========================
# STEP 2: TRAIN FINAL MODEL
# ===========================
print("\n" + "=" * 70)
print("STEP 2: TRAINING K-MEANS WITH K=3")
print("=" * 70)

optimal_k = 3
kmeans_final = KMeans(
    n_clusters=optimal_k,
    init='k-means++',  # Smart initialization
    random_state=42,
    n_init=10,
    max_iter=300
)

cluster_labels = kmeans_final.fit_predict(customers_scaled)
customers['cluster'] = cluster_labels

print(f"\nCluster Distribution:")
print(customers['cluster'].value_counts().sort_index())

# Get cluster centers in original scale
cluster_centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)

print("\n" + "=" * 70)
print("CLUSTER CHARACTERISTICS (Original Scale)")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = customers[customers['cluster'] == cluster_id]
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id} ({len(cluster_data)} customers)")
    print(f"{'='*70}")
    
    for col in customers.columns[:-1]:  # Exclude cluster column
        mean_val = cluster_data[col].mean()
        overall_mean = customers[col].mean()
        pct_diff = ((mean_val - overall_mean) / overall_mean) * 100
        
        print(f"{col:30s}: {mean_val:12.2f} "
              f"(avg: {overall_mean:12.2f}, {pct_diff:+6.1f}%)")

# ===========================
# STEP 3: VISUALIZATION
# ===========================
print("\n" + "=" * 70)
print("STEP 3: VISUALIZING CLUSTERS")
print("=" * 70)

# 2D Visualization (PCA reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
customers_pca = pca.fit_transform(customers_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(customers_pca[:, 0], customers_pca[:, 1], 
                     c=cluster_labels, cmap='viridis', s=100, alpha=0.6)

# Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           marker='X', s=500, c='red', edgecolors='black', linewidths=2,
           label='Cluster Centers')

plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Customer Segments (K-Means, K=3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_clusters_2d.png', dpi=150, bbox_inches='tight')
print("Saved kmeans_clusters_2d.png")

# 3D Visualization
pca_3d = PCA(n_components=3)
customers_pca_3d = pca_3d.fit_transform(customers_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    ax.scatter(customers_pca_3d[mask, 0], 
              customers_pca_3d[mask, 1],
              customers_pca_3d[mask, 2],
              label=f'Cluster {cluster_id}', s=50, alpha=0.6)

centers_pca_3d = pca_3d.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca_3d[:, 0], centers_pca_3d[:, 1], centers_pca_3d[:, 2],
          marker='X', s=300, c='red', edgecolors='black', linewidths=2,
          label='Centers')

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
ax.set_title('3D Cluster Visualization')
ax.legend()
plt.savefig('kmeans_clusters_3d.png', dpi=150, bbox_inches='tight')
print("Saved kmeans_clusters_3d.png")

# ===========================
# STEP 4: BUSINESS INSIGHTS
# ===========================
print("\n" + "=" * 70)
print("BUSINESS INSIGHTS & MARKETING STRATEGY")
print("=" * 70)

segment_names = {
    0: 'VALUE SEEKERS',
    1: 'REGULAR BUYERS',
    2: 'VIP CUSTOMERS'
}

marketing_strategies = {
    'VALUE SEEKERS': """
    📊 Profile: Low income, low spending, infrequent visitors
    💡 Strategy:
        • Offer budget-friendly products
        • Discount codes and promotional emails
        • Simplify purchasing process (reduce cart abandonment)
        • High-frequency touch points (weekly emails)
    """,
    'REGULAR BUYERS': """
    📊 Profile: Medium income, medium spending, regular visitors
    💡 Strategy:
        • Cross-sell and upsell opportunities
        • Loyalty rewards program
        • Personalized product recommendations
        • Monthly newsletters with curated offers
    """,
    'VIP CUSTOMERS': """
    📊 Profile: High income, high spending, very frequent visitors
    💡 Strategy:
        • Exclusive VIP early access to new products
        • Personalized concierge service
        • Premium membership benefits
        • Invitation-only events
        • Personal account manager
    """
}

for cluster_id in range(optimal_k):
    segment_name = segment_names[cluster_id]
    print(f"\n{'='*70}")
    print(f"SEGMENT {cluster_id}: {segment_name}")
    print(f"{'='*70}")
    print(marketing_strategies[segment_name])

# ===========================
# STEP 5: CONVERGENCE ANALYSIS
# ===========================
print("\n" + "=" * 70)
print("STEP 5: K-MEANS CONVERGENCE ANALYSIS")
print("=" * 70)

# Train with iteration tracking
kmeans_iter = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42,
    n_init=1,  # Single init to see convergence
    max_iter=10
)

inertias_per_iter = []
for i in range(1, 11):
    kmeans_iter.max_iter = i
    kmeans_iter.fit(customers_scaled)
    inertias_per_iter.append(kmeans_iter.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertias_per_iter, marker='o', linewidth=2, markersize=8)
plt.xlabel('Iteration')
plt.ylabel('Inertia')
plt.title('K-Means Convergence (Inertia per Iteration)')
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_convergence.png', dpi=150, bbox_inches='tight')
print("Saved kmeans_convergence.png")

# ===========================
# STEP 6: EVALUATION METRICS
# ===========================
print("\n" + "=" * 70)
print("CLUSTERING EVALUATION METRICS")
print("=" * 70)

# Silhouette Score (-1 to 1, higher is better)
silhouette_avg = silhouette_score(customers_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")
print("  Range: -1 (bad) to 1 (good)")
print(f"  Interpretation: {'Excellent' if silhouette_avg > 0.7 else 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.3 else 'Poor'}")

# Davies-Bouldin Index (lower is better)
db_index = davies_bouldin_score(customers_scaled, cluster_labels)
print(f"\nDavies-Bouldin Index: {db_index:.4f}")
print("  Range: 0 (good) to ∞ (bad)")
print(f"  Interpretation: {'Excellent' if db_index < 1 else 'Good' if db_index < 1.5 else 'Fair'}")

# Calinski-Harabasz Score (higher is better)
from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(customers_scaled, cluster_labels)
print(f"\nCalinski-Harabasz Score: {ch_score:.2f}")
print("  Higher values indicate better-defined clusters")

# Within-cluster sum of squares
wcss = kmeans_final.inertia_
print(f"\nWithin-Cluster Sum of Squares (WCSS): {wcss:.2f}")
print("  Lower values indicate tighter clusters")

# ===========================
# STEP 7: ADVANCED - VISUALIZATION BY PAIRS
# ===========================
print("\n" + "=" * 70)
print("DETAILED CLUSTER VISUALIZATION")
print("=" * 70)

top_features = ['annual_income', 'annual_spending', 'customer_lifetime_value', 'avg_order_value']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, (feat1, feat2) in enumerate([
    ('annual_income', 'annual_spending'),
    ('customer_lifetime_value', 'purchase_frequency'),
    ('avg_order_value', 'website_visits_month'),
    ('months_since_purchase', 'cart_abandonment_rate')
]):
    ax = axes[idx]
    
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        ax.scatter(customers.loc[mask, feat1], 
                  customers.loc[mask, feat2],
                  label=f'Cluster {cluster_id}', s=80, alpha=0.6)
    
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f'{feat1} vs {feat2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_feature_pairs.png', dpi=150, bbox_inches='tight')
print("Saved kmeans_feature_pairs.png")

# ===========================
# STEP 8: CLUSTER COMPARISON TABLE
# ===========================
print("\n" + "=" * 70)
print("CLUSTER COMPARISON TABLE")
print("=" * 70)

comparison_table = pd.DataFrame()
for cluster_id in range(optimal_k):
    cluster_data = customers[customers['cluster'] == cluster_id]
    comparison_table[f'Cluster {cluster_id}'] = cluster_data.iloc[:, :-1].mean()

print(comparison_table.round(2))

# ===========================
# STEP 9: PREDICT NEW CUSTOMERS
# ===========================
print("\n" + "=" * 70)
print("PREDICTING SEGMENT FOR NEW CUSTOMERS")
print("=" * 70)

# New customer data
new_customers = pd.DataFrame({
    'annual_income': [35000, 75000, 180000],
    'annual_spending': [4000, 14000, 50000],
    'purchase_frequency': [8, 24, 55],
    'customer_lifetime_value': [15000, 75000, 300000],
    'avg_order_value': [80, 280, 900],
    'website_visits_month': [4, 16, 45],
    'cart_abandonment_rate': [0.7, 0.3, 0.08],
    'months_since_purchase': [3, 1, 0.5]
})

# Scale new customers
new_customers_scaled = scaler.transform(new_customers)

# Predict
new_clusters = kmeans_final.predict(new_customers_scaled)
new_distances = kmeans_final.transform(new_customers_scaled)

print("\nNew Customer Predictions:")
for i, cluster in enumerate(new_clusters):
    distance_to_center = new_distances[i].min()
    print(f"\nCustomer {i+1}:")
    print(f"  Assigned Segment: {segment_names[cluster]}")
    print(f"  Distance to center: {distance_to_center:.4f}")
    print(f"  Input: {new_customers.iloc[i].to_dict()}")
```

### **When to Use Each Initialization:**

```python
print("\n" + "=" * 70)
print("K-MEANS INITIALIZATION METHODS")
print("=" * 70)

init_methods = ['k-means++', 'random']
results = {}

for init_method in init_methods:
    scores = []
    for trial in range(10):
        kmeans = KMeans(n_clusters=3, init=init_method, random_state=trial, n_init=1)
        kmeans.fit(customers_scaled)
        scores.append(silhouette_score(customers_scaled, kmeans.labels_))
    
    results[init_method] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }
    
    print(f"\n{init_method.upper()}:")
    print(f"  Mean Silhouette: {results[init_method]['mean']:.4f}")
    print(f"  Std Dev: {results[init_method]['std']:.4f}")
    print(f"  Range: [{results[init_method]['min']:.4f}, {results[init_method]['max']:.4f}]")

print("\n💡 k-means++ is almost always better than random initialization!")
```

**Real-World Applications:**
- 🛒 **Retail:** Customer segmentation (Macy's, Target)
- 🎮 **Gaming:** Player behavior clustering
- 🏥 **Healthcare:** Patient grouping for treatment plans
- 📊 **Finance:** Customer risk segmentation
- 🖼️ **Image Processing:** Image compression, color quantization

---

## 10. DBSCAN (Density-Based Clustering) ⭐⭐⭐⭐

### **What is it?**
Finds clusters based on **density** rather than distance. Great for arbitrary-shaped clusters.

### **How It Works:**
```
Two Key Parameters:
- eps: Maximum distance between points
- min_samples: Minimum points in eps radius

Three Point Types:
1. Core Points: ≥ min_samples neighbors within eps
2. Border Points: In neighborhood of core point
3. Noise Points: Not core or border

Clustering:
- Core points in same eps-neighborhood → same cluster
- Border points → assigned to core point's cluster
- Noise points → labeled as outliers (-1)
```

### **Why Use It?**
✅ **Finds Arbitrary Shapes:** Not limited to spheres  
✅ **Automatic K:** No need to specify clusters  
✅ **Outlier Detection:** Identifies noise points  
✅ **Density-based:** Good for varying densities  

❌ **Limitations:**
- Sensitive to eps and min_samples
- Struggles with varying densities
- Slower on large datasets

### **When to Use:**
- Unknown number of clusters
- Non-spherical clusters
- Want outlier detection
- Spatial data (GPS coordinates)

### **Detailed Example: Anomaly Detection in Geographic Data**

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate geographic location data (e.g., customer locations)
np.random.seed(42)

# Normal customer clusters
cluster1_lat = np.random.normal(40.7128, 0.05, 150)  # NYC
cluster1_lon = np.random.normal(-74.0060, 0.05, 150)

cluster2_lat = np.random.normal(34.0522, 0.05, 150)  # LA
cluster2_lon = np.random.normal(-118.2437, 0.05, 150)

cluster3_lat = np.random.normal(41.8781, 0.05, 120)  # Chicago
cluster3_lon = np.random.normal(-87.6298, 0.05, 120)

# Add outliers (fraudulent locations)
outliers_lat = np.random.uniform(25, 49, 20)
outliers_lon = np.random.uniform(-125, -66, 20)

# Combine
all_lats = np.concatenate([cluster1_lat, cluster2_lat, cluster3_lat, outliers_lat])
all_lons = np.concatenate([cluster1_lon, cluster2_lon, cluster3_lon, outliers_lon])

locations = pd.DataFrame({
    'latitude': all_lats,
    'longitude': all_lons
})

print("=" * 70)
print("GEOGRAPHIC CLUSTERING WITH DBSCAN")
print("=" * 70)
print(f"Total Points: {len(locations)}")

# Scale features
scaler = StandardScaler()
locations_scaled = scaler.fit_transform(locations)

# ===========================
# STEP 1: FIND OPTIMAL EPS
# ===========================
print("\n" + "=" * 70)
print("STEP 1: FINDING OPTIMAL EPS")
print("=" * 70)

# K-distance graph
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(locations_scaled)
distances, indices = neighbors_fit.kneighbors(locations_scaled)
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(12, 5))
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5-NN Distance')
plt.title('K-distance Graph for Optimal Eps Selection')
plt.axhline(y=0.3, color='r', linestyle='--', label='eps=0.3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_distance_graph.png', dpi=150, bbox_inches='tight')
print("Saved knn_distance_graph.png")

# ===========================
# STEP 2: DBSCAN CLUSTERING
# ===========================
print("\n" + "=" * 70)
print("STEP 2: DBSCAN WITH DIFFERENT PARAMETERS")
print("=" * 70)

eps_values = [0.2, 0.3, 0.5]
min_samples_values = [3, 5, 10]

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), 
                         figsize=(15, 12))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(locations_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        ax = axes[i, j]
        
        # Plot clusters
        mask = clusters != -1
        scatter = ax.scatter(locations.loc[mask, 'latitude'],
                           locations.loc[mask, 'longitude'],
                           c=clusters[mask], cmap='viridis', s=50, alpha=0.6,
                           label='Clusters')
        
        # Plot noise points
        noise_mask = clusters == -1
        ax.scatter(locations.loc[noise_mask, 'latitude'],
                  locations.loc[noise_mask, 'longitude'],
                  c='red', marker='X', s=200, label='Noise/Outliers',
                  edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_title(f'eps={eps}, min_samples={min_samples}\n'
                    f'Clusters: {n_clusters}, Noise: {n_noise}')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_parameter_comparison.png', dpi=150, bbox_inches='tight')
print("Saved dbscan_parameter_comparison.png")

# Best DBSCAN
print("\nOptimal DBSCAN: eps=0.3, min_samples=5")
dbscan_best = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan_best.fit_predict(locations_scaled)

locations['cluster'] = clusters

print(f"\nCluster Distribution:")
print(f"  Valid Clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
print(f"  Noise Points: {list(clusters).count(-1)}")

# Detailed cluster info
print("\n" + "=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        print(f"\nNOISE/OUTLIERS ({list(clusters).count(-1)} points)")
    else:
        cluster_data = locations[locations['cluster'] == cluster_id]
        center_lat = cluster_data['latitude'].mean()
        center_lon = cluster_data['longitude'].mean()
        print(f"\nCluster {cluster_id} ({len(cluster_data)} points)")
        print(f"  Center: ({center_lat:.4f}, {center_lon:.4f})")
        print(f"  Lat Range: [{cluster_data['latitude'].min():.4f}, {cluster_data['latitude'].max():.4f}]")
        print(f"  Lon Range: [{cluster_data['longitude'].min():.4f}, {cluster_data['longitude'].max():.4f}]")

# ===========================
# ANOMALY DETECTION APPLICATION
# ===========================
print("\n" + "=" * 70)
print("BUSINESS APPLICATION: FRAUD DETECTION")
print("=" * 70)

# Identify anomalies
anomalies = locations[locations['cluster'] == -1]

print(f"\nDetected {len(anomalies)} anomalous transactions:")
for idx, (_, row) in enumerate(anomalies.iterrows(), 1):
    print(f"  {idx}. Location: ({row['latitude']:.4f}, {row['longitude']:.4f})")

print("\n💡 These locations are spatially isolated - might indicate:")
print("   • Fraudulent transactions")
print("   • Data entry errors")
print("   • Legitimate but unusual patterns")
```

**Real-World Applications:**
- 🚨 **Cybersecurity:** Intrusion detection in network logs
- 🗺️ **GPS:** Identifying unusual travel patterns
- 🏭 **Manufacturing:** Quality control anomalies
- 📊 **Finance:** Unusual transaction detection

---

## 11. Gaussian Mixture Models (GMM) ⭐⭐⭐

### **What is it?**
Probabilistic model assuming each cluster is a Gaussian distribution.

### **vs K-Means:**
```
K-MEANS:
- Hard assignment (point belongs to one cluster)
- Each cluster = circle
- Example: Document belongs to Sports (100%)

GMM:
- Soft assignment (probability distribution)
- Each cluster = ellipse (covariance matrix)
- Example: Document is 70% Sports, 30% News
```

### **Why Use It?**
✅ **Probabilistic:** Get probability of cluster membership  
✅ **Flexible Shapes:** Elliptical clusters via covariance  
✅ **Statistical:** BIC/AIC for model selection  

### **Detailed Example: Document Topic Modeling**

```python
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer

# Document collection
documents = [
    "Machine learning algorithms for data science",
    "Deep neural networks and artificial intelligence",
    "Basketball playoffs and team strategies",
    "Football championship and player performance",
    "Python programming for machine learning",
    "Tennis tournament and athlete rankings",
    "Natural language processing with transformers",
    "Soccer matches and tactical analysis",
    "Computer vision and image classification",
    "Artificial intelligence in robotics"
]

print("=" * 70)
print("TOPIC MODELING WITH GMM")
print("=" * 70)

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=50)
doc_vectors = vectorizer.fit_transform(documents).toarray()

print(f"Documents: {len(documents)}")
print(f"Features: {doc_vectors.shape[1]}")

# Find optimal number of components
bic_scores = []
aic_scores = []
n_components_range = range(2, 6)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(doc_vectors)
    bic_scores.append(gmm.bic(doc_vectors))
    aic_scores.append(gmm.aic(doc_vectors))

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o', label='BIC')
plt.plot(n_components_range, aic_scores, marker='s', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('GMM Model Selection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gmm_model_selection.png', dpi=150, bbox_inches='tight')
print("Saved gmm_model_selection.png")

# Best model
optimal_components = n_components_range[np.argmin(bic_scores)]
print(f"\n✅ Optimal Components: {optimal_components}")

gmm = GaussianMixture(n_components=optimal_components, random_state=42)
topics = gmm.fit_predict(doc_vectors)
probabilities = gmm.predict_proba(doc_vectors)

# Display results
print("\n" + "=" * 70)
print("DOCUMENT-TOPIC ASSIGNMENTS")
print("=" * 70)

for i, (doc, topic_probs) in enumerate(zip(documents, probabilities)):
    print(f"\nDoc {i}: {doc}")
    for topic_id, prob in enumerate(topic_probs):
        if prob > 0.1:  # Show topics with >10% probability
            print(f"  Topic {topic_id}: {prob:.1%}")
```

**Real-World Applications:**
- 📰 **Media:** News topic classification
- 🎵 **Music:** Genre classification with soft labels
- 🏥 **Healthcare:** Disease subtype identification
- 📧 **Email:** Spam confidence scoring

---

# 📉 DIMENSIONALITY REDUCTION

## 12. PCA (Principal Component Analysis) ⭐⭐⭐⭐⭐

### **What is it?**
Projects high-dimensional data to fewer dimensions while preserving variance.

### **How It Works:**
```
Problem: 1000 features (genes in genome)

Goal: Reduce to 50 features that capture 95% of variance

Solution:
1. Find direction of maximum variance (PC1)
2. Find next direction orthogonal to PC1 (PC2)
3. Repeat for desired components

Result:
Original: 1000D → PCA: 50D
Lost Info: Only 5% (variance-wise)
Gain: 20x faster algorithms, less overfitting, visualization
```

### **Mathematical Concept:**
```
Input: X (n_samples × n_features)
Output: X_reduced (n_samples × n_components)

Steps:
1. Center data: X = X - mean(X)
2. Compute covariance matrix: C = (X^T × X) / n
3. Find eigenvectors of C
4. Project: X_reduced = X × eigenvectors[:, :n_components]

Variance explained by component i = eigenvalue_i / sum(all eigenvalues)
```

### **Why Use It?**
✅ **Visualization:** See high-D data in 2D/3D  
✅ **Preprocessing:** Reduce features before modeling  
✅ **Noise Reduction:** Remove low-variance noise  
✅ **Speed:** Dramatically faster algorithms  
✅ **Interpretability:** Understand feature relationships  

### **Detailed Example: Gene Expression Analysis**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulate gene expression data
# 5000 genes × 100 patients
np.random.seed(42)

# Generate 3 disease subtypes with different patterns
healthy = np.random.normal(5, 2, (30, 5000))  # Control
disease_a = np.random.normal(7, 2, (35, 5000))  # Type A: high expression
disease_b = np.random.normal(3, 2, (35, 5000))  # Type B: low expression

gene_expression = np.vstack([healthy, disease_a, disease_b])
patient_types = np.array(['Healthy']*30 + ['Type A']*35 + ['Type B']*35)

print("=" * 70)
print("GENE EXPRESSION ANALYSIS WITH PCA")
print("=" * 70)
print(f"Gene Expression Matrix: {gene_expression.shape}")
print(f"  Patients: {gene_expression.shape[0]}")
print(f"  Genes: {gene_expression.shape[1]}")
print(f"\nPatient Distribution:")
print(f"  Healthy: 30")
print(f"  Type A: 35")
print(f"  Type B: 35")

# Standardize
scaler = StandardScaler()
gene_expression_scaled = scaler.fit_transform(gene_expression)

# ===========================
# STEP 1: FIND OPTIMAL COMPONENTS
# ===========================
print("\n" + "=" * 70)
print("STEP 1: FINDING OPTIMAL NUMBER OF COMPONENTS")
print("=" * 70)

pca_full = PCA()
pca_full.fit(gene_expression_scaled)

# Cumulative variance explained
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Individual variance per component
ax1.bar(range(1, 21), pca_full.explained_variance_ratio_[:20])
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Variance Explained by Each Component')
ax1.grid(True, alpha=0.3)

# Cumulative variance
ax2.plot(cumsum_var[:50], marker='o')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax2.axvline(x=np.argmax(cumsum_var >= 0.95), color='g', linestyle='--',
           label=f'n_components={np.argmax(cumsum_var >= 0.95) + 1}')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Variance Explained')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=150, bbox_inches='tight')
print("Saved pca_variance_analysis.png")

# Find components for 95% variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")
print(f"Variance explained: {cumsum_var[n_components_95-1]:.2%}")
print(f"Dimensionality reduction: 5000 → {n_components_95} ({n_components_95/5000*100:.2f}%)")

# ===========================
# STEP 2: VISUALIZATION WITH 2D/3D PCA
# ===========================
print("\n" + "=" * 70)
print("STEP 2: VISUALIZATION")
print("=" * 70)

# 2D PCA
pca_2d = PCA(n_components=2)
gene_expr_2d = pca_2d.fit_transform(gene_expression_scaled)

plt.figure(figsize=(12, 8))
colors = {'Healthy': 'green', 'Type A': 'red', 'Type B': 'blue'}

for patient_type in ['Healthy', 'Type A', 'Type B']:
    mask = patient_types == patient_type
    plt.scatter(gene_expr_2d[mask, 0], gene_expr_2d[mask, 1],
               c=colors[patient_type], label=patient_type, s=100, alpha=0.6)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Gene Expression: Patient Grouping (2D PCA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pca_2d_patients.png', dpi=150, bbox_inches='tight')
print("Saved pca_2d_patients.png")

# 3D PCA
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
gene_expr_3d = pca_3d.fit_transform(gene_expression_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for patient_type in ['Healthy', 'Type A', 'Type B']:
    mask = patient_types == patient_type
    ax.scatter(gene_expr_3d[mask, 0], gene_expr_3d[mask, 1], gene_expr_3d[mask, 2],
              c=colors[patient_type], label=patient_type, s=50, alpha=0.6)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
ax.set_title('Gene Expression: Patient Grouping (3D PCA)')
ax.legend()
plt.savefig('pca_3d_patients.png', dpi=150, bbox_inches='tight')
print("Saved pca_3d_patients.png")

# ===========================
# STEP 3: PRINCIPAL COMPONENT INTERPRETATION
# ===========================
print("\n" + "=" * 70)
print("STEP 3: INTERPRETING PRINCIPAL COMPONENTS")
print("=" * 70)

pca_20 = PCA(n_components=20)
pca_20.fit(gene_expression_scaled)

# Get gene loadings (importance of each gene in PC1)
loadings = pca_20.components_[0]  # PC1 loadings
top_genes_pc1 = np.argsort(np.abs(loadings))[-10:]

print("\nTop 10 Genes Contributing to PC1:")
print(f"PC1 explains {pca_20.explained_variance_ratio_[0]:.2%} of variance")
for rank, gene_idx in enumerate(reversed(top_genes_pc1), 1):
    loading = loadings[gene_idx]
    print(f"  {rank:2d}. Gene {gene_idx:4d}: {loading:7.4f}")

# PC2 interpretation
loadings_pc2 = pca_20.components_[1]
top_genes_pc2 = np.argsort(np.abs(loadings_pc2))[-10:]

print(f"\nTop 10 Genes Contributing to PC2:")
print(f"PC2 explains {pca_20.explained_variance_ratio_[1]:.2%} of variance")
for rank, gene_idx in enumerate(reversed(top_genes_pc2), 1):
    loading = loadings_pc2[gene_idx]
    print(f"  {rank:2d}. Gene {gene_idx:4d}: {loading:7.4f}")

# ===========================
# STEP 4: RECONSTRUCTION ERROR
# ===========================
print("\n" + "=" * 70)
print("STEP 4: RECONSTRUCTION ERROR ANALYSIS")
print("=" * 70)

n_components_range = [2, 5, 10, 20, 50, 100]
reconstruction_errors = []

for n_comp in n_components_range:
    pca_temp = PCA(n_components=n_comp)
    gene_expr_reduced = pca_temp.fit_transform(gene_expression_scaled)
    gene_expr_reconstructed = pca_temp.inverse_transform(gene_expr_reduced)
    error = np.mean((gene_expression_scaled - gene_expr_reconstructed) ** 2)
    reconstruction_errors.append(error)
    print(f"Components: {n_comp:3d} | MSE: {error:.6f} | Var Explained: {np.sum(pca_temp.explained_variance_ratio_):.2%}")

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, reconstruction_errors, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction MSE')
plt.title('PCA: Reconstruction Error vs Components')
plt.grid(True, alpha=0.3)
plt.savefig('pca_reconstruction_error.png', dpi=150, bbox_inches='tight')
print("\nSaved pca_reconstruction_error.png")

# ===========================
# STEP 5: SPEED COMPARISON
# ===========================
print("\n" + "=" * 70)
print("STEP 5: COMPUTATIONAL SPEED BENEFIT")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Original data (5000 features)
print("Training on original data (5000 features)...")
start = time.time()
rf_original = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
scores_original = cross_val_score(rf_original, gene_expression_scaled, 
                                 (patient_types != 'Healthy').astype(int), cv=3)
time_original = time.time() - start

# PCA-reduced data (20 components)
print("Training on PCA-reduced data (20 components)...")
pca = PCA(n_components=20)
gene_expr_pca = pca.fit_transform(gene_expression_scaled)

start = time.time()
rf_pca = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
scores_pca = cross_val_score(rf_pca, gene_expr_pca, 
                            (patient_types != 'Healthy').astype(int), cv=3)
time_pca = time.time() - start

print("\n" + "=" * 70)
print("SPEED vs ACCURACY COMPARISON")
print("=" * 70)
print(f"\nOriginal (5000 features):")
print(f"  Accuracy: {scores_original.mean():.4f} (+/- {scores_original.std():.4f})")
print(f"  Time: {time_original:.2f}s")

print(f"\nPCA-reduced (20 components):")
print(f"  Accuracy: {scores_pca.mean():.4f} (+/- {scores_pca.std():.4f})")
print(f"  Time: {time_pca:.2f}s")

print(f"\nSpeedup: {time_original/time_pca:.1f}x")
print(f"Accuracy loss: {(scores_original.mean() - scores_pca.mean())*100:.2f}%")
```

**Real-World Applications:**
- 🧬 **Genomics:** Analyzing gene expression data
- 📊 **Finance:** Portfolio risk analysis
- 🖼️ **Image:** Face recognition (eigenfaces)
- 📡 **Signal Processing:** Noise reduction
- 🎮 **Recommendation:** Latent factor models

---

## 13. t-SNE (t-Distributed Stochastic Neighbor Embedding) ⭐⭐⭐⭐

### **What is it?**
Visualizes high-dimensional data in 2D/3D by **preserving local structure**.

### **vs PCA:**
```
PCA:
- Preserves global structure
- Linear transformation
- Deterministic

t-SNE:
- Preserves local neighborhoods
- Nonlinear
- Probabilistic (different each run)
- Better for visualization
```

### **Detailed Example: Neural Network Feature Visualization**

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits

print("=" * 70)
print("t-SNE VISUALIZATION EXAMPLES")
print("=" * 70)

# Example 1: Iris Dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print("\n1. IRIS DATASET")
print(f"Shape: {X_iris.shape} (150 samples, 4 features)")

# t-SNE with different perplexities
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, perplexity in enumerate(perplexities):
    print(f"\n  Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    iris_tsne = tsne.fit_transform(X_iris)
    
    ax = axes[idx]
    scatter = ax.scatter(iris_tsne[:, 0], iris_tsne[:, 1], 
                        c=y_iris, cmap='viridis', s=100, alpha=0.6)
    ax.set_title(f'Iris (perplexity={perplexity})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Class')

plt.tight_layout()
plt.savefig('tsne_iris_perplexity.png', dpi=150, bbox_inches='tight')
print("\nSaved tsne_iris_perplexity.png")

# Example 2: Handwritten Digits
digits = load_digits()
X_digits = digits.data[:500]  # Subset for speed
y_digits = digits.target[:500]

print("\n" + "=" * 70)
print("2. HANDWRITTEN DIGITS")
print(f"Shape: {X_digits.shape} (500 samples, 64 features)")

print("Computing t-SNE...")
tsne_digits = TSNE(n_components=2, random_state=42, n_iter=1000)
digits_tsne = tsne_digits.fit_transform(X_digits)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(digits_tsne[:, 0], digits_tsne[:, 1],
                     c=y_digits, cmap='tab10', s=100, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('Handwritten Digits - t-SNE Visualization')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.savefig('tsne_digits.png', dpi=150, bbox_inches='tight')
print("Saved tsne_digits.png")

# Example 3: 3D t-SNE
print("\n" + "=" * 70)
print("3. 3D t-SNE VISUALIZATION")
print("Computing 3D t-SNE...")

tsne_3d = TSNE(n_components=3, random_state=42, n_iter=1000)
digits_tsne_3d = tsne_3d.fit_transform(X_digits)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(digits_tsne_3d[:, 0], digits_tsne_3d[:, 1], digits_tsne_3d[:, 2],
                    c=y_digits, cmap='tab10', s=50, alpha=0.6)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
ax.set_title('3D t-SNE: Handwritten Digits')
plt.colorbar(scatter, ax=ax, label='Digit', pad=0.1)
plt.savefig('tsne_digits_3d.png', dpi=150, bbox_inches='tight')
print("Saved tsne_digits_3d.png")

# Example 4: Parameter Impact
print("\n" + "=" * 70)
print("4. PARAMETER IMPACT ANALYSIS")
print("=" * 70)

learning_rates = [10, 100, 1000]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, lr in enumerate(learning_rates):
    print(f"\n  Computing t-SNE (learning_rate={lr})...")
    tsne = TSNE(n_components=2, learning_rate=lr, random_state=42, n_iter=1000)
    iris_tsne = tsne.fit_transform(X_iris)
    
    ax = axes[idx]
    scatter = ax.scatter(iris_tsne[:, 0], iris_tsne[:, 1],
                        c=y_iris, cmap='viridis', s=100, alpha=0.6)
    ax.set_title(f'Iris (learning_rate={lr})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('tsne_learning_rate.png', dpi=150, bbox_inches='tight')
print("\nSaved tsne_learning_rate.png")
```

**Key Parameters:**

```python
print("""
t-SNE HYPERPARAMETERS:

1. perplexity (5-50, typical 30)
   • Balances local vs global structure
   • Lower = focus on local neighborhoods
   • Higher = focus on global structure

2. learning_rate (10-1000, typical 200)
   • Speed of optimization
   • Too low = slow, may not converge
   • Too high = poor convergence

3. n_iter (minimum 250, typical 1000)
   • Number of iterations
   • More = better but slower
   • Should be at least 250

4. perplexity vs n_samples
   • perplexity should be < n_samples / 3
   • Typical: perplexity = 30-50 for large datasets

⚠️ WARNING: t-SNE is SLOW for large datasets (>10K samples)
Use UMAP as alternative for large data!
""")
```

---

## 14. UMAP (Uniform Manifold Approximation and Projection) ⭐⭐⭐

### **What is it?**
Modern alternative to t-SNE: **faster, preserves both local and global structure**.

### **UMAP vs t-SNE:**
```
t-SNE:
- Very slow on large datasets
- Only 2D/3D visualization
- Excellent for visualization
- Time: O(n²)

UMAP:
- 10x faster than t-SNE
- Works in any dimension
- Preserves global structure too
- Time: O(n log n)
```

### **Quick Example:**

```python
import umap

print("=" * 70)
print("UMAP VS t-SNE SPEED COMPARISON")
print("=" * 70)

# Large dataset
X_large = np.random.randn(5000, 100)
y_large = np.random.randint(0, 10, 5000)

# t-SNE (slow!)
print("\nt-SNE (might take 1-2 minutes)...")
start = time.time()
tsne = TSNE(n_components=2, random_state=42, n_iter=500)
X_tsne = tsne.fit_transform(X_large)
time_tsne = time.time() - start
print(f"  Time: {time_tsne:.2f}s")

# UMAP (fast!)
print("\nUMAP (much faster)...")
start = time.time()
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_large)
time_umap = time.time() - start
print(f"  Time: {time_umap:.2f}s")

print(f"\nSpeedup: {time_tsne/time_umap:.1f}x faster!")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_large, cmap='tab10', s=20, alpha=0.6)
ax1.set_title(f't-SNE ({time_tsne:.1f}s)')

ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y_large, cmap='tab10', s=20, alpha=0.6)
ax2.set_title(f'UMAP ({time_umap:.1f}s)')

plt.tight_layout()
plt.savefig('umap_vs_tsne.png', dpi=150, bbox_inches='tight')
print("\nSaved umap_vs_tsne.png")
```

**Real-World Applications:**
- 🔬 **Biology:** Visualizing single-cell RNA data
- 🧠 **Neuroscience:** Analyzing brain imaging data
- 🛍️ **Recommendation:** Customer similarity embeddings
- 📱 **Deep Learning:** Visualizing learned representations

---

# 🗺️ COMPLETE ML ALGORITHM SELECTION FLOWCHART

---

## COMPREHENSIVE DECISION TREE WITH CODE

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

print("=" * 100)
print("COMPLETE MACHINE LEARNING ALGORITHM SELECTION GUIDE")
print("=" * 100)

# Create comprehensive decision guide
decision_guide = """

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING ALGORITHM SELECTION                        │
└─────────────────────────────────────────────────────────────────────────────────┘

START HERE: Do you have labeled data?
│
├─ YES (SUPERVISED LEARNING)
│   │
│   ├─ Is output continuous (regression)?
│   │   │
│   │   ├─ Linear relationship?
│   │   │   │
│   │   │   ├─ YES, simple & fast needed
│   │   │   │   └─→ LINEAR REGRESSION ⭐⭐⭐
│   │   │   │        • House prices, sales forecasting
│   │   │   │        • Training time: Milliseconds
│   │   │   │        • Interpretability: Excellent
│   │   │   │
│   │   │   ├─ YES, but many features (multicollinearity)
│   │   │   │   └─→ RIDGE/LASSO REGRESSION ⭐⭐⭐
│   │   │   │        • Feature selection needed
│   │   │   │        • Prevent overfitting
│   │   │   │
│   │   │   └─ NO, complex non-linear
│   │   │       └─ Continue below...
│   │   │
│   │   ├─ Non-linear relationship?
│   │   │   │
│   │   │   ├─ Simple curves (parabola, polynomial)
│   │   │   │   └─→ POLYNOMIAL REGRESSION ⭐⭐
│   │   │   │        • Degree 2-3 typical
│   │   │   │        • Watch for overfitting
│   │   │   │
│   │   │   ├─ Complex patterns in structured data?
│   │   │   │   │
│   │   │   │   ├─ Small-Medium (<100K) dataset
│   │   │   │   │   └─→ RANDOM FOREST ⭐⭐⭐⭐⭐
│   │   │   │   │        • Accurate baseline
│   │   │   │   │        • Feature importance
│   │   │   │   │        • Default choice
│   │   │   │   │
│   │   │   │   ├─ Medium-Large dataset, need SOTA accuracy
│   │   │   │   │   └─→ XGBOOST/LIGHTGBM ⭐⭐⭐⭐⭐
│   │   │   │   │        • Kaggle winner
│   │   │   │   │        • Best accuracy
│   │   │   │   │        • Handles missing values
│   │   │   │   │
│   │   │   │   └─ Many categorical features
│   │   │   │       └─→ CATBOOST ⭐⭐⭐⭐
│   │   │   │            • Built-in categorical handling
│   │   │   │            • Less tuning
│   │   │   │
│   │   │   ├─ Images, text, or sequences
│   │   │   │   └─→ NEURAL NETWORKS ⭐⭐⭐⭐
│   │   │   │        • Deep learning
│   │   │   │        • State-of-the-art
│   │   │   │        • Needs large dataset
│   │   │   │
│   │   │   └─ Time series / sequential data
│   │   │       └─→ LSTM/RNN ⭐⭐⭐⭐
│   │   │            • Temporal dependencies
│   │   │            • Stock prices, weather
│   │   │
│   │   └─ Speed critical (real-time predictions)?
│   │       └─→ LINEAR REGRESSION or POLYNOMIAL ⭐⭐⭐
│   │            • Millisecond latency
│   │            • Trade accuracy for speed
│   │
│   │
│   └─ Is output categorical (classification)?
│       │
│       ├─ Binary classification (Yes/No)?
│       │   │
│       │   ├─ Need probability scores?
│       │   │   ├─ YES
│       │   │   │   └─→ LOGISTIC REGRESSION ⭐⭐⭐⭐
│       │   │   │        • Email spam detection
│       │   │   │        • Disease prediction
│       │   │   │        • Get confidence scores
│       │   │   │
│       │   │   └─ NO, just class labels
│       │   │       └─ Continue below...
│       │   │
│       │   ├─ Linear decision boundary?
│       │   │   └─→ LOGISTIC REGRESSION ⭐⭐⭐⭐
│       │   │
│       │   ├─ Structured data + high accuracy?
│       │   │   ├─ Small dataset (<10K)
│       │   │   │   └─→ RANDOM FOREST ⭐⭐⭐⭐⭐
│       │   │   │
│       │   │   ├─ Large dataset + competitive accuracy
│       │   │   │   └─→ XGBOOST/LIGHTGBM ⭐⭐⭐⭐⭐
│       │   │   │
│       │   │   └─ Categorical features dominant
│       │   │       └─→ CATBOOST ⭐⭐⭐⭐
│       │   │
│       │   ├─ High-dimensional data (text, images)?
│       │   │   └─→ SVM ⭐⭐⭐⭐
│       │   │        • Text classification
│       │   │        • RBF kernel for non-linear
│       │   │
│       │   ├─ Complex patterns, unstructured data
│       │   │   └─→ NEURAL NETWORKS ⭐⭐⭐⭐
│       │   │        • Deep learning
│       │   │        • CNN for images
│       │   │        • RNN for sequences
│       │   │
│       │   ├─ Need very simple, interpretable model
│       │   │   └─→ DECISION TREE ⭐⭐⭐
│       │   │        • Easy to explain to non-technical
│       │   │        • But prone to overfitting
│       │   │
│       │   ├─ Need fast predictions?
│       │   │   └─→ KNN (small k) ⭐⭐
│       │   │        • Simple implementation
│       │   │        • Slow prediction on large data
│       │   │
│       │   └─ Need Naive Bayes (assuming feature independence)
│       │       └─→ NAIVE BAYES ⭐⭐⭐
│       │            • Text classification
│       │            • Baseline for NLP
│       │
│       │
│       └─ Multi-class classification (3+ classes)?
│           │
│           ├─ Structured data
│           │   └─→ RANDOM FOREST / XGBOOST ⭐⭐⭐⭐⭐
│           │        • Same as binary
│           │        • Works for multi-class natively
│           │
│           ├─ One-vs-Rest strategy needed
│           │   └─→ LOGISTIC REGRESSION ⭐⭐⭐⭐
│           │        • Can handle multi-class
│           │
│           ├─ Unstructured data
│           │   └─→ NEURAL NETWORKS ⭐⭐⭐⭐
│           │        • CNN/RNN for complex data
│           │
│           └─ Interpretability needed
│               └─→ DECISION TREE ⭐⭐⭐
│                    • Easy to visualize decision rules
│
│
└─ NO (UNSUPERVISED LEARNING)
    │
    ├─ Goal: Find groups/clusters?
    │   │
    │   ├─ Know number of clusters (K)?
    │   │   ├─ YES
    │   │   │   └─→ K-MEANS ⭐⭐⭐⭐⭐
    │   │   │        • Customer segmentation
    │   │   │        • Image compression
    │   │   │        • Fast, scalable
    │   │   │
    │   │   └─ NO, discover automatically
    │   │       │
    │   │       ├─ Dense, arbitrary-shaped clusters
    │   │       │   └─→ DBSCAN ⭐⭐⭐⭐
    │   │       │        • Anomaly detection
    │   │       │        • GPS clustering
    │   │       │
    │   │       ├─ Probabilistic, soft clusters
    │   │       │   └─→ GAUSSIAN MIXTURE MODELS ⭐⭐⭐
    │   │       │        • Get probability of membership
    │   │       │        • BIC/AIC for model selection
    │   │       │
    │   │       └─ Hierarchical relationships
    │   │           └─→ HIERARCHICAL CLUSTERING ⭐⭐⭐
    │   │                • Dendrogram visualization
    │   │                • Bottom-up or top-down
    │   │
    │   └─ Outlier detection?
    │       └─→ ISOLATION FOREST ⭐⭐⭐
    │            • Fast anomaly detection
    │            • Works in high dimensions
    │
    │
    ├─ Goal: Reduce dimensions / Visualization?
    │   │
    │   ├─ Need numerical reduction + interpretability?
    │   │   └─→ PCA (Principal Component Analysis) ⭐⭐⭐⭐⭐
    │   │        • Variance-based
    │   │        • Preserves global structure
    │   │        • Fast, scalable
    │   │        • Use before ML algorithms
    │   │
    │   ├─ Need visualization (2D/3D)?
    │   │   │
    │   │   ├─ Small dataset (<10K), static plot OK
    │   │   │   └─→ t-SNE ⭐⭐⭐⭐
    │   │   │        • Best visualization
    │   │   │        • Preserves local structure
    │   │   │        • SLOW on large data
    │   │   │
    │   │   ├─ Large dataset (>10K), need speed
    │   │   │   └─→ UMAP ⭐⭐⭐
    │   │   │        • 10x faster than t-SNE
    │   │   │        • Preserves global structure too
    │   │   │
    │   │   └─ Very large dataset (>100K)
    │   │       └─→ UMAP with negative_sample_rate
    │   │            • Memory efficient
    │   │
    │   └─ Autoencoders for complex patterns
    │       └─→ NEURAL NETWORK AUTOENCODER ⭐⭐⭐
    │            • Non-linear dimensionality reduction
    │            • Deep learning
    │
    │
    └─ Goal: Find relationships / patterns?
        │
        ├─ Association rules (market basket)
        │   └─→ APRIORI / ECLAT ⭐⭐⭐
        │        • Product recommendations
        │        • Rules like: "if buy A and B, likely buy C"
        │
        ├─ Word embeddings / topic modeling
        │   └─→ WORD2VEC / LDA ⭐⭐⭐
        │        • NLP applications
        │        • Document similarity
        │
        └─ Anomaly detection (normal vs outliers)
            └─→ ISOLATION FOREST / LOCAL OUTLIER FACTOR ⭐⭐⭐
                 • Credit card fraud
                 • Intrusion detection
                 • System monitoring


┌─────────────────────────────────────────────────────────────────────────────────┐
│                         QUICK REFERENCE MATRIX                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                  │ Speed  │Accuracy│Scaling │Interpretable│When to Use
─────────────────┼────────┼────────┼────────┼─────────────┼──────────────────────
Linear Regression│ ⚡⚡⚡  │   ⭐⭐  │  Large │  ✓ Excellent│ Baseline, linear data
Ridge/Lasso      │ ⚡⚡⚡  │   ⭐⭐  │  Large │  ✓ Excellent│ Many features
Polynomial       │ ⚡⚡   │  ⭐⭐⭐ │ Medium │  ✓ Good     │ Curved relationships
Logistic Reg.    │ ⚡⚡⚡  │   ⭐⭐⭐ │  Large │  ✓ Excellent│ Classification, probs
Decision Tree    │ ⚡⚡⚡  │  ⭐⭐⭐ │  Large │  ✓ Excellent│ Interpretability
Random Forest    │ ⚡⚡   │  ⭐⭐⭐⭐│  Large │  ◐ Fair     │ Default structured
SVM              │ ⚡    │  ⭐⭐⭐⭐│ Medium │  ✗ Poor     │ High-dimensional
XGBoost          │ ⚡    │  ⭐⭐⭐⭐⭐│ Medium │  ◐ Fair     │ Structured, accuracy
LightGBM         │ ⚡⚡   │  ⭐⭐⭐⭐⭐│  Large │  ◐ Fair     │ Large structured
CatBoost         │ ⚡    │  ⭐⭐⭐⭐⭐│ Medium │  ◐ Fair     │ Categorical features
KNN              │ ⚡    │  ⭐⭐⭐ │ Medium │  ✗ Poor     │ Small datasets
Naive Bayes      │ ⚡⚡⚡  │  ⭐⭐⭐ │  Large │  ✓ Excellent│ NLP, text
Neural Networks  │ ◐     │  ⭐⭐⭐⭐⭐│  Large │  ✗ Poor     │ Unstructured, deep
─────────────────┼────────┼────────┼────────┼─────────────┼──────────────────────
K-Means          │ ⚡⚡⚡  │   N/A  │  Large │  ✓ Excellent│ Segmentation, quick
DBSCAN           │ ⚡⚡   │   N/A  │ Medium │  ◐ Fair     │ Anomalies, arbitrary
GMM              │ ⚡⚡   │   N/A  │ Medium │  ✓ Good     │ Probabilistic clusters
PCA              │ ⚡⚡⚡  │   N/A  │  Large │  ✓ Good     │ Preprocessing
t-SNE            │ ⚡    │   N/A  │ Small  │  ✓ Excellent│ Visualization
UMAP             │ ⚡⚡   │   N/A  │  Large │  ✓ Excellent│ Vis + scale


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA CHARACTERISTICS GUIDE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

DATASET SIZE:
├─ Tiny (<1K):         KNN, SVM (any), Simple models
├─ Small (1K-10K):     Any supervised + some neural networks
├─ Medium (10K-1M):    Tree-based (RF, XGB), Neural networks
└─ Large (>1M):        LightGBM, Logistic Regression, KNN (approx)

FEATURE COUNT:
├─ Low (<10):          Any model
├─ Medium (10-100):    Tree-based, Linear models
├─ High (100-1K):      PCA reduce first, SVM, Linear models
└─ Very High (>1K):    PCA, Feature selection, Linear models

FEATURE TYPES:
├─ All numerical:      Any model
├─ All categorical:    Naive Bayes, Logistic Reg, CatBoost
└─ Mixed:              Tree-based, Neural networks

DATA DISTRIBUTION:
├─ Imbalanced classes: XGBoost (scale_pos_weight), Random Forest, SVM (class_weight)
├─ Outliers present:   Tree-based, Logistic Reg (scale), DBSCAN
└─ Missing values:     XGBoost/CatBoost (built-in), Median imputation

PROBLEM CHARACTERISTICS:
├─ Linear:             Linear Regression, Logistic Regression
├─ Non-linear:         Tree-based, SVM (RBF), Neural networks
├─ Complex patterns:    XGBoost, Neural networks, Ensemble
├─ Real-time <10ms:    Linear models, KNN (indexed)
└─ Interpretability:   Linear models, Decision trees

"""

print(decision_guide)

# Create visual flowchart
print("\n" + "=" * 100)
print("CREATING VISUAL FLOWCHART...")
print("=" * 100)

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(5, 13.5, 'ML Algorithm Selection Flowchart', fontsize=20, weight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Main question
start_box = FancyBboxPatch((3.5, 12), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2)
ax.add_patch(start_box)
ax.text(5, 12.4, 'Do you have labels?', fontsize=12, weight='bold', ha='center')

# Left branch - SUPERVISED
ax.arrow(4, 12, -1.5, -1, head_width=0.15, head_length=0.15, fc='green', ec='green')
supervised_box = FancyBboxPatch((0.5, 9.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(supervised_box)
ax.text(1.75, 9.9, 'SUPERVISED', fontsize=11, weight='bold', ha='center')

# Regression branch
ax.arrow(1.75, 9.5, 0, -1, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
reg_box = FancyBboxPatch((0.5, 7.8), 2.5, 0.7, boxstyle="round,pad=0.08",
                        edgecolor='blue', facecolor='lightcyan', linewidth=1.5)
ax.add_patch(reg_box)
ax.text(1.75, 8.15, 'Regression', fontsize=10, ha='center')

# Regression options
ax.text(0.2, 7.3, '• Linear Reg', fontsize=8)
ax.text(0.2, 6.95, '• Random Forest', fontsize=8)
ax.text(0.2, 6.6, '• XGBoost', fontsize=8)
ax.text(0.2, 6.25, '• Neural Net', fontsize=8)

# Classification branch
ax.arrow(1.75, 7.8, 0, -1, head_width=0.15, head_length=0.15, fc='purple', ec='purple')
class_box = FancyBboxPatch((0.5, 6), 2.5, 0.7, boxstyle="round,pad=0.08",
                          edgecolor='purple', facecolor='plum', linewidth=1.5)
ax.add_patch(class_box)
ax.text(1.75, 6.35, 'Classification', fontsize=10, ha='center')

# Classification options
ax.text(0.2, 5.5, '• Logistic Reg', fontsize=8)
ax.text(0.2, 5.15, '• Decision Tree', fontsize=8)
ax.text(0.2, 4.8, '• Random Forest', fontsize=8)
ax.text(0.2, 4.45, '• SVM / XGBoost', fontsize=8)

# Right branch - UNSUPERVISED
ax.arrow(6, 12, 1.5, -1, head_width=0.15, head_length=0.15, fc='orange', ec='orange')
unsupervised_box = FancyBboxPatch((6.5, 9.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                 edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(unsupervised_box)
ax.text(7.75, 9.9, 'UNSUPERVISED', fontsize=11, weight='bold', ha='center')

# Clustering branch
ax.arrow(7, 9.5, -0.75, -1, head_width=0.15, head_length=0.15, fc='red', ec='red')
clust_box = FancyBboxPatch((5.5, 7.8), 2.5, 0.7, boxstyle="round,pad=0.08",
                          edgecolor='red', facecolor='lightcoral', linewidth=1.5)
ax.add_patch(clust_box)
ax.text(6.75, 8.15, 'Clustering', fontsize=10, ha='center')

# Clustering options
ax.text(5.2, 7.3, '• K-Means', fontsize=8)
ax.text(5.2, 6.95, '• DBSCAN', fontsize=8)
ax.text(5.2, 6.6, '• GMM', fontsize=8)
ax.text(5.2, 6.25, '• Hierarchical', fontsize=8)

# Dimensionality reduction branch
ax.arrow(8.5, 9.5, 0.75, -1, head_width=0.15, head_length=0.15, fc='brown', ec='brown')
dim_red_box = FancyBboxPatch((8.5, 7.8), 2.5, 0.7, boxstyle="round,pad=0.08",
                            edgecolor='brown', facecolor='bisque', linewidth=1.5)
ax.add_patch(dim_red_box)
ax.text(9.75, 8.15, 'Dimensionality', fontsize=10, ha='center')

# Dim reduction options
ax.text(8.2, 7.3, '• PCA', fontsize=8)
ax.text(8.2, 6.95, '• t-SNE', fontsize=8)
ax.text(8.2, 6.6, '• UMAP', fontsize=8)
ax.text(8.2, 6.25, '• Autoencoder', fontsize=8)

# Add legend
legend_y = 5.5
ax.text(0.2, legend_y, '⭐ Rating Legend:', fontsize=9, weight='bold')
ax.text(0.2, legend_y-0.4, '⭐⭐⭐⭐⭐ = State-of-the-art, use first', fontsize=8)
ax.text(0.2, legend_y-0.7, '⭐⭐⭐⭐ = Excellent, very reliable', fontsize=8)
ax.text(0.2, legend_y-1.0, '⭐⭐⭐ = Good, solid choice', fontsize=8)
ax.text(0.2, legend_y-1.3, '⭐⭐ = Niche use, limited applications', fontsize=8)

# Add tips
tips_y = 3.5
ax.text(5, tips_y, 'PRO TIPS:', fontsize=10, weight='bold')
ax.text(5, tips_y-0.4, '1. Start with simplest model (baseline)', fontsize=8)
ax.text(5, tips_y-0.7, '2. Use cross-validation for evaluation', fontsize=8)
ax.text(5, tips_y-1.0, '3. Hyperparameter tuning comes after baseline', fontsize=8)
ax.text(5, tips_y-1.3, '4. Ensemble methods often beat single models', fontsize=8)
ax.text(5, tips_y-1.6, '5. Feature engineering > Model selection', fontsize=8)

plt.tight_layout()
plt.savefig('ml_selection_flowchart.png', dpi=150, bbox_inches='tight')
print("✅ Saved ml_selection_flowchart.png")

```

Now let me create the **final comprehensive selection code with examples**:

```python
print("\n" + "=" * 100)
print("ALGORITHM SELECTION EXAMPLES BY PROBLEM TYPE")
print("=" * 100)

# ===========================
# EXAMPLE 1: REGRESSION
# ===========================
print("\n" + "=" * 100)
print("EXAMPLE 1: HOUSE PRICE PREDICTION (REGRESSION)")
print("=" * 100)

regression_guide = """
PROBLEM:
  Input: Square feet, bedrooms, bathrooms, location, age
  Output: House price (continuous)
  Dataset: 10,000 houses

ALGORITHM SELECTION LOGIC:

Step 1: Check data size
  ✓ 10,000 samples → Medium size, any model OK

Step 2: Check relationship
  → Likely non-linear (more sqft = higher price, but with diminishing returns)

Step 3: Need interpretability?
  → YES - Realtors need to understand price drivers

RECOMMENDED (in order):
  1️⃣  RANDOM FOREST ⭐⭐⭐⭐⭐
      • Feature importance shows what drives prices
      • Handles non-linear relationships
      • Good baseline with minimal tuning
      • RMSE: ~$45,000

  2️⃣  XGBOOST ⭐⭐⭐⭐⭐
      • Better accuracy than Random Forest
      • RMSE: ~$38,000
      • More tuning required
      • Slower to train

  3️⃣  LINEAR REGRESSION + POLYNOMIAL ⭐⭐⭐
      • Simpler but less accurate
      • RMSE: ~$65,000
      • Fastest training & prediction
      • Better interpretability

  4️⃣  NEURAL NETWORKS ⭐⭐
      • Overkill for this dataset
      • Needs >100K samples for full potential
      • Hard to interpret

NOT RECOMMENDED:
  ❌ SVM - No particular advantage here
  ❌ KNN - Would work but slow on large datasets

FINAL CHOICE: Random Forest
  Why: Best balance of accuracy, speed, and interpretability
"""
print(regression_guide)

# ===========================
# EXAMPLE 2: CLASSIFICATION
# ===========================
print("\n" + "=" * 100)
print("EXAMPLE 2: CREDIT CARD FRAUD DETECTION (CLASSIFICATION)")
print("=" * 100)

classification_guide = """
PROBLEM:
  Input: Transaction amount, merchant, location, time, etc.
  Output: Fraud? (Yes/No)
  Dataset: 100,000 transactions, 2% fraud

ALGORITHM SELECTION LOGIC:

Step 1: Check data size
  ✓ 100,000 samples → Large size, need scalability

Step 2: Check class balance
  ⚠️  2% fraud vs 98% normal → IMBALANCED
      Solution: Use class_weight='balanced' or scale_pos_weight

Step 3: Need probability scores?
  ✓ YES - Want confidence, not just yes/no

Step 4: Real-time prediction?
  ✓ YES - Must predict in <10ms

RECOMMENDED (in order):
  1️⃣  XGBOOST with scale_pos_weight ⭐⭐⭐⭐⭐
      • Handles imbalance with scale_pos_weight
      • Accuracy: ~95%
      • ROC-AUC: ~0.95
      • Gets probability scores
      • Kaggle standard for this task

  2️⃣  LIGHTGBM ⭐⭐⭐⭐⭐
      • Faster training than XGBoost
      • Similar accuracy
      • Better for 100K+ samples
      • Recall (catch fraud): 85-90%

  3️⃣  RANDOM FOREST with class_weight ⭐⭐⭐⭐
      • Good baseline
      • Handles imbalance
      • Accuracy: ~92%
      • Slower prediction than boosting

  4️⃣  LOGISTIC REGRESSION ⭐⭐⭐
      • Fast training & prediction
      • Interpretable coefficients
      • Accuracy: ~88%
      • Good baseline for comparison

NOT RECOMMENDED:
  ❌ Decision Tree - Prone to overfitting with imbalanced data
  ❌ Naive Bayes - Assumes feature independence (unrealistic)
  ❌ KNN - Too slow for 100K samples

FINAL CHOICE: XGBoost with scale_pos_weight
  Why: Industry standard, handles imbalance, gets probabilities
  
IMPORTANT: Set scale_pos_weight = (normal_count / fraud_count) = 49
  This penalizes fraud misclassification 49x more than false alarms
"""
print(classification_guide)

# ===========================
# EXAMPLE 3: CLUSTERING
# ===========================
print("\n" + "=" * 100)
print("EXAMPLE 3: CUSTOMER SEGMENTATION (CLUSTERING)")
print("=" * 100)

clustering_guide = """
PROBLEM:
  Input: Customer spending, visit frequency, product categories, RFM metrics
  Output: Customer segments for marketing
  Dataset: 50,000 customers
  Goal: Unknown # of groups

ALGORITHM SELECTION LOGIC:

Step 1: Do we know K?
  ✗ NO - We want to discover segments automatically

Step 2: Cluster shape?
  → Likely spherical/elliptical (spending amount vs frequency)

Step 3: Outlier handling needed?
  ✓ YES - Some outlier customers

Step 4: Interpretability?
  ✓ YES - Marketing team needs to understand segments

RECOMMENDED (in order):
  1️⃣  K-MEANS ⭐⭐⭐⭐⭐
      • First find optimal K using elbow method
      • Silhouette Score: 0.65
      • Get 3-5 clear customer groups
      • Fast: O(nkd) complexity
      • Most popular for segmentation
      • Use k-means++ initialization

  2️⃣  DBSCAN ⭐⭐⭐⭐
      • Automatically finds K
      • Handles outliers well
      • Better for non-spherical clusters
      • Need to tune eps and min_samples
      • More complex but sometimes better results

  3️⃣  GAUSSIAN MIXTURE MODELS ⭐⭐⭐
      • Get probability of segment membership
      • Use BIC/AIC for optimal components
      • Can have soft assignments (70% Group A, 30% Group B)
      • Slower than K-Means

NOT RECOMMENDED:
  ❌ Hierarchical - Slow with 50K customers
  ❌ PCA - Wrong goal (we want clusters, not dimensionality reduction)

FINAL CHOICE: K-Means
  Why: Simplest, fastest, most interpretable, proven on millions of customers
  
WORKFLOW:
  1. Use elbow method to find K (likely 3-5)
  2. Train K-Means
  3. Analyze cluster centers
  4. Create marketing strategy per segment
"""
print(clustering_guide)

# ===========================
# EXAMPLE 4: VISUALIZATION
# ===========================
print("\n" + "=" * 100)
print("EXAMPLE 4: HIGH-DIMENSIONAL DATA VISUALIZATION (UNSUPERVISED)")
print("=" * 100)

visualization_guide = """
PROBLEM:
  Input: Single-cell RNA expression (3000 genes)
  Goal: Visualize cell types in 2D for publication
  Dataset: 10,000 cells

ALGORITHM SELECTION LOGIC:

Step 1: Dataset size?
  ✓ 10,000 samples → t-SNE might work, but UMAP better

Step 2: Need to preserve what?
  → Local structure (which cells are most similar)
  → Want to see cell type clusters

Step 3: Publication quality?
  ✓ YES - Need clear, reproducible visualization

RECOMMENDED (in order):
  1️⃣  UMAP ⭐⭐⭐⭐
      • Training: 10 seconds (vs 5 mins for t-SNE)
      • Preserves global structure better
      • Reproducible results
      • Great for 10K-100K cells
      • Publication standard

  2️⃣  t-SNE ⭐⭐⭐⭐
      • Beautiful visualizations
      • Better local structure preservation
      • SLOW (5-10 minutes for 10K)
      • Non-deterministic (varies between runs)
      • Best for <5K samples

  3️⃣  PCA + t-SNE ⭐⭐⭐
      • PCA first (fast), then t-SNE
      • Reduces t-SNE computation time
      • Good compromise
      • Keeps more global structure than t-SNE alone

NOT RECOMMENDED:
  ❌ PCA alone - Doesn't show cell type clusters well
  ❌ Regular 2D projection - Too simplistic

FINAL CHOICE: UMAP
  Why: 10x faster than t-SNE, preserves structure, reproducible
  
PYTHON CODE:
  import umap
  
  # Reduce to 50D PCA first (optional but helps)
  from sklearn.decomposition import PCA
  pca = PCA(n_components=50)
  X_pca = pca.fit_transform(X)  # 3000D → 50D
  
  # UMAP visualization
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
  X_umap = reducer.fit_transform(X_pca)  # 50D → 2D
  
  plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cell_types, cmap='tab20')
  plt.title('Cell Types - UMAP Visualization')
"""
print(visualization_guide)

print("\n" + "=" * 100)
print("ALGORITHM SELECTION COMPLETE!")
print("=" * 100)
```

This is **PART 3 complete** with:

✅ **K-Means Clustering** - Full customer segmentation example  
✅ **DBSCAN** - Anomaly detection with geographic data  
✅ **Gaussian Mixture Models** - Probabilistic clustering  
✅ **PCA** - Gene expression analysis  
✅ **t-SNE** - Detailed visualization  
✅ **UMAP** - Modern fast alternative  
✅ **Complete Decision Flowchart** - Visual + text guide  
✅ **4 Real-World Examples** - Regression, Classification, Clustering, Visualization  

---

Would you like me to continue with:

1. **⏰ Time Series Algorithms** (ARIMA, Prophet, LSTM)
2. **🧠 Deep Learning (CNNs, RNNs, Transformers)** - with detailed code
3. **📚 Feature Engineering Techniques**
4. **🎯 Hyperparameter Tuning Guide**
5. **📊 Model Evaluation & Validation** - Cross-validation, metrics, etc.

Which one should I focus on? 🚀

# 🚀 PART 4: Time Series, Deep Learning & Advanced Topics

---

# ⏰ TIME SERIES FORECASTING

## 15. ARIMA (AutoRegressive Integrated Moving Average) ⭐⭐⭐⭐

### **What is it?**
Statistical model for time series forecasting using past values to predict future.

### **Components:**
```
ARIMA(p, d, q)

p = AR (AutoRegressive)
  → How many past values affect current value
  → Example: Today's stock = f(Yesterday, Day Before)
  
d = I (Integrated)
  → How many times to difference the data
  → Makes non-stationary data stationary
  → Example: If trend exists, difference once
  
q = MA (Moving Average)
  → How many past forecast errors affect current
  → Example: If yesterday's forecast was wrong, adjust today

Example: ARIMA(2,1,1)
  ├─ p=2: Use last 2 values
  ├─ d=1: Difference once (remove trend)
  └─ q=1: Use 1 past error term
```

### **How It Works:**
```
ARIMA Formula:
y'(t) = μ + φ₁×y'(t-1) + φ₂×y'(t-2) + θ₁×ε(t-1) + ε(t)

Where:
  y'(t) = differenced value (if d>0)
  φ = AR coefficients
  θ = MA coefficients
  ε = error terms
```

### **When to Use:**
✅ **Univariate Time Series:** Single variable (price, temperature)  
✅ **Short-term Forecasting:** Days to weeks (not months/years)  
✅ **Stationary or Near-Stationary:** Works best with stable patterns  
✅ **Linear Relationships:** No complex non-linear patterns  

❌ **When NOT to Use:**
- ❌ Multivariate data (multiple variables)
- ❌ Complex non-linear patterns
- ❌ Structural breaks
- ❌ Multiple seasonality

### **Detailed Example: Stock Price Forecasting**

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic stock price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
trend = np.linspace(100, 150, 500)
seasonal = 10 * np.sin(np.arange(500) * 2 * np.pi / 365)
noise = np.random.normal(0, 3, 500)
prices = trend + seasonal + noise

stock_data = pd.DataFrame({
    'date': dates,
    'price': prices
})
stock_data.set_index('date', inplace=True)

print("=" * 70)
print("STOCK PRICE FORECASTING WITH ARIMA")
print("=" * 70)
print(f"Dataset: {len(stock_data)} trading days")
print(f"Price range: ${stock_data['price'].min():.2f} - ${stock_data['price'].max():.2f}")

# ===========================
# STEP 1: STATIONARITY TEST
# ===========================
print("\n" + "=" * 70)
print("STEP 1: CHECKING STATIONARITY (ADF Test)")
print("=" * 70)

def adf_test(series, name=''):
    result = adfuller(series, autolag='AIC')
    print(f'{name}:')
    print(f'  ADF Statistic: {result[0]:.6f}')
    print(f'  p-value: {result[1]:.6f}')
    print(f'  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print(f'  ✅ STATIONARY (reject null hypothesis)')
        return True
    else:
        print(f'  ❌ NON-STATIONARY (fail to reject null hypothesis)')
        return False

is_stationary = adf_test(stock_data['price'], 'Original Series')

# Difference if not stationary
if not is_stationary:
    print("\n  Differencing once...")
    stock_data['price_diff'] = stock_data['price'].diff()
    is_stationary_diff = adf_test(stock_data['price_diff'].dropna(), 'Differenced Series')

# ===========================
# STEP 2: ACF AND PACF ANALYSIS
# ===========================
print("\n" + "=" * 70)
print("STEP 2: ACF/PACF PLOTS FOR P AND Q SELECTION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original series
axes[0, 0].plot(stock_data['price'])
axes[0, 0].set_title('Original Series')
axes[0, 0].set_ylabel('Price ($)')

# Differenced series
axes[0, 1].plot(stock_data['price_diff'].dropna())
axes[0, 1].set_title('Differenced Series')
axes[0, 1].set_ylabel('Price Change')

# ACF
plot_acf(stock_data['price_diff'].dropna(), lags=40, ax=axes[1, 0])
axes[1, 0].set_title('ACF (helps determine q)')

# PACF
plot_pacf(stock_data['price_diff'].dropna(), lags=40, ax=axes[1, 1])
axes[1, 1].set_title('PACF (helps determine p)')

plt.tight_layout()
plt.savefig('arima_acf_pacf.png', dpi=150, bbox_inches='tight')
print("Saved arima_acf_pacf.png")

print("""
ACF/PACF Interpretation:
├─ If ACF cuts off abruptly, q = that lag
├─ If PACF cuts off abruptly, p = that lag
├─ If both decay gradually, might need both AR and MA
└─ Spikes at seasonal lags (365 for daily data) suggest seasonal ARIMA
""")

# ===========================
# STEP 3: AUTO ARIMA SELECTION
# ===========================
print("\n" + "=" * 70)
print("STEP 3: AUTOMATIC ARIMA ORDER SELECTION")
print("=" * 70)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product

# Manual grid search for ARIMA parameters
p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

best_aic = np.inf
best_order = None
results_list = []

print("\nTesting ARIMA parameters...")
for p, d, q in product(p_range, d_range, q_range):
    try:
        model = ARIMA(stock_data['price'], order=(p, d, q))
        fitted = model.fit()
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = (p, d, q)
        results_list.append({'order': (p, d, q), 'AIC': fitted.aic})
    except:
        continue

results_df = pd.DataFrame(results_list).sort_values('AIC').head(10)
print("\nTop 10 ARIMA Orders by AIC:")
print(results_df.to_string(index=False))

print(f"\n✅ Best ARIMA Order: {best_order}")
print(f"   AIC: {best_aic:.2f}")

# ===========================
# STEP 4: TRAIN ARIMA MODEL
# ===========================
print("\n" + "=" * 70)
print("STEP 4: TRAINING FINAL ARIMA MODEL")
print("=" * 70)

# Split data
train_size = int(len(stock_data) * 0.8)
train_data = stock_data['price'][:train_size]
test_data = stock_data['price'][train_size:]

print(f"Training set: {len(train_data)} days")
print(f"Test set: {len(test_data)} days")

# Train ARIMA
arima_model = ARIMA(train_data, order=best_order)
arima_fitted = arima_model.fit()

print("\n" + arima_fitted.summary().as_text())

# ===========================
# STEP 5: FORECASTING
# ===========================
print("\n" + "=" * 70)
print("STEP 5: MAKING PREDICTIONS")
print("=" * 70)

# In-sample predictions (on test set)
forecast = arima_fitted.get_forecast(steps=len(test_data))
forecast_df = forecast.conf_int()
forecast_mean = forecast.predicted_mean

# Calculate metrics
mae = mean_absolute_error(test_data, forecast_mean)
rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
mape = np.mean(np.abs((test_data - forecast_mean) / test_data)) * 100

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# ===========================
# STEP 6: VISUALIZATION
# ===========================
print("\n" + "=" * 70)
print("STEP 6: VISUALIZING RESULTS")
print("=" * 70)

plt.figure(figsize=(15, 8))

# Plot training data
plt.plot(train_data.index, train_data.values, label='Training Data', color='blue', linewidth=2)

# Plot test data
plt.plot(test_data.index, test_data.values, label='Actual Test Data', color='green', linewidth=2)

# Plot forecast
plt.plot(forecast_df.index, forecast_mean.values, label='ARIMA Forecast', 
         color='red', linewidth=2, linestyle='--')

# Confidence interval
plt.fill_between(forecast_df.index,
                 forecast_df.iloc[:, 0], 
                 forecast_df.iloc[:, 1],
                 color='red', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title(f'Stock Price Forecasting with ARIMA{best_order}')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=150, bbox_inches='tight')
print("Saved arima_forecast.png")

# ===========================
# STEP 7: RESIDUAL ANALYSIS
# ===========================
print("\n" + "=" * 70)
print("STEP 7: RESIDUAL DIAGNOSTICS")
print("=" * 70)

fig = arima_fitted.plot_diagnostics(figsize=(15, 10))
plt.tight_layout()
plt.savefig('arima_diagnostics.png', dpi=150, bbox_inches='tight')
print("Saved arima_diagnostics.png")

# Check for autocorrelation in residuals
residuals = arima_fitted.resid
lb_test = acf(residuals, nlags=20, fft=False)
print(f"\nResidual Autocorrelation at lag 1: {lb_test[1]:.4f}")
print(f"  ✓ Good if close to 0 (indicates residuals are white noise)")

# ===========================
# STEP 8: FUTURE FORECASTING
# ===========================
print("\n" + "=" * 70)
print("STEP 8: FORECASTING FUTURE PRICES")
print("=" * 70)

# Retrain on all data
final_model = ARIMA(stock_data['price'], order=best_order)
final_fitted = final_model.fit()

# Forecast next 30 days
future_forecast = final_fitted.get_forecast(steps=30)
future_df = future_forecast.conf_int()
future_mean = future_forecast.predicted_mean

print(f"\nNext 30 Days Forecast:")
print(f"  Mean Price: ${future_mean.mean():.2f}")
print(f"  Expected Range: ${future_df.iloc[:, 0].min():.2f} - ${future_df.iloc[:, 1].max():.2f}")

# Visualize future forecast
plt.figure(figsize=(15, 8))

# Historical data
plt.plot(stock_data.index, stock_data['price'], label='Historical Prices', 
         color='blue', linewidth=2)

# Future forecast
future_dates = pd.date_range(stock_data.index[-1], periods=31, freq='D')[1:]
plt.plot(future_dates, future_mean.values, label='30-Day Forecast', 
         color='red', linewidth=2, linestyle='--')

# Confidence interval
plt.fill_between(future_dates,
                 future_df.iloc[:, 0].values,
                 future_df.iloc[:, 1].values,
                 color='red', alpha=0.2, label='95% CI')

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Stock Price: Historical + 30-Day Forecast')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arima_future_forecast.png', dpi=150, bbox_inches='tight')
print("Saved arima_future_forecast.png")

# ===========================
# STEP 9: SEASONAL ARIMA
# ===========================
print("\n" + "=" * 70)
print("STEP 9: SEASONAL ARIMA (SARIMA)")
print("=" * 70)

print("""
If data has SEASONAL patterns:

SARIMA(p,d,q)×(P,D,Q)m

Where:
  (p,d,q) = non-seasonal parameters
  (P,D,Q) = seasonal parameters
  m = seasonal period (365 for daily, 12 for monthly)

Example: SARIMA(1,1,1)×(1,1,1)365
  └─ Weekly pattern + yearly seasonality

Advantages over ARIMA:
  ✓ Captures seasonal patterns
  ✓ Better forecasts with seasonality
  ✓ Same implementation (just different parameters)
""")

# ===========================
# STEP 10: COMPARISON WITH OTHER METHODS
# ===========================
print("\n" + "=" * 70)
print("STEP 10: ARIMA VS OTHER METHODS")
print("=" * 70)

comparison = pd.DataFrame({
    'Method': ['ARIMA', 'Exponential Smoothing', 'Prophet', 'LSTM', 'Random Forest'],
    'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡', '⚡⚡'],
    'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐'],
    'Interpretability': ['✓ Good', '✓ Good', '✓ Good', '✗ Poor', '◐ Fair'],
    'Seasonality': ['With SARIMA', 'Built-in', 'Built-in', 'Can learn', 'No'],
    'Data Requirement': ['Small', 'Small', 'Medium', 'Large', 'Medium'],
    'Non-linear': ['No', 'No', 'No', 'Yes', 'Yes']
})

print(comparison.to_string(index=False))
```

---

## 16. Prophet (Facebook) ⭐⭐⭐⭐⭐

### **What is it?**
Robust forecasting tool for time series with **seasonality, trends, and holidays**.

### **Why Prophet?**
✅ **Handles Missing Data:** No need for imputation  
✅ **Seasonality:** Multiple seasonal patterns  
✅ **Holiday Effects:** Built-in support  
✅ **Changepoints:** Detects trend changes  
✅ **Robust:** Works with messy real-world data  
✅ **Interpretable:** Clear trend + seasonality components  

### **Detailed Example: Website Traffic Forecasting**

```python
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate website traffic data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=730, freq='D')

# Traffic components
base_traffic = 1000
trend = np.linspace(0, 500, 730)  # Growing trend
weekly_seasonal = 200 * np.sin(np.arange(730) * 2 * np.pi / 7)
yearly_seasonal = 300 * np.sin(np.arange(730) * 2 * np.pi / 365)
noise = np.random.normal(0, 50, 730)

traffic = base_traffic + trend + weekly_seasonal + yearly_seasonal + noise

prophet_data = pd.DataFrame({
    'ds': dates,
    'y': np.maximum(traffic, 100)  # No negative traffic
})

print("=" * 70)
print("WEBSITE TRAFFIC FORECASTING WITH PROPHET")
print("=" * 70)
print(f"Dataset: {len(prophet_data)} days")
print(f"Traffic range: {prophet_data['y'].min():.0f} - {prophet_data['y'].max():.0f} visits/day")

# ===========================
# BASIC PROPHET MODEL
# ===========================
print("\n" + "=" * 70)
print("BASIC PROPHET MODEL")
print("=" * 70)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)
model.fit(prophet_data)

# Forecast 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

print("\nForecast Summary:")
print(f"  Mean forecast: {forecast['yhat'].tail(90).mean():.0f} visits/day")
print(f"  Max forecast: {forecast['yhat'].tail(90).max():.0f} visits/day")
print(f"  Min forecast: {forecast['yhat'].tail(90).min():.0f} visits/day")

# Plot
model.plot(forecast)
plt.title('Website Traffic Forecast with Prophet')
plt.xlabel('Date')
plt.ylabel('Daily Visits')
plt.savefig('prophet_forecast.png', dpi=150, bbox_inches='tight')
print("Saved prophet_forecast.png")

# Component analysis
fig = model.plot_components(forecast)
plt.tight_layout()
plt.savefig('prophet_components.png', dpi=150, bbox_inches='tight')
print("Saved prophet_components.png")

# ===========================
# ADVANCED: HOLIDAYS & EVENTS
# ===========================
print("\n" + "=" * 70)
print("MODELING HOLIDAYS AND SPECIAL EVENTS")
print("=" * 70)

holidays = pd.DataFrame({
    'holiday': 'Black Friday',
    'ds': pd.to_datetime(['2020-11-27', '2021-11-26']),
    'lower_window': 0,
    'upper_window': 1,  # Traffic boost for 1 day after
})

model_holidays = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays
)
model_holidays.add_seasonality(name='monthly', period=30, fourier_order=5)
model_holidays.fit(prophet_data)

forecast_holidays = model_holidays.predict(future)

print("\nHoliday Effects:")
print(f"  Black Friday impact: +{forecast_holidays['holidays'].max():.0f} visits (estimated)")

# Visualize with holidays
model_holidays.plot(forecast_holidays)
plt.title('Traffic with Holiday Effects')
plt.savefig('prophet_holidays.png', dpi=150, bbox_inches='tight')
print("Saved prophet_holidays.png")

# ===========================
# HYPERPARAMETER TUNING
# ===========================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

params_to_test = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1, 10],
    'seasonality_mode': ['additive', 'multiplicative']
}

results = []

for change_prior in params_to_test['changepoint_prior_scale']:
    for season_prior in params_to_test['seasonality_prior_scale']:
        for season_mode in params_to_test['seasonality_mode']:
            try:
                m = Prophet(
                    changepoint_prior_scale=change_prior,
                    seasonality_prior_scale=season_prior,
                    seasonality_mode=season_mode,
                    interval_width=0.95
                )
                
                # Split data
                train_size = int(len(prophet_data) * 0.8)
                train = prophet_data[:train_size]
                test = prophet_data[train_size:]
                
                m.fit(train)
                future_test = m.make_future_dataframe(periods=len(test))
                forecast_test = m.predict(future_test)
                
                # Calculate error
                forecast_test_vals = forecast_test['yhat'].iloc[train_size:train_size+len(test)].values
                mape = np.mean(np.abs((test['y'].values - forecast_test_vals) / test['y'].values)) * 100
                
                results.append({
                    'changepoint_prior': change_prior,
                    'seasonality_prior': season_prior,
                    'seasonality_mode': season_mode,
                    'MAPE': mape
                })
            except:
                pass

results_df = pd.DataFrame(results).sort_values('MAPE').head(5)
print("\nTop 5 Parameter Combinations:")
print(results_df.to_string(index=False))

# ===========================
# PROPHET VS ARIMA
# ===========================
print("\n" + "=" * 70)
print("PROPHET vs ARIMA COMPARISON")
print("=" * 70)

comparison_data = pd.DataFrame({
    'Aspect': [
        'Ease of Use',
        'Handling Missing Data',
        'Seasonality',
        'Holiday Effects',
        'Automatic Tuning',
        'Interpretability',
        'Speed',
        'Accuracy on Clean Data',
        'Accuracy on Messy Data',
        'Forecasting Horizon'
    ],
    'Prophet': [
        'Very Easy',
        'Excellent',
        'Excellent',
        'Built-in',
        'Automatic',
        'Excellent',
        'Medium',
        'Good',
        'Excellent',
        'Long-term'
    ],
    'ARIMA': [
        'Moderate',
        'Requires Imputation',
        'SARIMA needed',
        'Manual',
        'Grid Search',
        'Good',
        'Fast',
        'Excellent',
        'Fair',
        'Short-term'
    ]
})

print(comparison_data.to_string(index=False))

print("""
WHEN TO USE:

Prophet:
  ✓ Real business data (messy, missing values)
  ✓ Multiple seasonal patterns
  ✓ Holiday/event impacts matter
  ✓ Non-technical stakeholders
  ✓ Long-term forecasting (months/years)

ARIMA:
  ✓ Clean, complete time series
  ✓ Short-term forecasting (days/weeks)
  ✓ Statistical rigor needed
  ✓ Univariate data
  ✓ High interpretability
""")
```

---

## 17. LSTM (Long Short-Term Memory) ⭐⭐⭐⭐⭐

### **What is it?**
Deep learning RNN variant that learns **long-term dependencies** in sequences.

### **LSTM vs Standard RNN:**
```
RNN Problem (Vanishing Gradient):
├─ learns from nearby timesteps only
├─ forgets information from distant past
└─ struggles with long sequences

LSTM Solution (Memory Cells):
├─ Remember/forget gates
├─ Long-term and short-term memory
├─ Capture patterns over months/years
└─ Works with long sequences

Example:
  Sequence: [Jan, Feb, Mar, ..., Dec]
  
  RNN: Remembers [Nov, Dec] → forecasts Jan
  LSTM: Remembers [Jan, Feb, ...Dec] → better forecast
```

### **LSTM Architecture:**
```
Input → [Forget Gate] → [Input Gate] → [Output Gate] → Output
          (what to forget)  (what to add)  (what to output)
```

### **Detailed Example: Stock Price with LSTM**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic stock data with trend and volatility
np.random.seed(42)
n_days = 1000
dates = pd.date_range('2019-01-01', periods=n_days, freq='D')

# Generate realistic stock prices with trend, seasonality, and volatility
trend = np.linspace(100, 150, n_days)
seasonal = 10 * np.sin(np.arange(n_days) * 2 * np.pi / 365)
volatility = np.cumsum(np.random.normal(0, 1, n_days))  # Random walk
prices = trend + seasonal + volatility

# Add some trend changes
prices[500:] += 20  # Price jump at day 500

stock_prices = pd.DataFrame({
    'date': dates,
    'price': prices
})

print("=" * 70)
print("STOCK PRICE FORECASTING WITH LSTM")
print("=" * 70)
print(f"Dataset: {len(stock_prices)} days")
print(f"Price range: ${stock_prices['price'].min():.2f} - ${stock_prices['price'].max():.2f}")

# ===========================
# STEP 1: DATA PREPARATION
# ===========================
print("\n" + "=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)

# Normalize data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(stock_prices[['price']])

print(f"Original price range: {stock_prices['price'].min():.2f} - {stock_prices['price'].max():.2f}")
print(f"Scaled range: {scaled_prices.min():.4f} - {scaled_prices.max():.4f}")

# Create sequences for LSTM
look_back = 60  # Use last 60 days to predict next day

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, look_back)

print(f"\nSequence Creation:")
print(f"  Look-back window: {look_back} days")
print(f"  X shape: {X.shape}")  # (n_samples, look_back)
print(f"  y shape: {y.shape}")  # (n_samples,)

# Reshape for LSTM input [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
print(f"  X reshaped: {X.shape}")

# Split into train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nData Split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# ===========================
# STEP 2: BUILD LSTM MODEL
# ===========================
print("\n" + "=" * 70)
print("STEP 2: BUILDING LSTM MODEL")
print("=" * 70)

model = Sequential([
    # First LSTM layer with return sequences
    LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    # Third LSTM layer (no return sequences)
    LSTM(units=25),
    Dropout(0.2),
    
    # Dense output layer
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print(model.summary())

# ===========================
# STEP 3: TRAIN MODEL
# ===========================
print("\n" + "=" * 70)
print("STEP 3: TRAINING LSTM MODEL")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"✅ Training complete!")
print(f"  Final loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Model Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.set_title('Model MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
print("Saved lstm_training_history.png")

# ===========================
# STEP 4: MAKE PREDICTIONS
# ===========================
print("\n" + "=" * 70)
print("STEP 4: MAKING PREDICTIONS")
print("=" * 70)

# Predict on training set
y_train_pred = model.predict(X_train, verbose=0)

# Predict on test set
y_test_pred = model.predict(X_test, verbose=0)

# Inverse transform to original scale
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_train_pred_scaled = scaler.inverse_transform(y_train_pred)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_scaled = scaler.inverse_transform(y_test_pred)

# Calculate metrics
train_mae = mean_absolute_error(y_train_actual, y_train_pred_scaled)
test_mae = mean_absolute_error(y_test_actual, y_test_pred_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_scaled))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_scaled))

train_mape = np.mean(np.abs((y_train_actual - y_train_pred_scaled) / y_train_actual)) * 100
test_mape = np.mean(np.abs((y_test_actual - y_test_pred_scaled) / y_test_actual)) * 100

print(f"\nModel Performance:")
print(f"  Training MAE: ${train_mae:.2f}")
print(f"  Test MAE: ${test_mae:.2f}")
print(f"\n  Training RMSE: ${train_rmse:.2f}")
print(f"  Test RMSE: ${test_rmse:.2f}")
print(f"\n  Training MAPE: {train_mape:.2f}%")
print(f"  Test MAPE: {test_mape:.2f}%")

# ===========================
# STEP 5: VISUALIZATION
# ===========================
print("\n" + "=" * 70)
print("STEP 5: VISUALIZING PREDICTIONS")
print("=" * 70)

# Prepare dates for plotting
train_dates = stock_prices['date'].iloc[look_back:look_back+len(y_train)]
test_dates = stock_prices['date'].iloc[look_back+len(y_train):look_back+len(y_train)+len(y_test)]

plt.figure(figsize=(16, 8))

# Plot training set
plt.plot(train_dates, y_train_actual, label='Training Actual', color='blue', linewidth=2)
plt.plot(train_dates, y_train_pred_scaled, label='Training Predicted', 
         color='lightblue', linewidth=2, linestyle='--')

# Plot test set
plt.plot(test_dates, y_test_actual, label='Test Actual', color='green', linewidth=2)
plt.plot(test_dates, y_test_pred_scaled, label='Test Predicted', 
         color='lightgreen', linewidth=2, linestyle='--')

# Add vertical line separating train/test
plt.axvline(x=train_dates.iloc[-1], color='red', linestyle=':', linewidth=2, label='Train/Test Split')

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('LSTM Stock Price Forecasting')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_predictions.png', dpi=150, bbox_inches='tight')
print("Saved lstm_predictions.png")

# ===========================
# STEP 6: FUTURE FORECASTING
# ===========================
print("\n" + "=" * 70)
print("STEP 6: FORECASTING FUTURE PRICES")
print("=" * 70)

# Use last 60 days to predict next 30 days
last_sequence = scaled_prices[-look_back:].copy()
future_predictions = []

for i in range(30):
    next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0, 0]
    future_predictions.append(next_pred)
    
    # Update sequence: remove first, add prediction
    last_sequence = np.append(last_sequence[1:], next_pred)

# Inverse transform
future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print(f"\nNext 30 Days Forecast:")
print(f"  Current Price: ${stock_prices['price'].iloc[-1]:.2f}")
print(f"  Day 5 Forecast: ${future_predictions_scaled[4, 0]:.2f}")
print(f"  Day 10 Forecast: ${future_predictions_scaled[9, 0]:.2f}")
print(f"  Day 30 Forecast: ${future_predictions_scaled[29, 0]:.2f}")

# Visualize future forecast
future_dates = pd.date_range(stock_prices['date'].iloc[-1], periods=31, freq='D')[1:]

plt.figure(figsize=(16, 8))
plt.plot(stock_prices['date'], stock_prices['price'], label='Historical Prices', linewidth=2)
plt.plot(future_dates, future_predictions_scaled, label='30-Day Forecast', 
         color='red', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Stock Price: Historical + LSTM 30-Day Forecast')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_future_forecast.png', dpi=150, bbox_inches='tight')
print("Saved lstm_future_forecast.png")

# ===========================
# STEP 7: ATTENTION MECHANISM (Advanced)
# ===========================
print("\n" + "=" * 70)
print("STEP 7: ADVANCED - ATTENTION MECHANISM")
print("=" * 70)

print("""
Attention Mechanism:
├─ Allows model to "focus" on important timesteps
├─ LSTM: All timesteps equally important
├─ Attention: Weight recent timesteps more
├─ Result: Better forecasting

Architecture:
  Input → LSTM → Attention → Dense → Output
           (learns features)  (focuses on important)

Example:
  Predicting today's stock:
  ├─ Week 1: attention = 0.05 (old, less important)
  ├─ Week 2: attention = 0.15
  ├─ Week 3: attention = 0.30
  ├─ Week 4: attention = 0.50 (recent, most important)
  └─ Result: Model emphasizes recent trend

Implementation:
  from tensorflow.keras.layers import MultiHeadAttention
  
  attention = MultiHeadAttention(num_heads=4, key_dim=32)
  attended = attention(lstm_output, lstm_output)
""")

# ===========================
# STEP 8: COMPARISON WITH OTHER METHODS
# ===========================
print("\n" + "=" * 70)
print("STEP 8: LSTM VS OTHER TIME SERIES METHODS")
print("=" * 70)

comparison_ts = pd.DataFrame({
    'Method': ['ARIMA', 'Prophet', 'Exponential Smoothing', 'LSTM', 'Transformer'],
    'Data Size': ['Small OK', 'Medium OK', 'Small OK', 'Large Needed', 'Very Large'],
    'Training Time': ['Fast', 'Medium', 'Fast', 'Slow', 'Very Slow'],
    'Accuracy': ['Good', 'Very Good', 'Good', 'Excellent', 'State-of-art'],
    'Interpretability': ['Excellent', 'Excellent', 'Good', 'Poor', 'Poor'],
    'Seasonality': ['SARIMA', 'Built-in', 'Built-in', 'Learns', 'Learns'],
    'Non-linear': ['No', 'No', 'No', 'Yes', 'Yes'],
    'Missing Data': ['Needs imputation', 'Built-in', 'Needs imputation', 'Needs imputation', 'Needs imputation']
})

print(comparison_ts.to_string(index=False))

print("""
WHEN TO USE LSTM:

✓ USE LSTM IF:
  • Large historical dataset (>500 data points)
  • Complex non-linear patterns
  • Multiple variables/multivariate forecasting
  • You have computational resources
  • Accuracy is more important than interpretability

✗ DON'T USE LSTM IF:
  • Small dataset (<100 points)
  • Need interpretable model
  • Simple linear trend (ARIMA better)
  • Need fast training
  • Need to explain to non-technical stakeholders
""")
```

---

# 🧠 DEEP LEARNING FUNDAMENTALS

## 18. Neural Networks & CNNs ⭐⭐⭐⭐⭐

### **What is it?**
Network of interconnected neurons inspired by brain. **Learns non-linear transformations** of input to output.

### **Neural Network Basics:**
```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
    │                │                 │               │
    ● ──────────────► ● ──────────────► ● ──────────────► ● 
    │                │                 │               │
    ●                ●                 ●                ●
    │                │                 │               │
    ●                ●                 ●
    
Neurons connected by weights (w)
Each neuron: output = activation(∑(input × w) + bias)

Common Activation Functions:
├─ ReLU: max(0, x) - Fast, default choice
├─ Sigmoid: 1/(1+e^-x) - Probability output (0-1)
├─ Tanh: (e^x - e^-x)/(e^x + e^-x) - Better gradient (-1 to 1)
└─ Linear: x - Regression output
```

### **Detailed Example: Image Classification (MNIST)**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load handwritten digits dataset
print("=" * 70)
print("DIGIT RECOGNITION WITH NEURAL NETWORKS")
print("=" * 70)

digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset: {X.shape[0]} images, {X.shape[1]} features")
print(f"Image size: 8×8 pixels")
print(f"Classes: {len(np.unique(y))} digits (0-9)")

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Digit: {y[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('digit_examples.png', dpi=150, bbox_inches='tight')
print("Saved digit_examples.png")

# ===========================
# DATA PREPARATION
# ===========================
print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Normalize pixel values to [0, 1]
X = X / 16.0  # Max pixel value is 16

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Pixel range: [{X.min():.3f}, {X.max():.3f}]")

# ===========================
# BUILD NEURAL NETWORK
# ===========================
print("\n" + "=" * 70)
print("BUILDING NEURAL NETWORK")
print("=" * 70)

model = models.Sequential([
    # Input layer: 64 features (8×8 pixels flattened)
    
    # Hidden layers
    layers.Dense(128, activation='relu', input_shape=(64,)),
    layers.Dropout(0.2),  # Drop 20% to prevent overfitting
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    
    # Output layer: 10 classes (digits 0-9)
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)

print("\nModel Architecture:")
print(model.summary())

print("""
Explanation:
├─ Input: 64 neurons (flattened 8×8 image)
├─ Dense(128): 128 neurons × 64 inputs = 8,192 parameters
├─ Dropout(0.2): Randomly disable 20% of neurons during training
├─ Dense(64): 64 neurons × 128 inputs = 8,192 parameters
├─ Dense(32): 32 neurons × 64 inputs = 2,048 parameters
├─ Dense(10, softmax): 10 neurons for digit probabilities
└─ Total: ~18,500 parameters to learn

Softmax output:
├─ Converts logits to probabilities
├─ All sum to 1.0
├─ Example: [0.01, 0.05, 0.80, 0.05, 0.09, ...] = probably digit 2
""")

# ===========================
# TRAIN MODEL
# ===========================
print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print(f"✅ Training complete!")
print(f"  Epochs trained: {len(history.history['loss'])}")
print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Model Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nn_training_curves.png', dpi=150, bbox_inches='tight')
print("Saved nn_training_curves.png")

# ===========================
# EVALUATE MODEL
# ===========================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)

# Accuracy
test_acc = np.mean(y_pred_class == y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=[str(i) for i in range(10)]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Digit Recognition')
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved nn_confusion_matrix.png")

# ===========================
# VISUALIZE PREDICTIONS
# ===========================
print("\n" + "=" * 70)
print("VISUALIZING PREDICTIONS")
print("=" * 70)

# Show predictions vs actual
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
errors_count = 0

for i, ax in enumerate(axes.flat):
    # Find some correctly and incorrectly classified examples
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    pred = y_pred_class[i]
    actual = y_test[i]
    
    if pred == actual:
        title = f'✓ {actual}'
        color = 'green'
    else:
        title = f'✗ {actual} → {pred}'
        color = 'red'
        errors_count += 1
    
    ax.set_title(title, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle(f'Predictions (Red = Errors)', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('nn_sample_predictions.png', dpi=150, bbox_inches='tight')
print(f"Saved nn_sample_predictions.png")
print(f"  Correct: {len(axes.flat) - errors_count}/{len(axes.flat)}")

# ===========================
# CONVOLUTIONAL NEURAL NETWORK (CNN)
# ===========================
print("\n" + "=" * 70)
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("=" * 70)

print("""
CNN Architecture (better for images):

Input (8×8 image)
    ↓
Convolutional Layer 1 (learns features like edges)
    ├─ 32 filters
    ├─ 3×3 kernel
    └─ Learns: edges, corners, patterns
    ↓
MaxPool (2×2)
    └─ Reduces size by 2× (keeps important features)
    ↓
Convolutional Layer 2 (learns combinations)
    ├─ 64 filters
    └─ Learns: shapes, parts of digits
    ↓
Flatten
    └─ Convert 2D to 1D for dense layers
    ↓
Dense Layer (classification)
    ├─ 128 neurons
    └─ Learns: digit patterns
    ↓
Output (10 classes)
    └─ Softmax probabilities

Advantages of CNN:
✓ Parameter sharing: Fewer parameters than dense networks
✓ Local connectivity: Learns spatial patterns
✓ Translational invariance: Recognizes digit even if moved slightly
✓ State-of-the-art: Best for image tasks
""")

# Reshape data for CNN [samples, height, width, channels]
X_train_cnn = X_train.reshape(-1, 8, 8, 1)
X_test_cnn = X_test.reshape(-1, 8, 8, 1)

# Build CNN
cnn_model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nCNN Model:")
print(cnn_model.summary())

# Train CNN
history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=0
)

# Evaluate CNN
y_pred_cnn = cnn_model.predict(X_test_cnn, verbose=0)
y_pred_cnn_class = np.argmax(y_pred_cnn, axis=1)
cnn_acc = np.mean(y_pred_cnn_class == y_test)

print(f"\nCNN Test Accuracy: {cnn_acc:.4f}")

# ===========================
# COMPARISON: DENSE vs CNN
# ===========================
print("\n" + "=" * 70)
print("DENSE NETWORK vs CNN COMPARISON")
print("=" * 70)

comparison_nn = pd.DataFrame({
    'Aspect': [
        'Architecture',
        'Parameters',
        'Training Speed',
        'Accuracy',
        'Spatial Awareness',
        'Interpretability',
        'Image Tasks',
        'Text Tasks'
    ],
    'Dense Network': [
        'Fully connected',
        'High (~18,500)',
        'Fast',
        f'{test_acc:.4f}',
        'No',
        'Very high',
        'Good',
        'Not ideal'
    ],
    'CNN': [
        'Convolutional',
        'Low (~5,000)',
        'Fast',
        f'{cnn_acc:.4f}',
        'Yes (spatial)',
        'Medium',
        'Excellent',
        'Not ideal'
    ]
})

print(comparison_nn.to_string(index=False))

print("""
WHEN TO USE:

Dense Network:
  ✓ Tabular data (spreadsheets)
  ✓ Small images (like MNIST)
  ✓ Time series (non-sequential)
  ✓ Fast inference needed
  ✓ Need interpretability

CNN:
  ✓ Large images (photos, medical scans)
  ✓ Object detection
  ✓ Image classification
  ✓ Medical imaging
  ✓ Computer vision tasks
  ✓ Fewer parameters needed
""")
```

---

## 19. Transformers & Attention ⭐⭐⭐⭐⭐

### **What is it?**
**Modern deep learning architecture** using self-attention instead of RNNs. Powers GPT, BERT, ChatGPT!

### **Why Transformers?**
✅ **Parallel Processing:** Can process entire sequence at once (vs RNN sequential)  
✅ **Long Dependencies:** Attention mechanism captures distant relationships  
✅ **Pre-training:** Can pre-train on huge datasets then fine-tune  
✅ **State-of-the-Art:** Best results on NLP tasks  

### **Transformer vs LSTM:**
```
LSTM:
├─ Processes sequentially: word1 → word2 → word3
├─ Slow: Can't parallelize
└─ Limited memory: ~500 words max

Transformer:
├─ Processes all at once: [word1, word2, word3]
├─ Fast: Can parallelize (GPUs/TPUs)
├─ Long memory: Can handle 2000+ words
└─ Attention: Focus on relevant words

Example - translating "The cat sat on the mat":
LSTM: The(time 1) → cat(time 2) → sat(time 3) → ...
Transformer: Can attend to "cat" from position 5 instantly
```

### **Self-Attention Mechanism:**
```
Question: What should "it" refer to in "The trophy didn't fit in the suitcase because it was too large"?

Self-Attention Process:
1. Embed each word
2. Create Query, Key, Value matrices
3. Attention = softmax(Q × K^T / √d) × V
4. Result: "it" attends to [suitcase: 70%, trophy: 20%, large: 10%]

In context:
├─ "it" (pronoun) → query
├─ All words → keys/values
├─ Attention weights show which words are important
└─ Final representation considers all relevant context
```

### **Detailed Example: Text Classification with Transformers**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Generate text sentiment dataset
print("=" * 70)
print("TEXT SENTIMENT CLASSIFICATION WITH TRANSFORMERS")
print("=" * 70)

# Sample movie reviews
positive_reviews = [
    "This movie was amazing! I loved every minute of it!",
    "Excellent film, great acting and wonderful storyline.",
    "Fantastic! One of the best movies I've ever seen!",
    "Outstanding performance by all actors. Highly recommended.",
    "I was blown away. Absolutely brilliant cinematography.",
    "Best movie ever! Made me laugh and cry.",
    "Incredible! Worth watching multiple times.",
    "Perfect blend of action and emotion. Loved it!"
]

negative_reviews = [
    "Terrible movie. Complete waste of time.",
    "Awful script and bad acting. Very disappointed.",
    "Horrible! One of the worst films I've seen.",
    "Don't bother watching this boring, predictable mess.",
    "Absolutely terrible. Couldn't even finish it.",
    "Dreadful screenplay and pathetic acting.",
    "Unwatchable garbage. Total disaster.",
    "Painfully bad. Regret watching this."
]

# Create dataset
reviews = positive_reviews + negative_reviews
labels = [1]*len(positive_reviews) + [0]*len(negative_reviews)

print(f"Dataset: {len(reviews)} reviews")
print(f"Positive: {sum(labels)}")
print(f"Negative: {len(labels) - sum(labels)}")

# ===========================
# TEXT PREPROCESSING
# ===========================
print("\n" + "=" * 70)
print("TEXT PREPROCESSING")
print("=" * 70)

vocab_size = 1000
max_length = 20

# Tokenize text
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(reviews)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(reviews)

# Pad sequences to same length
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(labels)

print(f"Vocabulary size: {len(tokenizer.word_index)} words")
print(f"Max sequence length: {max_length}")
print(f"X shape: {X.shape}")

# Example tokenization
print(f"\nExample tokenization:")
example = reviews[0]
print(f"Original: '{example}'")
print(f"Tokenized: {sequences[0]}")
print(f"Padded: {X[0]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ===========================
# TRANSFORMER BLOCK
# ===========================
print("\n" + "=" * 70)
print("TRANSFORMER ARCHITECTURE")
print("=" * 70)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Build Transformer model
embed_dim = 32  # Embedding dimension
num_heads = 2   # Number of attention heads
ff_dim = 32     # Feed-forward dimension

inputs = layers.Input(shape=(max_length,))
embedding_layer = layers.Embedding(vocab_size, embed_dim)(inputs)

# Add positional encoding (tells model about word order)
positions = tf.range(start=0, limit=max_length, delta=1)
position_embedding = layers.Embedding(
    input_dim=max_length, output_dim=embed_dim
)(positions)
x = embedding_layer + position_embedding

# Transformer blocks
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)

# Global average pooling
x = layers.GlobalAveragePooling1D()(x)

# Classification head
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\nTransformer Model:")
print(model.summary())

print("""
Architecture Explanation:
├─ Embedding Layer: Converts words to 32-D vectors
├─ Positional Encoding: Adds position information
├─ Transformer Block:
│  ├─ Multi-Head Attention: Learns relationships
│  ├─ Feed-Forward: Non-linear transformations
│  └─ Layer Normalization: Stabilizes training
├─ Global Average Pooling: Summarize sequence
├─ Dense(20): Classification layer
└─ Sigmoid: Binary classification output

Key Components:
├─ Multi-Head Attention: 2 heads, each learns different patterns
├─ Head 1: Might focus on sentiment words
└─ Head 2: Might focus on negations ("not good")
""")

# ===========================
# TRAIN MODEL
# ===========================
print("\n" + "=" * 70)
print("TRAINING TRANSFORMER")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=4,
    validation_split=0.2,
    verbose=0
)

print(f"✅ Training complete!")
print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Transformer Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Transformer Training Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transformer_training.png', dpi=150, bbox_inches='tight')
print("Saved transformer_training.png")

# ===========================
# EVALUATE
# ===========================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test, verbose=0)
y_pred_class = (y_pred > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=['Negative', 'Positive']))

# ===========================
# ATTENTION VISUALIZATION
# ===========================
print("\n" + "=" * 70)
print("VISUALIZING ATTENTION WEIGHTS")
print("=" * 70)

print("""
Attention Visualization shows which words the model focuses on:

Example: "This movie was amazing"
├─ Word "amazing": Positive sentiment
├─ Attention to "amazing": High (100%)
├─ Attention to "movie": Low (30%)
└─ Attention to "was": Low (20%)

This explains model's predictions!
""")

# ===========================
# PRE-TRAINED TRANSFORMERS
# ===========================
print("\n" + "=" * 70)
print("USING PRE-TRAINED TRANSFORMERS")
print("=" * 70)

print("""
Popular Pre-trained Transformers:

1. BERT (Bidirectional Encoder Representations from Transformers)
   ├─ Pre-trained on Wikipedia + Books
   ├─ Best for: Classification, Named Entity Recognition
   └─ Library: transformers.BertModel

2. GPT-2 / GPT-3 (Generative Pre-trained Transformer)
   ├─ Pre-trained on 40GB web text
   ├─ Best for: Text generation, chat
   └─ Library: transformers.GPT2LMHeadModel

3. RoBERTa (Robustly Optimized BERT)
   ├─ Better pre-training than BERT
   ├─ Best for: All BERT tasks, better accuracy
   └─ Library: transformers.RobertaModel

4. T5 (Text-to-Text Transfer Transformer)
   ├─ Unified framework (translation, summarization, QA)
   ├─ Best for: Multiple NLP tasks
   └─ Library: transformers.T5ForConditionalGeneration

5. DistilBERT
   ├─ 40% smaller, 60% faster than BERT
   ├─ Best for: Production (speed/efficiency)
   └─ Library: transformers.DistilBertModel

USAGE EXAMPLE:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize
inputs = tokenizer("This is amazing!", return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Model predictions

# Result: [negative_score, positive_score]
""")
```

---

# 📊 COMPLETE ALGORITHM SELECTION TOOL

```python
print("\n" + "=" * 100)
print("INTERACTIVE ALGORITHM SELECTOR")
print("=" * 100)

class AlgorithmSelector:
    def __init__(self):
        self.algorithms = {
            # Regression
            ('supervised', 'continuous', 'linear'): [
                ('Linear Regression', '⭐⭐⭐⭐', 'Fast, interpretable'),
                ('Ridge/Lasso', '⭐⭐⭐⭐', 'With regularization'),
            ],
            ('supervised', 'continuous', 'nonlinear'): [
                ('Random Forest', '⭐⭐⭐⭐⭐', 'Accurate, interpretable'),
                ('XGBoost', '⭐⭐⭐⭐⭐', 'State-of-art accuracy'),
                ('Neural Networks', '⭐⭐⭐⭐', 'For complex patterns'),
            ],
            
            # Classification
            ('supervised', 'categorical', 'linear'): [
                ('Logistic Regression', '⭐⭐⭐⭐', 'Fast, interpretable'),
                ('SVM', '⭐⭐⭐⭐', 'High-dimensional'),
            ],
            ('supervised', 'categorical', 'nonlinear'): [
                ('Decision Tree', '⭐⭐⭐', 'Interpretable'),
                ('Random Forest', '⭐⭐⭐⭐⭐', 'Reliable baseline'),
                ('XGBoost', '⭐⭐⭐⭐⭐', 'Best accuracy'),
                ('Neural Networks', '⭐⭐⭐⭐', 'Complex patterns'),
            ],
            
            # Unsupervised - Clustering
            ('unsupervised', 'clustering', 'known_k'): [
                ('K-Means', '⭐⭐⭐⭐⭐', 'Fast, scalable'),
            ],
            ('unsupervised', 'clustering', 'unknown_k'): [
                ('DBSCAN', '⭐⭐⭐⭐', 'Arbitrary shapes'),
                ('Gaussian Mixture', '⭐⭐⭐', 'Probabilistic'),
            ],
            
            # Unsupervised - Dimensionality Reduction
            ('unsupervised', 'dimension_reduction', 'numerical'): [
                ('PCA', '⭐⭐⭐⭐⭐', 'Fast, interpretable'),
            ],
            ('unsupervised', 'dimension_reduction', 'visualization'): [
                ('t-SNE', '⭐⭐⭐⭐', 'Beautiful 2D plots'),
                ('UMAP', '⭐⭐⭐⭐', 'Fast, preserves structure'),
            ],
            
            # Time Series
            ('timeseries', 'short_term', 'simple'): [
                ('ARIMA', '⭐⭐⭐', 'Statistical'),
            ],
            ('timeseries', 'medium_term', 'complex'): [
                ('Prophet', '⭐⭐⭐⭐⭐', 'Handles holidays/seasonality'),
            ],
            ('timeseries', 'long_term', 'deep'): [
                ('LSTM', '⭐⭐⭐⭐⭐', 'Deep learning'),
                ('Transformer', '⭐⭐⭐⭐⭐', 'State-of-art'),
            ],
        }
    
    def recommend(self, learning_type, data_type, complexity):
        key = (learning_type, data_type, complexity)
        if key in self.algorithms:
            return self.algorithms[key]
        return [('No match', '?', 'Try different parameters')]
    
    def print_guide(self):
        guide = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ALGORITHM SELECTION QUICK REFERENCE                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

STEP 1: PROBLEM TYPE
├─ Supervised (have labels Y)?
│  ├─ Regression (continuous output)
│  │  ├─ Linear relationship → Linear Regression ⭐⭐⭐⭐
│  │  └─ Non-linear → Random Forest/XGBoost ⭐⭐⭐⭐⭐
│  │
│  └─ Classification (categorical output)
│     ├─ Binary → Logistic Regression / Random Forest
│     └─ Multi-class → XGBoost / Neural Networks
│
├─ Unsupervised (no labels)?
│  ├─ Clustering
│  │  ├─ Know K → K-Means ⭐⭐⭐⭐⭐
│  │  └─ Discover K → DBSCAN ⭐⭐⭐⭐
│  │
│  └─ Dimensionality Reduction
│     ├─ Numerical → PCA ⭐⭐⭐⭐⭐
│     └─ Visualization → UMAP ⭐⭐⭐⭐
│
└─ Time Series?
   ├─ Short-term (days) → ARIMA ⭐⭐⭐
   ├─ Medium-term (weeks/months) → Prophet ⭐⭐⭐⭐⭐
   └─ Long-term (complex) → LSTM / Transformer ⭐⭐⭐⭐⭐


STEP 2: DATA CHARACTERISTICS
├─ Sample size
│  ├─ <1K: Any model OK
│  ├─ 1K-100K: Most models good
│  └─ >100K: Need scalable (LightGBM, Linear, KNN)
│
├─ Feature types
│  ├─ All numerical: Any model
│  ├─ All categorical: Naive Bayes, CatBoost
│  └─ Mixed: Tree-based, Neural Networks
│
├─ Missing data
│  ├─ None: Any model
│  ├─ <5%: Impute or use XGBoost/CatBoost
│  └─ >5%: Prophet (if TS), or filter rows
│
└─ Class imbalance (classification)
   ├─ Balanced: Any model
   ├─ Imbalanced: XGBoost (scale_pos_weight), SMOTE
   └─ Highly imbalanced: Anomaly detection methods


STEP 3: REQUIREMENTS
├─ Speed needed?
│  ├─ Real-time (<10ms): Linear models, KNN (indexed)
│  ├─ Batch (<1s): Tree-based, Neural Networks
│  └─ Offline: Any (prioritize accuracy)
│
├─ Interpretability?
│  ├─ Yes: Linear, Decision Trees, SHAP values
│  ├─ Somewhat: Random Forest, PCA
│  └─ No: Neural Networks, SVM
│
├─ Training resources?
│  ├─ Limited: Linear, Tree-based (no GPU)
│  ├─ Medium: Deep Learning (GPU helpful)
│  └─ Unlimited: Transformers, Large ensembles
│
└─ Accuracy priority?
   ├─ Baseline/MVP: Linear models, simple trees
   ├─ Production: Random Forest, XGBoost
   └─ State-of-art: Neural Networks, Transformers


╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ALGORITHM COMPLEXITY CHART                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Accuracy vs Interpretability Trade-off:

    Accuracy
      ▲
  100%│                    Neural Networks
      │                    Transformers
      │              ●
      │        XGBoost  ●
      │              ● Random Forest
      │         ●
      │    Decision Tree
      │   ● ┐
      │  ╱   └─ Naive Bayes
      │ ╱  ●
   0% ├────────────────────────────────► Interpretability
      Linear Reg   Decision Tree    Neural Networks
      (High)       (Medium)         (Low)


Speed vs Accuracy:

    Accuracy
      ▲
   100%│ ■ Neural Networks
      │ ■ Transformers
      │    ■ XGBoost
      │       ■ Random Forest
      │           ■ Linear Reg
      │              ■ KNN
    0%├────────────────────────────────► Speed
      Slow           Medium           Fast


Dataset Size Recommendations:

    Model
      ▲
   Comp
    lexity│  ╭─ Neural Networks
         │ ╱  Transformers
         │╱   
         │    ╭─ XGBoost
         │   ╱  Random Forest
         │  ╱
         │ ╱─ Linear Models
         │╱
     0%  ├────────────────────────────────► Samples
        <1K    10K    100K   1M    10M


╔═══════════════════════════════════════════════════════════════════════════════╗
║                          REAL-WORLD DECISION TREE                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

BANK CREDIT RISK PREDICTION:
├─ Problem: Regression (risk score) or Classification (approve/deny)?
│  └─ Classification (binary)
│
├─ Data: 100K applications, 50 features (mix of numerical & categorical)
│  └─ XGBoost (handles categorical, large dataset, high accuracy)
│
├─ Requirements
│  ├─ Interpretability: Need to explain to regulators
│  │  └─ Use SHAP values with XGBoost
│  ├─ Speed: Real-time decisions <100ms
│  │  └─ XGBoost is fast enough
│  └─ Accuracy: ~95% is target
│     └─ XGBoost achieves this
│
└─ FINAL: XGBoost + SHAP explanations


NETFLIX MOVIE RECOMMENDATION:
├─ Problem: Unsupervised (no explicit "good" label)
│  └─ Collaborative Filtering or Deep Learning
│
├─ Data: 100M users × 10K movies (sparse matrix)
│  └─ Matrix Factorization or Neural Networks
│
├─ Requirements
│  ├─ Accuracy: >80% hit rate
│  ├─ Speed: <1s prediction per user
│  └─ Scalability: 100M users
│
└─ FINAL: Deep Learning (Embedding layers) + Matrix Factorization


COVID-19 DIAGNOSIS FROM CT SCANS:
├─ Problem: Image Classification (COVID / Normal)
│  └─ Deep Learning (CNN)
│
├─ Data: 10K CT scans (small for deep learning, but images have structure)
│  └─ CNN (ConvNet) or Transfer Learning
│
├─ Requirements
│  ├─ Accuracy: 99%+ (medical application)
│  ├─ Interpretability: Show which regions detected infection
│  │  └─ Use Grad-CAM visualization
│  └─ Speed: Real-time processing
│
└─ FINAL: ResNet50 (transfer learning) + Grad-CAM


AMAZON PRODUCT INVENTORY FORECASTING:
├─ Problem: Time Series Forecasting
│  └─ Multiple seasonal patterns (weekly, yearly)
│
├─ Data: 5 years daily sales (1825 points), with holidays/events
│  └─ Prophet or LSTM
│
├─ Requirements
│  ├─ Accuracy: MAPE <5%
│  ├─ Seasonality: Multiple patterns
│  └─ Interpretability: Management needs to understand
│
└─ FINAL: Prophet (handles seasonality, holidays, interpretable)


FRAUD DETECTION IN REAL-TIME:
├─ Problem: Classification (fraud / normal)
│  └─ Must be fast & accurate
│
├─ Data: Streaming transactions, massive scale
│  └─ XGBoost or Isolation Forest
│
├─ Requirements
│  ├─ Speed: <10ms prediction
│  ├─ Accuracy: Recall >95% (catch fraud)
│  └─ Interpretability: Explain why flagged
│
└─ FINAL: XGBoost with class_weight or Isolation Forest


CUSTOMER SEGMENTATION:
├─ Problem: Unsupervised Clustering
│  └─ Unknown number of segments
│
├─ Data: Customer features (spending, frequency, etc.)
│  └─ K-Means or DBSCAN
│
├─ Requirements
│  ├─ Simplicity: Managers understand segments
│  ├─ Interpretability: Feature importance
│  └─ Scalability: 1M+ customers
│
└─ FINAL: K-Means (with elbow method) + PCA visualization


TEXT SENTIMENT ANALYSIS:
├─ Problem: Classification (positive / negative / neutral)
│  └─ NLP task (text data)
│
├─ Data: 100K reviews (unstructured text)
│  └─ Deep Learning or Pre-trained Transformers
│
├─ Requirements
│  ├─ Accuracy: >85%
│  ├─ Speed: Batch process OK
│  └─ Interpretability: Show influential words
│
└─ FINAL: DistilBERT (pre-trained) + fine-tuning


STOCK PRICE FORECASTING:
├─ Problem: Time Series Regression
│  └─ Predict next value(s)
│
├─ Data: Daily prices (univariate), 10+ years
│  └─ LSTM or Prophet
│
├─ Requirements
│  ├─ Accuracy: R² > 0.7
│  ├─ Interpretability: Low (difficult problem)
│  └─ Speed: Batch processing OK
│
└─ FINAL: LSTM (learns complex patterns)

"""
        print(guide)

selector = AlgorithmSelector()
selector.print_guide()

# Example usage
print("\n" + "=" * 100)
print("EXAMPLE RECOMMENDATIONS")
print("=" * 100)

examples = [
    ('supervised', 'continuous', 'nonlinear'),
    ('supervised', 'categorical', 'linear'),
    ('unsupervised', 'clustering', 'unknown_k'),
    ('timeseries', 'long_term', 'deep'),
]

for example in examples:
    print(f"\nQuery: {example}")
    recommendations = selector.recommend(*example)
    for algo, rating, description in recommendations:
        print(f"  ➜ {algo:<30} {rating}  {description}")

print("\n" + "=" * 100)
print("✅ ALGORITHM SELECTION GUIDE COMPLETE!")
print("=" * 100)
```

---

This completes **PART 4** with:

✅ **ARIMA** - Statistical time series forecasting  
✅ **Prophet** - Robust forecasting with seasonality  
✅ **LSTM** - Deep learning for sequences  
✅ **Dense Neural Networks** - Image classification  
✅ **CNNs** - Convolutional networks  
✅ **Transformers** - Modern state-of-art  
✅ **Complete Algorithm Selector** - Interactive guide  
✅ **Real-world decision trees** - 7 business examples  

---

Would you like me to create:

1. **🎯 Model Evaluation & Validation** - Cross-validation, metrics, A/B testing
2. **🛠️ Feature Engineering Guide** - Advanced techniques with code
3. **⚙️ Hyperparameter Tuning Deep Dive** - Grid search, Bayesian optimization
4. **🚀 Model Deployment & Production** - Docker, APIs, monitoring
5. **📚 Complete End-to-End Projects** - Full pipeline examples

Which would be most helpful? 🚀

# 🎯 COMPLETE ML ALGORITHM SELECTION & IMPLEMENTATION GUIDE

I can see you're working on the MLOps project README. Let me create a **comprehensive, production-ready guide** that extends what you have with actionable code examples.

---

## 📝 CREATE/UPDATE: Algorithm Implementation Guide

````markdown
# Machine Learning Algorithm Selection & Implementation Guide

## 🗺️ Quick Navigation

- [Decision Flowchart](#decision-flowchart)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Time Series](#time-series)
- [Deep Learning](#deep-learning)
- [Production Deployment](#production-deployment)

---

## 🔄 Decision Flowchart

```
START: Do you have labeled data (Y values)?
│
├─ YES → SUPERVISED LEARNING
│   ├─ Continuous output (numbers)?
│   │   ├─ Linear relationship → Linear Regression ⭐⭐⭐⭐
│   │   ├─ Non-linear, structured → Random Forest ⭐⭐⭐⭐⭐
│   │   ├─ Maximum accuracy → XGBoost/LightGBM ⭐⭐⭐⭐⭐
│   │   └─ Images/sequences → Neural Networks ⭐⭐⭐⭐
│   │
│   └─ Categorical output (classes)?
│       ├─ Binary, need probabilities → Logistic Regression ⭐⭐⭐⭐
│       ├─ Structured, high accuracy → XGBoost ⭐⭐⭐⭐⭐
│       ├─ Interpretability needed → Decision Tree ⭐⭐⭐
│       ├─ Text/Images → Neural Networks ⭐⭐⭐⭐
│       └─ High-dimensional → SVM ⭐⭐⭐⭐
│
└─ NO → UNSUPERVISED LEARNING
    ├─ Find groups/clusters?
    │   ├─ Know K clusters → K-Means ⭐⭐⭐⭐⭐
    │   ├─ Discover K → DBSCAN ⭐⭐⭐⭐
    │   └─ Probabilistic → Gaussian Mixture Model ⭐⭐⭐
    │
    └─ Reduce dimensions/visualize?
        ├─ Numerical reduction → PCA ⭐⭐⭐⭐⭐
        ├─ Visualization 2D/3D → t-SNE/UMAP ⭐⭐⭐⭐
        └─ Complex patterns → Autoencoder ⭐⭐⭐
```

---

## 📊 SUPERVISED LEARNING

### 1. Linear Regression ⭐⭐⭐⭐

**When to use:**
- Continuous output (house prices, temperature)
- Linear relationship expected
- Need fast training & inference
- Interpretability important

**Pros:** Fast, interpretable, low computational cost
**Cons:** Only works for linear data

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Example: House Price Prediction
data = pd.DataFrame({
    'square_feet': [1000, 1500, 2000, 2500, 3000],
    'price': [200000, 300000, 400000, 500000, 600000]
})

X = data[['square_feet']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Coefficient: {model.coef_[0]:.2f}")  # Price per square foot
print(f"Intercept: {model.intercept_:.2f}")  # Base price
```

---

### 2. Logistic Regression ⭐⭐⭐⭐

**When to use:**
- Binary classification (Yes/No)
- Need probability scores
- Fast predictions needed
- Interpretable model

**Pros:** Fast, probabilistic output, interpretable
**Cons:** Assumes linear decision boundary

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Example: Email Spam Detection
X_train = [[2, 5], [1, 3], [4, 7], [5, 8]]  # Feature vectors
y_train = [0, 0, 1, 1]  # 0=ham, 1=spam

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_train)
y_proba = model.predict_proba(X_train)  # Probabilities

print(f"Predictions: {y_pred}")
print(f"Probabilities:\n{y_proba}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_proba[:, 1]):.4f}")
```

---

### 3. Random Forest ⭐⭐⭐⭐⭐

**When to use:**
- Structured/tabular data
- High accuracy important
- Need feature importance
- Don't want to tune hyperparameters

**Pros:** Accurate, robust, handles non-linearity, feature importance
**Cons:** Slower than linear models, less interpretable

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Example: Iris Flower Classification
iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Feature importance
model.fit(X, y)
importances = model.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

---

### 4. XGBoost ⭐⭐⭐⭐⭐

**When to use:**
- Kaggle competitions
- Maximum accuracy needed
- Structured data with 1K-10M rows
- Hyperparameter tuning acceptable

**Pros:** Best accuracy, handles missing values, regularization
**Cons:** Slower training, requires tuning

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Example: Credit Card Fraud Detection
# (Assume X_train, X_test, y_train, y_test are loaded)

# Basic model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=99,  # Handle imbalanced data
    random_state=42
)
model.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

---

### 5. Support Vector Machines (SVM) ⭐⭐⭐⭐

**When to use:**
- Text classification
- High-dimensional data
- Small to medium datasets
- Non-linear boundaries (RBF kernel)

**Pros:** Effective in high dims, non-linear with kernels
**Cons:** Slow on large data, needs scaling

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM works best with scaled features
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

---

## 🔍 UNSUPERVISED LEARNING

### 1. K-Means Clustering ⭐⭐⭐⭐⭐

**When to use:**
- Customer segmentation
- Know the number of clusters
- Fast clustering needed
- Large datasets

**Pros:** Fast, scalable, simple
**Cons:** Must specify K, assumes spherical clusters

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Find optimal K using Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.tight_layout()
plt.show()

# Final model
optimal_k = K_range[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)
```

---

### 2. PCA (Dimensionality Reduction) ⭐⭐⭐⭐⭐

**When to use:**
- 100+ features
- Remove multicollinearity
- Visualize high-dimensional data
- Speed up other algorithms

**Pros:** Fast, removes multicollinearity, interpretable
**Cons:** Less interpretable after transformation

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"Variance explained: {cumsum[-1]:.2%}")

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(label='Class')
plt.show()

# Or use n_components='mle' to preserve 95% variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(f"Features reduced from {X.shape[1]} to {X_reduced.shape[1]}")
```

---

### 3. DBSCAN (Density-Based Clustering) ⭐⭐⭐⭐

**When to use:**
- Unknown number of clusters
- Arbitrary cluster shapes
- Anomaly detection
- Geospatial clustering

**Pros:** Discovers K, handles arbitrary shapes, finds outliers
**Cons:** Sensitive to eps parameter

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances = neighbors_fit.kneighbors_graph(X).data
distances = np.sort(distances, )[::-1]

plt.plot(distances)
plt.ylabel('epsilon')
plt.xlabel('Points sorted by distance')
plt.title('K-distance graph for eps selection')
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_outliers = list(clusters).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Outliers: {n_outliers}")
```

---

## ⏰ TIME SERIES FORECASTING

### 1. ARIMA ⭐⭐⭐

**When to use:**
- Univariate time series
- Short-term forecasting
- Stationary data

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load time series data
data = pd.read_csv('stock_prices.csv', index_col='date', parse_dates=True)

# Check stationarity and plot ACF/PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data, lags=40, ax=axes[0])
plot_pacf(data, lags=40, ax=axes[1])
plt.show()

# Fit ARIMA(1, 1, 1)
model = ARIMA(data, order=(1, 1, 1))
fitted = model.fit()

print(fitted.summary())

# Forecast next 30 days
forecast = fitted.get_forecast(steps=30)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data, label='Historical')
plt.plot(forecast_ci.index, forecast.predicted_mean, label='Forecast')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 alpha=0.2)
plt.legend()
plt.show()
```

---

### 2. Prophet (Facebook) ⭐⭐⭐⭐⭐

**When to use:**
- Multiple seasonal patterns
- Holiday effects matter
- Messy real-world data
- Long-term forecasting

```python
from fbprophet import Prophet

# Prepare data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365),
    'y': np.random.randn(365).cumsum() + 100
})

# Define holidays
holidays = pd.DataFrame({
    'holiday': 'Black Friday',
    'ds': pd.to_datetime(['2020-11-27', '2021-11-26']),
    'lower_window': 0,
    'upper_window': 3,
})

# Fit model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays
)
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.show()

# Component analysis
model.plot_components(forecast)
```

---

### 3. LSTM (Deep Learning) ⭐⭐⭐⭐⭐

**When to use:**
- Complex non-linear patterns
- Long sequences (100+ timesteps)
- Large datasets
- Maximum accuracy needed

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Create sequences
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
```

---

## 🧠 DEEP LEARNING

### 1. Convolutional Neural Networks (CNN) ⭐⭐⭐⭐

**When to use:**
- Image classification
- Computer vision tasks
- Pattern recognition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Reshape for CNN [samples, height, width, channels]
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## 🚀 PRODUCTION DEPLOYMENT

### 1. Model Persistence

```python
import joblib
import pickle

# Save model
joblib.dump(model, 'trained_model.pkl')

# Load model
model = joblib.load('trained_model.pkl')
```

### 2. Flask API for Serving

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array([data['features']])
    prediction = model.predict(X)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

### 3. Model Monitoring

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename='model_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def predict_with_monitoring(X):
    prediction = model.predict(X)
    confidence = model.predict_proba(X).max()
    
    logging.info(f"Prediction: {prediction[0]}, Confidence: {confidence:.4f}")
    
    # Alert if confidence too low
    if confidence < 0.6:
        logging.warning(f"Low confidence prediction: {confidence:.4f}")
    
    return prediction, confidence
```

---

## 📈 Model Evaluation Metrics

| Task | Metric | Good Range | Formula |
|------|--------|------------|---------|
| **Regression** | R² | 0.7 - 1.0 | 1 - (SS_res / SS_tot) |
| | RMSE | Lower | √(MSE) |
| | MAE | Lower | Mean(\|y - ŷ\|) |
| **Classification** | Accuracy | 0.8 - 1.0 | Correct / Total |
| | Precision | 0.8 - 1.0 | TP / (TP + FP) |
| | Recall | 0.8 - 1.0 | TP / (TP + FN) |
| | F1 Score | 0.8 - 1.0 | 2 × (Precision × Recall) / (P + R) |
| | ROC-AUC | 0.8 - 1.0 | Area under ROC curve |
| **Clustering** | Silhouette | 0.5 - 1.0 | (b - a) / max(a, b) |
| | Inertia | Lower | Within-cluster sum of squares |

---

## 🎯 Complete Algorithm Comparison

```python
# Create comparison DataFrame
algorithm_comparison = pd.DataFrame({
    'Algorithm': [
        'Linear Regression', 'Logistic Regression', 'Decision Tree',
        'Random Forest', 'XGBoost', 'SVM', 'KNN', 'Naive Bayes',
        'K-Means', 'DBSCAN', 'PCA', 'LSTM', 'CNN'
    ],
    'Type': [
        'Supervised', 'Supervised', 'Supervised',
        'Supervised', 'Supervised', 'Supervised', 'Supervised', 'Supervised',
        'Unsupervised', 'Unsupervised', 'Unsupervised', 'Supervised', 'Supervised'
    ],
    'Speed': [
        '⚡⚡⚡', '⚡⚡⚡', '⚡⚡⚡',
        '⚡⚡', '⚡', '⚡', '⚡', '⚡⚡⚡',
        '⚡⚡⚡', '⚡⚡', '⚡⚡⚡', '⚡', '⚡'
    ],
    'Accuracy': [
        '⭐⭐', '⭐⭐⭐', '⭐⭐⭐',
        '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐',
        'N/A', 'N/A', 'N/A', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'
    ],
    'Interpretability': [
        'Excellent', 'Excellent', 'Excellent',
        'Good', 'Fair', 'Poor', 'Poor', 'Excellent',
        'Excellent', 'Good', 'Good', 'Poor', 'Poor'
    ],
    'Best For': [
        'Linear data', 'Binary classification', 'Quick baselines',
        'Balanced accuracy', 'Maximum accuracy', 'Text/high-dim', 'Small datasets', 'NLP',
        'Segmentation', 'Anomalies', 'Preprocessing', 'Sequences', 'Images'
    ]
})

print(algorithm_comparison.to_string(index=False))
```

---

## ✅ Checklist: Before Training

- [ ] Data cleaned (missing values handled)
- [ ] Features scaled/normalized
- [ ] Train/test split done (80/20)
- [ ] Baseline model created
- [ ] Cross-validation setup
- [ ] Metrics defined
- [ ] Hyperparameters identified
- [ ] Hardware requirements checked
- [ ] Model monitoring plan ready
- [ ] Documentation complete

````

---

## 📋 CREATE: Implementation Checklist

````python
"""
ML Algorithm Implementation Checklist
Complete before deploying any model
"""

PREPROCESSING_CHECKLIST = {
    "Data Quality": [
        "✓ Check for missing values (use .isnull().sum())",
        "✓ Handle missing values (drop, impute, or flag)",
        "✓ Identify and handle outliers (IQR, Z-score)",
        "✓ Check data types (ensure numerical/categorical correct)",
        "✓ Check for duplicates (use .duplicated())",
        "✓ Verify data distribution (use histograms, KDE plots)",
    ],
    
    "Feature Engineering": [
        "✓ Create new features from domain knowledge",
        "✓ Remove low-variance features",
        "✓ Handle categorical variables (encode/one-hot)",
        "✓ Scale numerical features (StandardScaler, MinMaxScaler)",
        "✓ Remove multicollinearity (check VIF > 10)",
        "✓ Create interaction features if needed",
    ],
    
    "Data Splitting": [
        "✓ Split into train/test (80/20 or 70/30)",
        "✓ Use stratified split for imbalanced data",
        "✓ Create validation set for hyperparameter tuning",
        "✓ Ensure no data leakage (fit scaler on train only)",
    ]
}

MODEL_TRAINING_CHECKLIST = {
    "Baseline Model": [
        "✓ Create simple baseline (dummy classifier/regressor)",
        "✓ Establish performance threshold",
        "✓ Compare improvements against baseline",
    ],
    
    "Model Selection": [
        "✓ Try multiple algorithms (start with 3-5)",
        "✓ Use cross-validation (5-fold minimum)",
        "✓ Compare training vs validation performance",
        "✓ Check for overfitting/underfitting",
    ],
    
    "Hyperparameter Tuning": [
        "✓ Use GridSearchCV or RandomizedSearchCV",
        "✓ Define parameter ranges based on domain",
        "✓ Use cross-validation during tuning",
        "✓ Monitor computational resources",
        "✓ Save best model",
    ],
}

EVALUATION_CHECKLIST = {
    "Metrics": [
        "✓ Choose appropriate metrics (not just accuracy!)",
        "✓ Calculate precision, recall, F1 for classification",
        "✓ Check ROC-AUC for binary classification",
        "✓ Use RMSE, MAE for regression",
        "✓ Report metrics on test set",
    ],
    
    "Error Analysis": [
        "✓ Analyze misclassifications (confusion matrix)",
        "✓ Check performance per class",
        "✓ Identify patterns in errors",
        "✓ Test on edge cases",
    ],
    
    "Model Validation": [
        "✓ Use multiple evaluation metrics",
        "✓ Plot learning curves",
        "✓ Check calibration (for probabilistic models)",
        "✓ Test on different data distributions",
    ],
}

PRODUCTION_CHECKLIST = {
    "Model Persistence": [
        "✓ Save trained model (joblib, pickle, or native format)",
        "✓ Save scaler/transformer separately",
        "✓ Document model version",
        "✓ Store hyperparameters",
        "✓ Save training metadata (date, data version, etc.)",
    ],
    
    "Deployment": [
        "✓ Create prediction API (Flask/FastAPI)",
        "✓ Handle input validation",
        "✓ Add error handling",
        "✓ Set up logging",
        "✓ Create health check endpoint",
        "✓ Document API endpoints",
    ],
    
    "Monitoring": [
        "✓ Log predictions and confidences",
        "✓ Monitor prediction latency",
        "✓ Track data drift",
        "✓ Alert on low confidence predictions",
        "✓ Monitor for concept drift",
        "✓ Set up retraining schedule",
    ],
}


def print_checklist():
    """Print complete checklist"""
    for section, items in {
        **PREPROCESSING_CHECKLIST,
        **MODEL_TRAINING_CHECKLIST,
        **EVALUATION_CHECKLIST,
        **PRODUCTION_CHECKLIST
    }.items():
        print(f"\n{'='*60}")
        print(f"📋 {section.upper()}")
        print('='*60)
        for item in items:
            print(f"  {item}")


if __name__ == '__main__':
    print_checklist()
````

---

## 🎯 Quick Recommendation Engine

````python
"""
Interactive Algorithm Recommendation System
Based on problem characteristics
"""

class AlgorithmRecommender:
    def __init__(self):
        self.recommendations = {
            # (learning_type, output_type, data_size, interpretability_need) → algorithms
            ('supervised', 'continuous', 'small', True): [
                ('Linear Regression', '⭐⭐⭐⭐', 'Fast, interpretable'),
                ('Decision Tree', '⭐⭐⭐', 'Very interpretable'),
            ],
            ('supervised', 'continuous', 'medium', False): [
                ('Random Forest', '⭐⭐⭐⭐⭐', 'Accurate, robust'),
                ('XGBoost', '⭐⭐⭐⭐⭐', 'State-of-the-art'),
            ],
            ('supervised', 'continuous', 'large', False): [
                ('XGBoost', '⭐⭐⭐⭐⭐', 'Scales well'),
                ('LightGBM', '⭐⭐⭐⭐⭐', 'Very fast on large data'),
                ('Linear Regression', '⭐⭐⭐⭐', 'If linear relationship'),
            ],
            ('supervised', 'categorical', 'small', True): [
                ('Logistic Regression', '⭐⭐⭐⭐', 'Interpretable probs'),
                ('Decision Tree', '⭐⭐⭐', 'Visual rules'),
            ],
            ('supervised', 'categorical', 'medium', False): [
                ('Random Forest', '⭐⭐⭐⭐⭐', 'Go-to algorithm'),
                ('XGBoost', '⭐⭐⭐⭐⭐', 'Competitive'),
                ('SVM', '⭐⭐⭐⭐', 'If non-linear'),
            ],
            ('supervised', 'categorical', 'large', False): [
                ('XGBoost', '⭐⭐⭐⭐⭐', 'Proven at scale'),
                ('LightGBM', '⭐⭐⭐⭐⭐', 'Faster training'),
                ('Logistic Regression', '⭐⭐⭐⭐', 'Baseline'),
            ],
            ('unsupervised', 'clustering', 'small', True): [
                ('K-Means', '⭐⭐⭐⭐⭐', 'Easy to understand'),
                ('Hierarchical', '⭐⭐⭐⭐', 'Visual dendrogram'),
            ],
            ('unsupervised', 'clustering', 'medium', False): [
                ('K-Means', '⭐⭐⭐⭐⭐', 'Fast, reliable'),
                ('DBSCAN', '⭐⭐⭐⭐', 'Arbitrary shapes'),
            ],
            ('unsupervised', 'dimensionality_reduction', 'any', False): [
                ('PCA', '⭐⭐⭐⭐⭐', 'Fast, interpretable'),
                ('UMAP', '⭐⭐⭐⭐', 'Preserves structure'),
                ('t-SNE', '⭐⭐⭐⭐', 'Best visualization'),
            ],
        }
    
    def recommend(self, learning_type, output_type, data_size, interpretability_needed=False):
        """
        Get algorithm recommendations
        
        Args:
            learning_type: 'supervised' or 'unsupervised'
            output_type: 'continuous', 'categorical', 'clustering', 'dimensionality_reduction'
            data_size: 'small' (<1K), 'medium' (1K-100K), 'large' (>100K)
            interpretability_needed: boolean
        """
        key = (learning_type, output_type, data_size, interpretability_needed)
        
        if key in self.recommendations:
            return self.recommendations[key]
        return [("No exact match, try manual selection", "?", "")]
    
    def print_guide(self):
        """Print detailed selection guide"""
        guide = """
ALGORITHM SELECTION GUIDE
==========================

STEP 1: Problem Type
├─ Do you have labels?
│  ├─ YES → Supervised Learning
│  │   ├─ Output is number? → REGRESSION
│  │   └─ Output is category? → CLASSIFICATION
│  │
│  └─ NO → Unsupervised Learning
│      ├─ Want to group items? → CLUSTERING
│      └─ Want to reduce features? → DIMENSIONALITY REDUCTION

STEP 2: Data Size
├─ Small (<1K samples): Any model, focus on interpretability
├─ Medium (1K-100K): Tree-based, Neural Networks
└─ Large (>100K): Scalable models (LightGBM, Linear, KNN)

STEP 3: Key Requirements
├─ Speed: Linear models, KNN
├─ Accuracy: XGBoost, Neural Networks
├─ Interpretability: Linear, Decision Trees
└─ Both Accuracy & Speed: Random Forest
"""
        print(guide)


# Usage
recommender = AlgorithmRecommender()
recommender.print_guide()

# Get recommendations
print("\n" + "="*60)
print("EXAMPLE RECOMMENDATIONS")
print("="*60)

examples = [
    ('supervised', 'continuous', 'small', True),
    ('supervised', 'categorical', 'medium', False),
    ('unsupervised', 'clustering', 'large', False),
]

for args in examples:
    print(f"\nProblem: {args}")
    for algo, rating, reason in recommender.recommend(*args):
        print(f"  → {algo:<25} {rating}  {reason}")
````

---

## ✅ **Summary**

I've created **3 production-ready files**:

1. **`algorithm_selection_guide.md`** - Complete guide with code examples for:
   - Linear Regression, Logistic Regression, Random Forest, XGBoost, SVM
   - K-Means, PCA, DBSCAN
   - ARIMA, Prophet, LSTM
   - CNN examples
   - Deployment & monitoring

2. **`implementation_checklist.py`** - Comprehensive checklist for:
   - Data preprocessing
   - Model training
   - Evaluation
   - Production deployment

3. **`algorithm_recommender.py`** - Interactive recommendation system

---

Would you like me to create additional files for:
- ✅ **Hyperparameter Tuning Strategies**
- ✅ **Model Evaluation Metrics Deep Dive**
- ✅ **Feature Engineering Cookbook**
- ✅ **Error Analysis & Debugging Guide**

Let me know! 🚀
# 🎯 ADVANCED ML PRODUCTION GUIDE - PART 5

Creating comprehensive guides for hyperparameter tuning, metrics, feature engineering, and debugging.

---

## 📊 CREATE: Hyperparameter Tuning Strategies

````python
"""
Complete Hyperparameter Tuning Guide
Strategies, code examples, and best practices for production
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYPERPARAMETER TUNING: COMPLETE PRODUCTION GUIDE")
print("=" * 80)

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================================
# 1. GRID SEARCH - EXHAUSTIVE PARAMETER SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("1️⃣  GRID SEARCH - Exhaustive Search")
print("=" * 80)

print("""
Grid Search: Try ALL combinations of parameters

Pros:
  ✓ Guaranteed to find best in search space
  ✓ Parallelizable (n_jobs=-1)
  ✓ Good for small parameter spaces

Cons:
  ✗ Slow for large parameter spaces
  ✗ Exponential growth: 3 params × 5 values = 125 combinations

When to use:
  • Small parameter space (<100 combinations)
  • Have computational resources
  • Need guaranteed best result
  • Production model tuning

Example: Random Forest with Grid Search
""")

param_grid_rf = {
    'n_estimators': [50, 100, 200],           # 3 values
    'max_depth': [5, 10, 15, None],           # 4 values
    'min_samples_split': [2, 5, 10],          # 3 values
    'min_samples_leaf': [1, 2, 4],            # 3 values
    'max_features': ['sqrt', 'log2']          # 2 values
    # Total combinations: 3 × 4 × 3 × 3 × 2 = 216
}

print(f"Total combinations to test: {np.prod([len(v) for v in param_grid_rf.values()])}")

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,                    # 5-fold cross-validation
    scoring='roc_auc',       # Metric to optimize
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)

print("\n⏳ Running Grid Search (this may take a minute)...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Grid Search Complete!")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Test set performance
y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)
test_score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"Test set ROC-AUC: {test_score:.4f}")

# Visualize results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

print("\nTop 10 Parameter Combinations:")
print(results_df[['param_n_estimators', 'param_max_depth', 'mean_test_score']].head(10))

# ============================================================================
# 2. RANDOMIZED SEARCH - RANDOM SAMPLING
# ============================================================================

print("\n" + "=" * 80)
print("2️⃣  RANDOMIZED SEARCH - Random Sampling")
print("=" * 80)

print("""
Randomized Search: Sample random combinations

Pros:
  ✓ Fast for large parameter spaces
  ✓ Can explore more values
  ✓ Less computational cost
  ✓ Often finds near-optimal solutions

Cons:
  ✗ Not guaranteed to find global optimum
  ✗ Results vary (random sampling)

When to use:
  • Large parameter space (>200 combinations)
  • Quick exploration needed
  • Limited computational resources
  • Initial model tuning

Example: Gradient Boosting with Randomized Search
""")

param_dist_gb = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None]
    # Total possible: 5 × 5 × 6 × 4 × 4 × 3 × 3 = 36,000 combinations!
}

print(f"\nTotal possible combinations: {np.prod([len(v) for v in param_dist_gb.values()])}")
print("We'll sample 50 random combinations...")

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist_gb,
    n_iter=50,               # Sample 50 combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n⏳ Running Randomized Search...")
random_search.fit(X_train, y_train)

print(f"\n✅ Randomized Search Complete!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Test set performance
y_pred_rand = random_search.predict(X_test)
y_pred_proba_rand = random_search.predict_proba(X_test)
test_score_rand = roc_auc_score(y_test, y_pred_proba_rand[:, 1])
print(f"Test set ROC-AUC: {test_score_rand:.4f}")

# ============================================================================
# 3. BAYESIAN OPTIMIZATION - SMART SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("3️⃣  BAYESIAN OPTIMIZATION - Intelligent Search")
print("=" * 80)

print("""
Bayesian Optimization: Use probabilistic model to guide search

How it works:
  1. Start with random samples
  2. Build surrogate model (Gaussian Process)
  3. Use model to predict promising regions
  4. Iteratively sample from promising regions
  5. Update model

Pros:
  ✓ Very efficient (fewer evaluations needed)
  ✓ Learns from previous trials
  ✓ Best for expensive evaluations
  ✓ Handles continuous parameters well

Cons:
  ✗ More complex to implement
  ✗ Harder to interpret
  ✗ Slower per iteration

When to use:
  • Limited budget (few evaluations)
  • Training is very expensive
  • Continuous hyperparameters
  • Need optimal solution quickly

Library: scikit-optimize (skopt)
""")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    
    # Define parameter space
    space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.001, 0.2, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
    ]
    
    @use_named_args(space)
    def objective(n_estimators, learning_rate, max_depth, min_samples_split):
        """Function to minimize (we'll minimize -score to maximize score)"""
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
        return -score  # Minimize negative score = maximize score
    
    print("\n⏳ Running Bayesian Optimization (20 iterations)...")
    
    result = gp_minimize(
        objective,
        space,
        n_calls=20,           # Number of evaluations
        n_initial_points=5,   # Random points before optimization
        random_state=42,
        verbose=1
    )
    
    print(f"\n✅ Bayesian Optimization Complete!")
    print(f"Best score: {-result.fun:.4f}")
    print(f"Best parameters:")
    for param_name, param_val in zip(['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split'], 
                                      result.x):
        print(f"  {param_name}: {param_val}")
    
except ImportError:
    print("⚠️  scikit-optimize not installed. Install with: pip install scikit-optimize")

# ============================================================================
# 4. HYPERBAND - PROGRESSIVE RESOURCE ALLOCATION
# ============================================================================

print("\n" + "=" * 80)
print("4️⃣  HYPERBAND - Progressive Resource Allocation")
print("=" * 80)

print("""
Hyperband: Efficient bandit-based algorithm

How it works:
  1. Start with many candidates, limited resources
  2. Evaluate all with few resources (e.g., 10% of data)
  3. Keep top performers, double resources
  4. Repeat until one candidate remains

Pros:
  ✓ Very efficient (exponential elimination)
  ✓ Handles many hyperparameters
  ✓ Fast in practice
  ✓ Theoretically sound

Cons:
  ✗ Complex to implement from scratch
  ✗ Library support limited (use Ray Tune or Optuna)

When to use:
  • Many hyperparameters to tune
  • Limited computational budget
  • Want efficient, modern approach
  • Large-scale experiments

Libraries: Ray Tune, Optuna
""")

print("Example: Using Optuna for Hyperband-like tuning\n")

try:
    import optuna
    from optuna.samplers import TPESampler
    
    def objective_optuna(trial):
        """Optuna objective function"""
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.2)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
        return score
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    print("⏳ Running Optuna optimization (20 trials)...")
    study.optimize(objective_optuna, n_trials=20, show_progress_bar=True)
    
    print(f"\n✅ Optuna Optimization Complete!")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
except ImportError:
    print("⚠️  Optuna not installed. Install with: pip install optuna")

# ============================================================================
# 5. MANUAL TUNING WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("5️⃣  MANUAL TUNING WORKFLOW - Systematic Approach")
print("=" * 80)

print("""
Manual Tuning Strategy:
  1. Start with defaults (sklearn defaults are often good)
  2. Tune critical parameters first (usually: learning_rate, tree_depth)
  3. Use coarse grid, then fine-tune promising regions
  4. Monitor for overfitting (training vs validation curves)
  5. Ensemble final models if needed

Example: XGBoost Manual Tuning
""")

# Step 1: Start with reasonable defaults
print("\n📍 Step 1: Baseline with defaults")
xgb_base = XGBClassifier(random_state=42)
xgb_base.fit(X_train, y_train)
base_score = xgb_base.score(X_test, y_test)
print(f"Default accuracy: {base_score:.4f}")

# Step 2: Tune tree_depth and learning_rate (most important)
print("\n📍 Step 2: Coarse tuning of tree_depth and learning_rate")
param_grid_xgb_step2 = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.15]
}

grid_step2 = GridSearchCV(
    XGBClassifier(n_estimators=100, random_state=42),
    param_grid_xgb_step2,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_step2.fit(X_train, y_train)
print(f"Best params: {grid_step2.best_params_}")
print(f"Best CV score: {grid_step2.best_score_:.4f}")

# Step 3: Fine-tune around best values
print("\n📍 Step 3: Fine-tuning around best values")
best_depth = grid_step2.best_params_['max_depth']
best_lr = grid_step2.best_params_['learning_rate']

param_grid_xgb_step3 = {
    'max_depth': [best_depth - 1, best_depth, best_depth + 1],
    'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

grid_step3 = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid_xgb_step3,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_step3.fit(X_train, y_train)
print(f"Final best params: {grid_step3.best_params_}")
print(f"Final CV score: {grid_step3.best_score_:.4f}")

# Step 4: Check for overfitting
print("\n📍 Step 4: Check learning curves")

def plot_learning_curves(model, X_train, y_train, title="Learning Curves"):
    """Plot training vs validation curves"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, label='Validation', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

final_model = grid_step3.best_estimator_
plot_learning_curves(final_model, X_train, y_train)
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
print("✅ Saved learning_curves.png")

# ============================================================================
# 6. COMPARISON: WHICH METHOD IS BEST?
# ============================================================================

print("\n" + "=" * 80)
print("6️⃣  TUNING METHODS COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Method': ['Grid Search', 'Randomized Search', 'Bayesian Opt', 'Hyperband', 'Manual Tuning'],
    'Speed': ['Slow', 'Medium', 'Fast', 'Very Fast', 'Depends'],
    'Accuracy': ['Guaranteed Best', 'Near-optimal', 'Near-optimal', 'Good', 'Variable'],
    'Use Cases': [
        'Small space, need best',
        'Large space, medium budget',
        'Expensive evals, continuous',
        'Many params, limited budget',
        'Understanding, interpretability'
    ],
    'Complexity': ['Low', 'Low', 'High', 'High', 'Medium'],
    'Best For': [
        'Production, <200 combos',
        'Exploration, >200 combos',
        'High compute cost/eval',
        'Modern, scalable systems',
        'Small models, learning'
    ]
})

print(comparison.to_string(index=False))

# ============================================================================
# 7. PARAMETER TUNING GUIDES FOR COMMON ALGORITHMS
# ============================================================================

print("\n" + "=" * 80)
print("7️⃣  PARAMETER TUNING GUIDES")
print("=" * 80)

tuning_guides = {
    'Random Forest': {
        'Critical Parameters': {
            'n_estimators': 'More is usually better (100-1000), but with diminishing returns',
            'max_depth': 'Control overfitting (3-30), deeper = more overfitting',
        },
        'Secondary Parameters': {
            'min_samples_split': 'Min samples to split node (2-20), higher = simpler trees',
            'min_samples_leaf': 'Min samples in leaf (1-10), higher = smoother',
            'max_features': 'sqrt or log2 for classification, None for regression',
        },
        'Tuning Strategy': [
            '1. Start with n_estimators=100',
            '2. Find optimal max_depth (try 5-30)',
            '3. Fine-tune min_samples_split and min_samples_leaf',
            '4. Increase n_estimators if validation improves'
        ]
    },
    
    'XGBoost': {
        'Critical Parameters': {
            'learning_rate': 'Lower = slower but better (0.001-0.3)',
            'max_depth': 'Tree depth, deeper = more overfitting (3-10)',
            'n_estimators': 'Number of trees (50-1000)',
        },
        'Secondary Parameters': {
            'subsample': 'Row sampling (0.5-1.0), lower = more regularization',
            'colsample_bytree': 'Column sampling (0.5-1.0)',
            'min_child_weight': 'Min child weight (0-5), higher = simpler',
        },
        'Tuning Strategy': [
            '1. Fix learning_rate=0.1 initially',
            '2. Find best max_depth (tree_depth tuning)',
            '3. Tune subsample and colsample',
            '4. Reduce learning_rate and increase n_estimators',
            '5. Fine-tune min_child_weight'
        ]
    },
    
    'SVM': {
        'Critical Parameters': {
            'C': 'Regularization strength (0.1-1000), higher = less regularization',
            'gamma': 'Kernel coefficient (0.0001-10), smaller = larger influence',
            'kernel': 'rbf or poly for non-linear',
        },
        'Tuning Strategy': [
            '1. Start with kernel=rbf, C=1.0, gamma=auto',
            '2. Use GridSearch on log scale: C=[0.1, 1, 10, 100]',
            '3. Fine-tune gamma: [0.0001, 0.001, 0.01, 0.1, 1]',
            '4. Consider kernel=poly if rbf underperforms'
        ]
    },
    
    'Logistic Regression': {
        'Critical Parameters': {
            'C': 'Inverse regularization (0.001-1000), smaller = more regularization',
            'solver': 'lbfgs, liblinear, or saga',
        },
        'Tuning Strategy': [
            '1. Try different C values on log scale',
            '2. Usually C=1.0 or C=0.1 work well',
            '3. LR has few hyperparameters - usually not much tuning needed'
        ]
    },
    
    'Neural Networks': {
        'Critical Parameters': {
            'learning_rate': '0.001-0.1, lower is slower but more stable',
            'batch_size': '16-128, smaller = more updates',
            'n_layers': 'Usually 2-4 hidden layers',
            'layer_size': 'Usually 64-512 neurons per layer',
        },
        'Secondary Parameters': {
            'dropout': '0.2-0.5 to prevent overfitting',
            'optimizer': 'adam, sgd, or rmsprop',
        },
        'Tuning Strategy': [
            '1. Start with: learning_rate=0.001, batch_size=32, 2 hidden layers',
            '2. Increase layer size until validation stops improving',
            '3. Add dropout if overfitting detected',
            '4. Adjust learning_rate based on loss curves',
            '5. Try different optimizers'
        ]
    }
}

for algo, guide in tuning_guides.items():
    print(f"\n{'='*80}")
    print(f"🔧 {algo.upper()}")
    print('='*80)
    
    print(f"\n📌 Critical Parameters:")
    for param, desc in guide['Critical Parameters'].items():
        print(f"  • {param}: {desc}")
    
    if 'Secondary Parameters' in guide:
        print(f"\n📌 Secondary Parameters:")
        for param, desc in guide['Secondary Parameters'].items():
            print(f"  • {param}: {desc}")
    
    print(f"\n📌 Tuning Strategy:")
    for step in guide['Tuning Strategy']:
        print(f"  {step}")

# ============================================================================
# 8. BEST PRACTICES
# ============================================================================

print("\n" + "=" * 80)
print("8️⃣  HYPERPARAMETER TUNING BEST PRACTICES")
print("=" * 80)

best_practices = """
✅ DO:
   1. Use cross-validation (5-fold minimum)
   2. Scale/normalize features before tuning
   3. Start with critical parameters first
   4. Use coarse grid, then fine-tune promising regions
   5. Monitor both training and validation scores
   6. Save best model and best parameters
   7. Test on hold-out test set (not CV set)
   8. Log all experiments and results
   9. Use appropriate scoring metric for your problem
   10. Parallelize (n_jobs=-1) to speed up search

❌ DON'T:
   1. Tune on test set (data leakage!)
   2. Search entire parameter space exhaustively for large spaces
   3. Use only accuracy for imbalanced classification
   4. Forget to scale features before distance-based algorithms
   5. Tune too many parameters simultaneously
   6. Use small CV folds (use at least 5)
   7. Stop after first improvement (might be local optimum)
   8. Ignore computational cost (time ≠ accuracy)
   9. Tune without baseline (don't know if improvement is real)
   10. Overfit to validation set (use separate test set)

🎯 WORKFLOW:
   1. Create baseline with default parameters
   2. Identify most critical parameters (usually 2-3)
   3. Use coarse grid search on critical parameters
   4. Fine-tune around best values
   5. Check learning curves (overfitting?)
   6. Possibly ensemble multiple models
   7. Final evaluation on test set
   8. Document all decisions and results

⏱️  TIME ALLOCATION:
   • Quick Exploration: Randomized Search (50-100 iterations)
   • Production Model: Grid Search (100-500 iterations)
   • Research/Competition: Bayesian Opt (100-1000 iterations)
   • Time Budget: ~1-2 hours manual tuning, ~4-8 hours auto search
"""

print(best_practices)

print("\n" + "=" * 80)
print("✅ HYPERPARAMETER TUNING GUIDE COMPLETE")
print("=" * 80)
````

---

## 📈 CREATE: Model Evaluation Metrics Deep Dive

````python
"""
Complete Guide to Model Evaluation Metrics
Detailed explanations with code examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, matthews_corrcoef
)
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import seaborn as sns

print("=" * 100)
print("MODEL EVALUATION METRICS: COMPLETE GUIDE")
print("=" * 100)

# ============================================================================
# PART 1: CLASSIFICATION METRICS
# ============================================================================

print("\n" + "=" * 100)
print("PART 1: CLASSIFICATION METRICS")
print("=" * 100)

# Load example data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "-" * 100)
print("1️⃣  ACCURACY - Simple Correctness")
print("-" * 100)

accuracy = accuracy_score(y_test, y_pred)

print(f"""
Accuracy = (Correct Predictions) / (Total Predictions)
         = (TP + TN) / (TP + TN + FP + FN)
         = {accuracy:.4f}

📊 What it means: {accuracy*100:.1f}% of predictions are correct

✅ WHEN TO USE:
   • Balanced datasets (similar number of each class)
   • All classes equally important
   • Quick overall performance check

❌ WHEN NOT TO USE:
   • Imbalanced datasets (e.g., 99% class 0, 1% class 1)
   • When false positives/negatives have different costs
   • Example: Disease detection (wrong positive ≠ wrong negative)

⚠️  Example: Accuracy can be misleading!
   • Dataset: 99 healthy, 1 sick person
   • Naive model: Always predict "healthy"
   • Accuracy = 99% (seems great!)
   • But: Misses all disease cases (0% recall)
""")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(y_test, y_pred)

print("\n" + "-" * 100)
print("2️⃣  CONFUSION MATRIX - Detailed Breakdown")
print("-" * 100)

print(f"""
Confusion Matrix:
        Predicted Negative    Predicted Positive
Actual Negative   TN={cm[0,0]:<6}            FP={cm[0,1]:<6}
Actual Positive   FN={cm[1,0]:<6}            TP={cm[1,1]:<6}

Where:
  • TP (True Positive): Model said YES, actually YES ✓
  • TN (True Negative): Model said NO, actually NO ✓
  • FP (False Positive): Model said YES, actually NO ✗ (Type I error)
  • FN (False Negative): Model said NO, actually YES ✗ (Type II error)

Interpretation:
  • TP + TN = Correct predictions
  • FP + FN = Wrong predictions
  • FP = False alarms
  • FN = Missed cases
""")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved confusion_matrix.png")

# ============================================================================
# PRECISION & RECALL
# ============================================================================

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print("\n" + "-" * 100)
print("3️⃣  PRECISION - Positive Prediction Accuracy")
print("-" * 100)

print(f"""
Precision = TP / (TP + FP)
          = {precision:.4f}

📊 What it means:
   Of all POSITIVE predictions, {precision*100:.1f}% were correct
   When model says "YES", it's right {precision*100:.1f}% of the time

✅ USE FOR:
   • Minimizing false alarms
   • Spam detection: Want few false positives
   • Fraud detection: False alarms are costly
   • Email filtering: False positives are annoying

Example: 
   • Model predicts: 100 emails are spam
   • Actually spam: {int(precision*100)} of them
   • False positives: {int((1-precision)*100)} legitimate emails blocked
""")

print("\n" + "-" * 100)
print("4️⃣  RECALL (Sensitivity) - Positive Detection Rate")
print("-" * 100)

print(f"""
Recall = TP / (TP + FN)
       = {recall:.4f}

📊 What it means:
   Of all ACTUAL positives, {recall*100:.1f}% were found
   Catch {recall*100:.1f}% of true cases

✅ USE FOR:
   • Minimizing missed cases
   • Disease detection: Can't miss sick patients
   • Fraud detection: Can't miss actual fraud
   • Security: Want to catch all threats

Example:
   • 100 actual disease cases
   • Model catches: {int(recall*100)} of them
   • Missed cases: {int((1-recall)*100)}
""")

print("\n" + "-" * 100)
print("5️⃣  SPECIFICITY - Negative Detection Rate")
print("-" * 100)

print(f"""
Specificity = TN / (TN + FP)
            = {specificity:.4f}

📊 What it means:
   Of all ACTUAL negatives, {specificity*100:.1f}% correctly identified
   Correctly identify {specificity*100:.1f}% of negative cases

👥 Real-world example:
   • 1000 healthy people
   • Model correctly identifies: {int(specificity*1000)}
   • False alarms: {int((1-specificity)*1000)}
""")

print("\n" + "-" * 100)
print("PRECISION-RECALL TRADEOFF")
print("-" * 100)

print(f"""
When you adjust decision threshold:

High Threshold (Require high confidence for "YES"):
  ✓ High Precision (fewer false alarms)
  ✗ Low Recall (misses some cases)
  Use: Spam detection, fraud detection

Low Threshold (Easy to predict "YES"):
  ✗ Low Precision (many false alarms)
  ✓ High Recall (catches most cases)
  Use: Disease screening, security

Perfect Precision vs Perfect Recall:
  • Perfect Precision: Only say YES when 100% sure → Misses cases
  • Perfect Recall: Say YES to be safe → Many false alarms
  • Trade-off: Choose based on use case

Current model:
  • Precision: {precision:.4f} (low false alarms)
  • Recall: {recall:.4f} (catch rate)
""")

# ============================================================================
# F1 SCORE
# ============================================================================

f1 = f1_score(y_test, y_pred)

print("\n" + "-" * 100)
print("6️⃣  F1 SCORE - Harmonic Mean of Precision & Recall")
print("-" * 100)

print(f"""
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = {f1:.4f}

📊 What it means:
   Balanced score between precision and recall
   Penalizes models that are good at only one
   Range: 0 (worst) to 1 (perfect)

✅ USE FOR:
   • Imbalanced datasets
   • When false positives AND negatives matter equally
   • Need single score to optimize

Comparison:
  • Accuracy: Doesn't account for class imbalance
  • Precision: Ignores false negatives
  • Recall: Ignores false positives
  • F1: Balances both

Current model:
  • Precision: {precision:.4f}
  • Recall: {recall:.4f}
  • F1: {f1:.4f} (harmonic mean)

Example:
  • Perfect Precision + Poor Recall = Low F1
  • Poor Precision + Perfect Recall = Low F1
  • Good balance = High F1
""")

# ============================================================================
# ROC-AUC
# ============================================================================

roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

print("\n" + "-" * 100)
print("7️⃣  ROC-AUC - Threshold-Independent Performance")
print("-" * 100)

print(f"""
ROC-AUC = Area Under ROC Curve
        = {roc_auc:.4f}

📊 What it means:
   Probability model ranks random positive higher than random negative
   Range: 0.5 (random guessing) to 1.0 (perfect)

ROC Curve plots:
  • True Positive Rate (Sensitivity): TP / (TP + FN)
  • False Positive Rate (1 - Specificity): FP / (FP + TN)
  • At different classification thresholds

✅ USE FOR:
   • Comparing models across thresholds
   • Imbalanced datasets
   • Probability predictions
   • When threshold is not fixed yet

Interpretation:
  • 0.5: Random model
  • 0.7: Fair model
  • 0.8: Good model
  • 0.9+: Excellent model

Current model: {roc_auc:.4f} = {'Good' if roc_auc > 0.8 else 'Fair' if roc_auc > 0.7 else 'Poor'} model

👉 Example: Medical Test
   • AUC = 0.9 means:
   • If you pick 1 sick & 1 healthy person
   • 90% chance model ranks sick person higher
""")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved roc_curve.png")

# ============================================================================
# PART 2: REGRESSION METRICS
# ============================================================================

print("\n" + "=" * 100)
print("PART 2: REGRESSION METRICS")
print("=" * 100)

# Load regression data
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train regression model
model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = model_reg.predict(X_test_reg)

# ============================================================================
# MAE, RMSE, MAPE
# ============================================================================

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mape = np.mean(np.abs((y_test_reg - y_pred_reg) / y_test_reg)) * 100
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n" + "-" * 100)
print("1️⃣  MAE - Mean Absolute Error")
print("-" * 100)

print(f"""
MAE = (1/n) × Σ|y_actual - y_predicted|
    = {mae:.2f}

📊 What it means:
   Average absolute difference between predictions and actual values
   {mae:.2f} units off on average

✅ USE FOR:
   • Easy interpretation (same units as target)
   • Robust to outliers (linear error)
   • When all errors equally important

Example:
   • Predicting house prices
   • MAE = $5,000
   • On average, predictions are off by $5,000
   • Easy to explain to non-technical people
""")

print("\n" + "-" * 100)
print("2️⃣  RMSE - Root Mean Squared Error")
print("-" * 100)

print(f"""
RMSE = √((1/n) × Σ(y_actual - y_predicted)²)
     = {rmse:.2f}

📊 What it means:
   Root of average squared errors
   Penalizes large errors more than MAE
   {rmse:.2f} units off (on average, penalizing outliers)

✅ USE FOR:
   • When large errors are very bad
   • Statistical/mathematical properties
   • Comparing against baselines

Comparison with MAE:
   • MAE = {mae:.2f} (balanced)
   • RMSE = {rmse:.2f} (emphasizes large errors)
   • RMSE > MAE always (due to squaring)

Real-world example:
   • Stock price: 1 error of $10, 99 errors of $1
   • MAE = ~$1.09
   • RMSE = ~$1.90 (penalizes the $10 error)
   • Which metric matters? Depends on use case!
""")

print("\n" + "-" * 100)
print("3️⃣  MAPE - Mean Absolute Percentage Error")
print("-" * 100)

print(f"""
MAPE = (1/n) × Σ(|y_actual - y_predicted| / |y_actual|) × 100
     = {mape:.2f}%

📊 What it means:
   Average percentage error
   Predictions are off by {mape:.2f}% on average

✅ USE FOR:
   • When scale varies across samples
   • Business metrics (interpret as percentage)
   • Comparing across different scales

Example:
   • Forecast revenue: $1M accuracy (MAPE 2%)
   • Forecast small shop: $10K accuracy (MAPE 20%)
   • Different MAPE, but both good predictions!
""")

print("\n" + "-" * 100)
print("4️⃣  R² Score - Coefficient of Determination")
print("-" * 100)

print(f"""
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_actual - y_pred)² / Σ(y_actual - mean)²)
   = {r2:.4f}

📊 What it means:
   Model explains {r2*100:.1f}% of variance in data
   Range: -∞ (very bad) to 1 (perfect)

✅ USE FOR:
   • Primary regression metric
   • Assessing model fit
   • Comparing regression models

Interpretation:
   • 1.0: Perfect predictions
   • 0.9: Excellent (90% variance explained)
   • 0.7: Good (70% variance explained)
   • 0.5: Fair (50% variance explained)
   • 0.0: Same as predicting mean
   • <0: Worse than predicting mean

Example:
   • House price model
   • R² = 0.85 means model explains 85% of price variation
   • 15% unexplained (other factors, randomness)
""")

# ============================================================================
# PART 3: IMBALANCED CLASSIFICATION METRICS
# ============================================================================

print("\n" + "=" * 100)
print("PART 3: IMBALANCED CLASSIFICATION METRICS")
print("=" * 100)

print("""
When classes are imbalanced (e.g., 95% healthy, 5% sick):
  • Accuracy is misleading
  • Need metrics that account for imbalance

Example Dataset: 100 people, 1 is sick, 99 are healthy

Naive Model (always predict healthy):
  • Accuracy = 99/100 = 99% ← Looks great!
  • But: Never detects the sick person
  • Precision = undefined (no positive predictions)
  • Recall = 0% (catches 0 of 1 sick person)

This is why we need specific metrics for imbalanced data!
""")

# Create imbalanced dataset
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, weights=[0.95, 0.05], random_state=42
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42
)

# Train model on imbalanced data
model_imb = RandomForestClassifier(n_estimators=100, random_state=42)
model_imb.fit(X_train_imb, y_train_imb)

y_pred_imb = model_imb.predict(X_test_imb)
y_pred_proba_imb = model_imb.predict_proba(X_test_imb)[:, 1]

# Calculate metrics
acc_imb = accuracy_score(y_test_imb, y_pred_imb)
prec_imb = precision_score(y_test_imb, y_pred_imb)
rec_imb = recall_score(y_test_imb, y_pred_imb)
f1_imb = f1_score(y_test_imb, y_pred_imb)
roc_auc_imb = roc_auc_score(y_test_imb, y_pred_proba_imb)

print("\n" + "-" * 100)
print("📊 IMBALANCED DATA METRICS")
print("-" * 100)

metrics_imb = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Score': [f'{acc_imb:.4f}', f'{prec_imb:.4f}', f'{rec_imb:.4f}', f'{f1_imb:.4f}', f'{roc_auc_imb:.4f}'],
    'Interpretation': [
        f'{acc_imb*100:.1f}% correct (misleading with imbalance!)',
        f'{prec_imb*100:.1f}% of positive predictions correct',
        f'{rec_imb*100:.1f}% of actual positives found',
        f'Balanced score ({f1_imb:.4f})',
        f'Threshold-independent ({roc_auc_imb:.4f})'
    ],
    'Best For Imbalance': [
        '❌ NO',
        '✅ YES (if FP costly)',
        '✅ YES (if FN costly)',
        '✅ YES (balanced)',
        '✅ YES (best choice)'
    ]
})

print(metrics_imb.to_string(index=False))

print(f"""
✅ RECOMMENDED APPROACH FOR IMBALANCED DATA:
   1. Use ROC-AUC as primary metric
   2. Use F1 Score as secondary metric
   3. Check Precision and Recall separately
   4. Report Confusion Matrix
   5. Avoid using Accuracy alone
""")

# ============================================================================
# PART 4: METRIC SELECTION GUIDE
# ============================================================================

print("\n" + "=" * 100)
print("PART 4: WHEN TO USE EACH METRIC")
print("=" * 100)

selection_guide = pd.DataFrame({
    'Problem': [
        'Balanced Classification',
        'Imbalanced Classification',
        'Binary with Thresholds',
        'Binary, Probabilities',
        'Multi-class',
        'Regression (Interpretable)',
        'Regression (Math)',
        'Regression (Business)',
        'Cost-sensitive (FP > FN)',
        'Cost-sensitive (FN > FP)',
    ],
    'Primary Metric': [
        'Accuracy',
        'F1 Score',
        'Precision or Recall',
        'ROC-AUC',
        'Macro F1 or Weighted F1',
        'MAE',
        'RMSE',
        'MAPE',
        'Precision',
        'Recall'
    ],
    'Secondary Metrics': [
        'F1, Precision, Recall',
        'ROC-AUC, Precision, Recall',
        'F1, ROC-AUC',
        'PR-AUC, F1',
        'Per-class metrics',
        'RMSE, MAPE, R²',
        'MAE, R²',
        'MAE, RMSE',
        'Recall, F1',
        'Precision, F1'
    ],
    'Example Domain': [
        'Image classification',
        'Disease detection, Fraud',
        'Email spam filter',
        'Credit approval',
        'Customer segmentation',
        'Stock price prediction',
        'Weather forecasting',
        'Sales forecasting',
        'Spam filtering',
        'Cancer screening'
    ]
})

print(selection_guide.to_string(index=False))

# ============================================================================
# PART 5: BEST PRACTICES
# ============================================================================

print("\n" + "=" * 100)
print("PART 5: METRIC BEST PRACTICES")
print("=" * 100)

best_practices_metrics = """
✅ DO:
   1. Choose metrics BEFORE training (avoid cherry-picking)
   2. Use cross-validation for metric estimates
   3. Report multiple metrics (not just one)
   4. Consider domain-specific costs (FP vs FN)
   5. Visualize predictions (confusion matrix, ROC curve)
   6. Compare against baseline model
   7. Monitor both training and test metrics
   8. Report confidence intervals around metrics
   9. Document why you chose each metric
   10. Track metrics over time

❌ DON'T:
   1. Use only accuracy for imbalanced data
   2. Report metrics on training set (use test/validation)
   3. Optimize metric on test set (data leakage)
   4. Ignore class imbalance
   5. Use inappropriate metrics (e.g., MAE for binary class)
   6. Report single metric without context
   7. Change metrics after seeing results
   8. Forget to compare with baseline
   9. Optimize micro-averaged metrics (often useless)
   10. Report rounded metrics (0.95 vs 0.9523)

🎯 WORKFLOW:
   1. Identify business problem
   2. Determine cost of different error types
   3. Select appropriate metrics
   4. Choose primary and secondary metrics
   5. Set success thresholds
   6. Train model
   7. Evaluate on validation set
   8. Report all metrics on test set
   9. Analyze error patterns
   10. Document decisions

⚠️  COMMON MISTAKES:
   • Only using accuracy for medical diagnosis
   • Only using precision for fraud detection
   • Forgetting to account for class imbalance
   • Reporting metrics on training set
   • Changing metrics based on results
   • Not comparing with baselines
   • Using inappropriate metrics for problem type
"""

print(best_practices_metrics)

print("\n" + "=" * 100)
print("✅ MODEL EVALUATION METRICS GUIDE COMPLETE")
print("=" * 100)
````

---

## 🔧 CREATE: Feature Engineering Cookbook

````python
"""
Feature Engineering Cookbook
Advanced techniques with production code examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("FEATURE ENGINEERING COOKBOOK - Advanced Techniques")
print("=" * 100)

# Load example data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")

# ============================================================================
# 1. HANDLING MISSING VALUES
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  HANDLING MISSING VALUES")
print("=" * 100)

# Create dataset with missing values
df = pd.DataFrame(X_train, columns=feature_names)
df.iloc[::50, 0] = np.nan  # Add some missing values
df.iloc[::100, 3] = np.nan

print(f"Missing values:\n{df.isnull().sum().sum()} total missing values")

print("""
STRATEGIES:

1. REMOVAL (Simple, lose data)
   • Remove rows with missing values
   • Remove columns with >30% missing
   Use: When few missing values (<5%)

2. MEAN/MEDIAN IMPUTATION (Fast, simple)
   • Replace with mean/median
   • Use median for outliers
   Use: Quick baseline, numerical data

3. FORWARD/BACKWARD FILL (Time series)
   • Use previous/next value
   Use: Time series with temporal order

4. KNN IMPUTATION (Uses similar samples)
   • Use k nearest neighbors values
   Use: When sample similarity matters

5. ITERATIVE IMPUTATION (Advanced)
   • Use other features to predict missing
   Use: Many missing values, complex relationships

6. DOMAIN-SPECIFIC (Best quality)
   • Use domain knowledge
   Use: When you understand the data

EXAMPLE CODE:
""")

# Strategy 1: Removal
df_removed = df.dropna()
print(f"\nRemoval: {df_removed.shape[0]} rows remaining (from {df.shape[0]})")

# Strategy 2: Mean imputation
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(
    imputer_mean.fit_transform(df),
    columns=df.columns
)
print(f"Mean imputation: {df_mean.isnull().sum().sum()} missing values")

# Strategy 3: KNN imputation
from sklearn.impute import KNNImputer
imputer_knn = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    imputer_knn.fit_transform(df),
    columns=df.columns
)
print(f"KNN imputation: {df_knn.isnull().sum().sum()} missing values")

# Strategy 4: Iterative imputation
from sklearn.impute import IterativeImputer
imputer_iter = IterativeImputer(max_iter=10, random_state=42)
df_iter = pd.DataFrame(
    imputer_iter.fit_transform(df),
    columns=df.columns
)
print(f"Iterative imputation: {df_iter.isnull().sum().sum()} missing values")

# ============================================================================
# 2. HANDLING OUTLIERS
# ============================================================================

print("\n" + "=" * 100)
print("2️⃣  HANDLING OUTLIERS")
print("=" * 100)

feature_idx = 0
feature_data = df[feature_names[feature_idx]]

# Calculate statistics
mean = feature_data.mean()
std = feature_data.std()
q1 = feature_data.quantile(0.25)
q3 = feature_data.quantile(0.75)
iqr = q3 - q1

print(f"""
Feature: {feature_names[feature_idx]}
Mean: {mean:.2f}, Std: {std:.2f}
Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}

OUTLIER DETECTION METHODS:

1. Z-SCORE METHOD
   • Outliers: |z-score| > 3
   • Assumes normal distribution
   Use: Normal data, identify extreme values
""")

# Z-score method
z_scores = np.abs(stats.zscore(feature_data.dropna()))
outlier_indices = np.where(z_scores > 3)[0]
print(f"Z-score outliers: {len(outlier_indices)} values with |z| > 3")

print("""
2. IQR METHOD (Tukey's Fences)
   • Outliers: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
   • Robust to distribution
   Use: Any distribution, most common
""")

# IQR method
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
print(f"IQR outliers: {outlier_mask.sum()} values outside bounds ({lower_bound:.2f}, {upper_bound:.2f})")

print("""
3. ISOLATION FOREST (Machine Learning)
   • Tree-based anomaly detection
   • Detects multivariate outliers
   Use: Complex patterns, high dimensions
""")

from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso_forest.fit_predict(df)
print(f"Isolation Forest outliers: {(outlier_pred == -1).sum()} anomalies")

print("""
HANDLING STRATEGIES:

1. REMOVAL (Lose data)
   Use: Truly invalid values, few outliers

2. CAPPING (Clip to bounds)
   Use: Natural limits (age: 0-120)

3. LOG TRANSFORMATION (Compress scale)
   Use: Right-skewed data

4. ROBUST SCALING (Less sensitive)
   Use: When keeping outliers matters
""")

# Visualize outlier detection
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Z-score
axes[0, 0].hist(feature_data, bins=30, alpha=0.7, label='Data')
axes[0, 0].axvline(mean + 3*std, color='r', linestyle='--', label='Z=±3')
axes[0, 0].axvline(mean - 3*std, color='r', linestyle='--')
axes[0, 0].set_title('Z-Score Method')
axes[0, 0].legend()

# IQR
axes[0, 1].hist(feature_data, bins=30, alpha=0.7)
axes[0, 1].axvline(lower_bound, color='r', linestyle='--', label='IQR bounds')
axes[0, 1].axvline(upper_bound, color='r', linestyle='--')
axes[0, 1].set_title('IQR Method')
axes[0, 1].legend()

# Box plot
axes[1, 0].boxplot(feature_data)
axes[1, 0].set_title('Box Plot')

# Scatter plot (before/after)
axes[1, 1].scatter(range(len(feature_data)), sorted(feature_data), alpha=0.5)
axes[1, 1].scatter(np.where(outlier_mask)[0], feature_data[outlier_mask], 
                   color='r', label='Outliers')
axes[1, 1].set_title('Outlier Detection')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=150, bbox_inches='tight')
print("✅ Saved outlier_detection.png")

# ============================================================================
# 3. SCALING & NORMALIZATION
# ============================================================================

print("\n" + "=" * 100)
print("3️⃣  SCALING & NORMALIZATION")
print("=" * 100)

print("""
WHY SCALE?
  • Distance-based algorithms: KNN, KMeans, SVM (sensitive to scale)
  • Gradient descent: Neural Networks, Logistic Regression (faster convergence)
  • Tree-based: RF, XGBoost (don't need scaling, but doesn't hurt)

TECHNIQUES:

1. STANDARDIZATION (Z-score normalization)
   • Formula: (x - mean) / std
   • Result: Mean=0, Std=1
   • Use: Most common, neural networks, assumes normal distribution
""")

# Standardization
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_train)
print(f"Standardized - Mean: {X_standard.mean():.4f}, Std: {X_standard.std():.4f}")

print("""
2. MIN-MAX SCALING (Normalization)
   • Formula: (x - min) / (max - min)
   • Result: Range [0, 1]
   • Use: Neural networks, images, bounded features
""")

# Min-Max scaling
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_train)
print(f"Min-Max scaled - Min: {X_minmax.min():.4f}, Max: {X_minmax.max():.4f}")

print("""
3. ROBUST SCALING
   • Formula: (x - median) / IQR
   • Result: Robust to outliers
   • Use: When data has outliers
""")

# Robust scaling
from sklearn.preprocessing import RobustScaler
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_train)
print(f"Robust scaled - Median around 0, IQR-based")

print("""
4. LOG TRANSFORMATION
   • Formula: log(x) or log(x + 1)
   • Result: Compress right-skewed data
   • Use: Right-skewed distributions (income, prices)
""")

# ============================================================================
# 4. POLYNOMIAL FEATURES
# ============================================================================

print("\n" + "=" * 100)
print("4️⃣  POLYNOMIAL & INTERACTION FEATURES")
print("=" * 100)

print("""
POLYNOMIAL FEATURES:
  • Create higher-order terms
  • Example: [x1, x2] → [x1, x2, x1², x2², x1×x2]
  • Use: Capture non-linear relationships

CAUTION:
  • Dimensionality explodes: n features → n^d features
  • Overfitting risk
  • Example: 10 features with degree 3 → 286 features!
""")

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train[:, :3])  # Use only 3 features for demo

print(f"Original: {X_train[:, :3].shape}")
print(f"With polynomial (degree 2): {X_poly.shape}")
print(f"Features: {poly.get_feature_names_out(feature_names[:3])[:10]}...")

print("""
INTERACTION FEATURES:
  • Combine features that interact
  • Example: height × weight for BMI-like feature
  • More interpretable than full polynomial
  Use: When you know features interact
""")

# Manual interaction features
df_inter = df[[feature_names[0], feature_names[1]]].copy()
df_inter['interaction'] = df_inter.iloc[:, 0] * df_inter.iloc[:, 1]
print(f"Original 2 features + interaction: {df_inter.shape[1]} features")

# ============================================================================
# 5. FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 100)
print("5️⃣  FEATURE SELECTION - Reduce Dimensionality")
print("=" * 100)

print("""
WHY SELECT FEATURES?
  • Improve model performance (remove noise)
  • Speed up training
  • Reduce overfitting
  • Improve interpretability
  • Reduce storage/computation

METHODS:

1. UNIVARIATE SELECTION
   • Select k best features individually
   • Fast, simple
   Use: Baseline, high-dimensional data
""")

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

print(f"Top 5 features by univariate F-score:")
for feature in selected_features:
    print(f"  • {feature}")

print("""
2. TREE-BASED FEATURE IMPORTANCE
   • Use feature importance from trees
   • Captures interactions
   Use: Tree-based models, interpretable
""")

# Tree-based importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 features by Random Forest importance:")
print(feature_importance.head())

# Visualize importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ Saved feature_importance.png")

print("""
3. CORRELATION-BASED SELECTION
   • Remove highly correlated features
   • Reduces multicollinearity
   Use: When features correlate with each other
""")

# Calculate correlation
corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (r > 0.95)")
if high_corr_pairs:
    print("Sample:")
    for f1, f2, corr in high_corr_pairs[:3]:
        print(f"  {f1} ↔ {f2}: {corr:.4f}")

# ============================================================================
# 6. DIMENSIONALITY REDUCTION
# ============================================================================

print("\n" + "=" * 100)
print("6️⃣  DIMENSIONALITY REDUCTION - PCA & UMAP")
print("=" * 100)

print("""
PCA (Principal Component Analysis):
  • Linear dimensionality reduction
  • Creates uncorrelated components
  • Preserves maximum variance
  Use: Baseline reduction, visualization, when interpretability matters
""")

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_train)

print(f"PCA reduced from {X_train.shape[1]} to {X_pca.shape[1]} features")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize PCA variance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Variance Explained')
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(10), pca.explained_variance_ratio_[:10])
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Variance Ratio')
axes[1].set_title('Individual Component Variance')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved pca_analysis.png")

print("""
UMAP (Uniform Manifold Approximation):
  • Non-linear dimensionality reduction
  • Better preserves structure
  Use: Visualization, complex manifolds
  Library: pip install umap-learn
""")

try:
    from umap import UMAP
    umap_reducer = UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_train)
    print(f"UMAP reduced to {X_umap.shape[1]} dimensions")
    print("✅ UMAP available")
except ImportError:
    print("⚠️  UMAP not installed (pip install umap-learn)")

# ============================================================================
# 7. CATEGORICAL ENCODING
# ============================================================================

print("\n" + "=" * 100)
print("7️⃣  CATEGORICAL ENCODING - Handle Categorical Features")
print("=" * 100)

# Create example categorical data
df_cat = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'] * 20,
    'size': ['small', 'medium', 'large', 'small', 'large'] * 20
})

print("""
ENCODING METHODS:

1. LABEL ENCODING
   • Convert to integers: red=0, blue=1, green=2
   • Use: Tree-based models, ordinal features
""")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cat['color_encoded'] = le.fit_transform(df_cat['color'])
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

print("""
2. ONE-HOT ENCODING
   • Create binary columns for each category
   • Example: color_red=1, color_blue=0, color_green=0
   • Use: Linear models, neural networks
""")

df_onehot = pd.get_dummies(df_cat[['color', 'size']], drop_first=False)
print(f"One-hot encoded shape: {df_onehot.shape}")
print(f"Columns: {list(df_onehot.columns)}")

print("""
3. ORDINAL ENCODING
   • Map categories to ordered integers
   • Example: small=1, medium=2, large=3
   • Use: When categories have natural order
""")

ordinal_map = {'small': 1, 'medium': 2, 'large': 3}
df_cat['size_ordinal'] = df_cat['size'].map(ordinal_map)
print("Ordinal mapping:", ordinal_map)

# ============================================================================
# 8. FEATURE ENGINEERING BEST PRACTICES
# ============================================================================

print("\n" + "=" * 100)
print("8️⃣  BEST PRACTICES & WORKFLOW")
print("=" * 100)

best_practices = """
✅ DO:
   1. Explore data first (distributions, correlations, missing values)
   2. Fit transformers ONLY on training data
   3. Apply same transformation to test/validation data
   4. Document feature engineering decisions
   5. Create interpretable features (domain knowledge)
   6. Validate feature importance (not just model accuracy)
   7. Remove redundant/correlated features
   8. Handle outliers appropriately for your problem
   9. Use proper scaling for distance-based algorithms
   10. Monitor feature distributions over time (production)

❌ DON'T:
   1. Use test data to fit transformers (data leakage!)
   2. Engineer features without understanding data
   3. Keep irrelevant features (worse generalization)
   4. Ignore missing values (implicit bias)
   5. Over-engineer (simpler is often better)
   6. Skip baseline (validate improvements)
   7. Add too many polynomial features (curse of dimensionality)
   8. Forget categorical encoding
   9. Use mean imputation for MCAR data
   10. Engineer features on full dataset before splitting

WORKFLOW:
   1. EDA: Understand data distributions, relationships
   2. Data Cleaning: Handle missing values, duplicates, errors
   3. Outlier Treatment: Decide removal, transformation, or keeping
   4. Scaling: Apply appropriate scaling
   5. Feature Creation: Domain-driven new features
   6. Feature Selection: Remove irrelevant features
   7. Encoding: Convert categorical to numerical
   8. Validation: Check feature quality and importance
   9. Deployment: Monitor feature distributions, retrain on new data

PIPELINE EXAMPLE:
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   from sklearn.ensemble import RandomForestClassifier
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('pca', PCA(n_components=0.95)),
       ('classifier', RandomForestClassifier())
   ])
   
   pipeline.fit(X_train, y_train)
   pipeline.predict(X_test)
"""

print(best_practices)

print("\n" + "=" * 100)
print("✅ FEATURE ENGINEERING COOKBOOK COMPLETE")
print("=" * 100)
````

---

## 🐛 CREATE: Error Analysis & Debugging Guide

````python
"""
Comprehensive Error Analysis & Debugging Guide
Diagnose and fix ML model problems systematically
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("ML DEBUGGING GUIDE - Diagnose and Fix Model Problems")
print("=" * 100)

# ============================================================================
# 1. DIAGNOSING THE PROBLEM
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  DIAGNOSING PERFORMANCE ISSUES")
print("=" * 100)

print("""
DECISION TREE - What's wrong with your model?

START: Model performance is poor
  │
  ├─ Training accuracy HIGH, test accuracy LOW
  │  └─→ OVERFITTING (memorized training data)
  │      └─ Diagnose: High train score, low val score
  │      └─ Fix: Reduce model complexity, regularization, more data
  │
  ├─ Training accuracy LOW, test accuracy LOW
  │  └─→ UNDERFITTING (too simple for problem)
  │      └─ Diagnose: Low scores on both
  │      └─ Fix: Increase model complexity, remove regularization
  │
  ├─ Good performance on train/test, bad in production
  │  └─→ DATA DRIFT / DISTRIBUTION CHANGE
  │      └─ Diagnose: Different data in production
  │      └─ Fix: Retrain on new data, monitor distributions
  │
  └─ Random performance, varies wildly between runs
     └─→ HIGH VARIANCE (unstable model)
         └─ Diagnose: Large std in cross-validation
         └─ Fix: More data, ensemble methods, cross-validation

""")

# Create datasets with different problems
np.random.seed(42)

# Problem 1: Overfitting
print("-" * 100)
print("EXAMPLE 1: OVERFITTING")
print("-" * 100)

X_small = np.random.randn(50, 10)
y_small = (X_small[:, 0] + X_small[:, 1] > 0).astype(int)

X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
    X_small, y_small, test_size=0.3, random_state=42
)

# Overfit model (deep tree)
overfit_model = RandomForestClassifier(n_estimators=1, max_depth=20, random_state=42)
overfit_model.fit(X_train_small, y_train_small)

train_score_overfit = overfit_model.score(X_train_small, y_train_small)
test_score_overfit = overfit_model.score(X_test_small, y_test_small)

print(f"Training accuracy: {train_score_overfit:.4f}")
print(f"Test accuracy: {test_score_overfit:.4f}")
print(f"Overfitting gap: {train_score_overfit - test_score_overfit:.4f}")

if train_score_overfit - test_score_overfit > 0.15:
    print("✋ DIAGNOSIS: OVERFITTING DETECTED!")
    print("   Solutions:")
    print("   1. Reduce model complexity (max_depth, n_estimators)")
    print("   2. Add regularization")
    print("   3. Collect more training data")
    print("   4. Use ensemble methods")

# Proper model
proper_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
proper_model.fit(X_train_small, y_train_small)

train_score_proper = proper_model.score(X_train_small, y_train_small)
test_score_proper = proper_model.score(X_test_small, y_test_small)

print(f"\nWith proper hyperparameters:")
print(f"Training accuracy: {train_score_proper:.4f}")
print(f"Test accuracy: {test_score_proper:.4f}")
print(f"Overfitting gap: {train_score_proper - test_score_proper:.4f}")

# Problem 2: Underfitting
print("\n" + "-" * 100)
print("EXAMPLE 2: UNDERFITTING")
print("-" * 100)

# Underfit model (too simple)
underfit_model = LogisticRegression(max_iter=100, random_state=42)
underfit_model.fit(X_train_small, y_train_small)

train_score_underfit = underfit_model.score(X_train_small, y_train_small)
test_score_underfit = underfit_model.score(X_test_small, y_test_small)

print(f"Training accuracy: {train_score_underfit:.4f}")
print(f"Test accuracy: {test_score_underfit:.4f}")

if train_score_underfit < 0.7:
    print("✋ DIAGNOSIS: UNDERFITTING DETECTED!")
    print("   Solutions:")
    print("   1. Increase model complexity")
    print("   2. Feature engineering")
    print("   3. Reduce regularization")
    print("   4. Better hyperparameters")

# ============================================================================
# 2. LEARNING CURVES - VISUAL DIAGNOSIS
# ============================================================================

print("\n" + "=" * 100)
print("2️⃣  LEARNING CURVES - Visual Diagnosis Tool")
print("=" * 100)

def plot_learning_curves_detailed(model, X_train, y_train, title=""):
    """Plot learning curves with detailed analysis"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    ax.plot(train_sizes, train_mean, label='Training', marker='o', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    ax.plot(train_sizes, val_mean, label='Validation', marker='o', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, train_mean, val_mean

print("""
LEARNING CURVE INTERPRETATION:

OVERFITTING (Large gap between curves):
  Training ↗↗↗ (high)
  Validation ↗ (low, far below training)
  → Training loss low, validation loss high
  → Model memorized training data
  → Fix: More regularization, simpler model, more data

UNDERFITTING (Both curves low, close together):
  Training ↗ (low)
  Validation ↗ (low, parallel to training)
  → Both losses high
  → Model too simple for problem
  → Fix: Increase complexity, better features

GOOD FIT (Curves close, both high):
  Training ↗↗ (high)
  Validation ↗↗ (high, close to training)
  → Model learning well
  → Generalization likely good
  → Can stop or fine-tune

HIGH VARIANCE (Curves very noisy):
  Curves jump around
  → Model sensitive to training set
  → Fix: More data, ensemble methods
  → Try: Cross-validation to average out noise
""")

# Generate example data
X_fit, y_fit = make_classification(n_samples=200, n_features=20, n_informative=10,
                                   n_classes=2, random_state=42)
X_train_fit, X_test_fit, y_train_fit, y_test_fit = train_test_split(
    X_fit, y_fit, test_size=0.3, random_state=42
)

# Overfit model
model_overfit = RandomForestClassifier(n_estimators=1, max_depth=15, random_state=42)
fig1, train1, val1 = plot_learning_curves_detailed(
    model_overfit, X_train_fit, y_train_fit,
    "Overfitting: High train, low validation"
)
plt.savefig('learning_curve_overfitting.png', dpi=150, bbox_inches='tight')
print("✅ Saved learning_curve_overfitting.png")

# Proper model
model_proper = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
fig2, train2, val2 = plot_learning_curves_detailed(
    model_proper, X_train_fit, y_train_fit,
    "Good Fit: Train and validation close"
)
plt.savefig('learning_curve_proper.png', dpi=150, bbox_inches='tight')
print("✅ Saved learning_curve_proper.png")

# ============================================================================
# 3. CROSS-VALIDATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("3️⃣  CROSS-VALIDATION ANALYSIS")
print("=" * 100)

print("""
CROSS-VALIDATION METRICS:
  • Mean: Average performance across folds
  • Std Dev: Consistency between folds
  • High std = unstable model (data-dependent)

INTERPRETATION:

High mean, low std (IDEAL):
  ✓ Model generalizes well
  ✓ Consistent across different data splits
  ✓ Production-ready

High mean, high std (UNSTABLE):
  ⚠️  Good average, but inconsistent
  ⚠️  Model depends on specific training examples
  ⚠️  Risky for production
  Fix: More data, ensemble methods, regularization

Low mean, low std (UNDERFITTING):
  ✗ Model not learning
  ✗ Consistently poor across all folds
  Fix: Increase complexity, better features

Low mean, high std (UNRELIABLE):
  ✗ Worst case - poor and inconsistent
  ✗ Can't trust predictions
  Fix: Investigate data quality, more samples
""")

# Calculate cross-validation scores
cv_scores_proper = cross_val_score(model_proper, X_train_fit, y_train_fit, cv=5, scoring='accuracy')
cv_scores_overfit = cross_val_score(model_overfit, X_train_fit, y_train_fit, cv=5, scoring='accuracy')

print(f"\nProper model CV scores:")
print(f"  Mean: {cv_scores_proper.mean():.4f}")
print(f"  Std:  {cv_scores_proper.std():.4f}")
print(f"  Range: {cv_scores_proper.min():.4f} - {cv_scores_proper.max():.4f}")

print(f"\nOverfit model CV scores:")
print(f"  Mean: {cv_scores_overfit.mean():.4f}")
print(f"  Std:  {cv_scores_overfit.std():.4f}")
print(f"  Range: {cv_scores_overfit.min():.4f} - {cv_scores_overfit.max():.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

folds = range(1, len(cv_scores_proper) + 1)
ax.plot(folds, cv_scores_proper, marker='o', label='Proper Model', linewidth=2)
ax.plot(folds, cv_scores_overfit, marker='s', label='Overfit Model', linewidth=2)

ax.axhline(cv_scores_proper.mean(), linestyle='--', alpha=0.5, label='Proper Mean')
ax.axhline(cv_scores_overfit.mean(), linestyle='--', alpha=0.5, label='Overfit Mean')

ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy')
ax.set_title('Cross-Validation Scores by Fold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cv_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved cv_analysis.png")

# ============================================================================
# 4. CONFUSION MATRIX ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("4️⃣  CONFUSION MATRIX DEEP DIVE")
print("=" * 100)

model_proper.fit(X_train_fit, y_train_fit)
y_pred_proper = model_proper.predict(X_test_fit)

cm = confusion_matrix(y_test_fit, y_pred_proper)

print(f"""
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test_fit, y_pred_proper)}

ANALYSIS:
  • TP={cm[1,1]}: True positives (correct positive predictions)
  • TN={cm[0,0]}: True negatives (correct negative predictions)
  • FP={cm[0,1]}: False positives (incorrect positive predictions)
  • FN={cm[1,0]}: False negatives (incorrect negative predictions)

ERROR PATTERNS:
""")

# Analyze error patterns
if cm[0, 1] > cm[1, 0]:
    print(f"  More FALSE POSITIVES ({cm[0,1]}) than FALSE NEGATIVES ({cm[1,0]})")
    print("  Model predicts 'positive' too often")
    print("  Solutions: Lower decision threshold, increase regularization")
else:
    print(f"  More FALSE NEGATIVES ({cm[1,0]}) than FALSE POSITIVES ({cm[0,1]})")
    print("  Model misses positive cases")
    print("  Solutions: Raise decision threshold, adjust class weights")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].set_title('Confusion Matrix')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].set_title('Normalized Confusion Matrix')

plt.tight_layout()
plt.savefig('confusion_matrix_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved confusion_matrix_analysis.png")

# ============================================================================
# 5. DEBUGGING CHECKLIST
# ============================================================================

print("\n" + "=" * 100)
print("5️⃣  SYSTEMATIC DEBUGGING CHECKLIST")
print("=" * 100)

debugging_checklist = """
🔍 DATA QUALITY CHECKS:
   □ Missing values: How many? Which features?
   □ Duplicates: Any exact duplicates?
   □ Data types: Are categorical features encoded?
   □ Outliers: Any suspicious extreme values?
   □ Class balance: Are classes balanced?
   □ Data leakage: Using future information?
   □ Scaling: Are features on similar scales?

🔍 TRAINING CHECKS:
   □ Random seed: Setting random_state for reproducibility?
   □ Train/test split: Stratified split for classification?
   □ Scaling: Fit on train, apply to test?
   □ Convergence: Did model converge during training?
   □ Loss curves: Are they decreasing?
   □ Baseline: Have you tested baseline models?
   □ Hyperparameters: Have you tuned key parameters?

🔍 EVALUATION CHECKS:
   □ Right metrics: Using appropriate metrics for problem?
   □ Test set: Evaluating on hold-out test set only?
   □ Cross-validation: Results consistent across folds?
   □ Learning curves: Checking train/validation gap?
   □ Confusion matrix: Analyzing error patterns?
   □ Baseline comparison: How much better than baseline?

🔍 PRODUCTION CHECKS:
   □ Preprocessing: Same preprocessing in production?
   □ Data drift: Monitoring input distributions?
   □ Prediction drift: Monitoring model outputs?
   □ Performance monitoring: Tracking metrics over time?
   □ Retraining: Plan for retraining on new data?
   □ Version control: Tracking model versions?

🔍 COMMON MISTAKES:
   □ Data leakage (fit scaler on full data)
   □ Reporting metrics on training set
   □ Changing metrics based on results
   □ Not comparing with baseline
   □ Tuning on test set
   □ Ignoring class imbalance
   □ Using inappropriate metrics
   □ Not handling missing values
   □ Forgetting to scale features
   □ Not doing cross-validation

"""

print(debugging_checklist)

# ============================================================================
# 6. SOLUTION STRATEGIES
# ============================================================================

print("\n" + "=" * 100)
print("6️⃣  SOLUTION STRATEGIES BY PROBLEM")
print("=" * 100)

strategies = """
PROBLEM: Low training accuracy
├─ Issue: Model isn't learning
├─ Quick Fixes:
│  1. Check data quality (labels correct?)
│  2. Increase model complexity
│  3. Increase learning rate / iterations
│  4. Better features (feature engineering)
│  5. Reduce regularization
└─ Root Causes:
   • Data too noisy / mislabeled
   • Model too simple
   • Poor features
   • Wrong algorithm for problem

PROBLEM: Good training, bad test accuracy (OVERFITTING)
├─ Issue: Memorized training data
├─ Quick Fixes:
│  1. Reduce model complexity
│  2. Increase regularization
│  3. Add more training data
│  4. Remove features (feature selection)
│  5. Early stopping
│  6. Dropout / Layer dropout
└─ Root Causes:
   • Too many parameters
   • Not enough data
   • Noise in training data
   • Too much capacity

PROBLEM: Both train & test low (UNDERFITTING)
├─ Issue: Model too simple
├─ Quick Fixes:
│  1. Increase model complexity
│  2. Better features
│  3. Reduce regularization
│  4. Increase training iterations
│  5. Try different algorithm
└─ Root Causes:
   • Model not powerful enough
   • Poor features
   • Missing important features
   • Wrong algorithm

PROBLEM: High variance across folds
├─ Issue: Model sensitive to data splits
├─ Quick Fixes:
│  1. More training data
│  2. Ensemble methods (bagging, boosting)
│  3. Stratified k-fold
│  4. Regularization
│  5. Feature selection
└─ Root Causes:
   • Too few samples
   • Noise in data
   • High model complexity

PROBLEM: Good validation, bad production
├─ Issue: Training data differs from production
├─ Quick Fixes:
│  1. Monitor data distributions
│  2. Retrain on recent data
│  3. Check for data drift
│  4. Collect production examples
│  5. A/B test before deployment
└─ Root Causes:
   • Data distribution changed
   • Data quality issues
   • Different data sources
   • Seasonal effects

PROBLEM: Imbalanced classification
├─ Issue: One class much rarer
├─ Quick Fixes:
│  1. Use appropriate metrics (F1, ROC-AUC)
│  2. Class weights / sample weights
│  3. Oversample minority / undersample majority
│  4. SMOTE (synthetic oversampling)
│  5. Ensemble methods
└─ Root Causes:
   • Naturally imbalanced problem
   • Biased data collection

"""

print(strategies)

# ============================================================================
# 7. DEBUGGING WORKFLOW
# ============================================================================

print("\n" + "=" * 100)
print("7️⃣  COMPLETE DEBUGGING WORKFLOW")
print("=" * 100)

workflow = """
STEP 1: VERIFY BASELINE
  1. Create simple baseline (sklearn defaults)
  2. Measure performance on test set
  3. This is your starting point

STEP 2: CHECK DATA
  1. Examine raw data (plots, statistics)
  2. Check for missing values, duplicates
  3. Verify labels are correct
  4. Check class balance
  5. Look for data quality issues

STEP 3: ANALYZE TRAINING
  1. Plot learning curves (train vs validation)
  2. Calculate cross-validation scores
  3. Check for overfitting/underfitting gap
  4. Monitor loss curves during training

STEP 4: ANALYZE PREDICTIONS
  1. Confusion matrix analysis
  2. Error pattern analysis
  3. Per-class performance
  4. Prediction distribution

STEP 5: SYSTEMATIC IMPROVEMENTS
  1. Feature engineering → check improvement
  2. Hyperparameter tuning → check improvement
  3. Model selection → try different models
  4. Ensemble methods → combine models

STEP 6: PRODUCTION CHECK
  1. Evaluate on test set (not training!)
  2. Cross-validate across splits
  3. Check inference speed
  4. Set up monitoring

STEP 7: DOCUMENT & DEPLOY
  1. Document all decisions
  2. Create reproducible pipeline
  3. Set up version control
  4. Monitor in production

EXAMPLE CODE TEMPLATE:

from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, classification_report

# 1. Baseline
baseline_model = LogisticRegression()
baseline_score = cross_val_score(baseline_model, X, y, cv=5).mean()
print(f"Baseline: {baseline_score:.4f}")

# 2. Check data
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Class balance: {pd.Series(y).value_counts()}")

# 3. Learning curves
train_sizes, train_scores, val_scores = learning_curve(baseline_model, X, y)
# Plot and analyze

# 4. Confusion matrix
y_pred = baseline_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Try improvements
better_model = RandomForestClassifier(n_estimators=100)
better_score = cross_val_score(better_model, X, y, cv=5).mean()
print(f"Improved: {better_score:.4f} (improvement: {better_score - baseline_score:.4f})")

# 6. Evaluate
better_model.fit(X_train, y_train)
final_score = better_model.score(X_test, y_test)
print(f"Final test score: {final_score:.4f}")
"""

print(workflow)

print("\n" + "=" * 100)
print("✅ ERROR ANALYSIS & DEBUGGING GUIDE COMPLETE")
print("=" * 100)
````

---

## ✅ Summary

I've created **4 comprehensive production-ready guides**:

### 1. **Hyperparameter Tuning Guide** (`hyperparameter_tuning_guide.py`)
   - Grid Search, Randomized Search, Bayesian Optimization, Hyperband
   - Manual tuning workflow with examples
   - Parameter tuning guides for RF, XGBoost, SVM, LR, Neural Networks
   - Best practices and comparison table

### 2. **Model Evaluation Metrics** (`model_evaluation_metrics.py`)
   - Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Regression metrics: MAE, RMSE, MAPE, R²
   - Imbalanced data metrics
   - Complete metric selection guide with examples

### 3. **Feature Engineering Cookbook** (`feature_engineering_cookbook.py`)
   - Handling missing values (removal, imputation, KNN, iterative)
   - Outlier detection (Z-score, IQR, Isolation Forest)
   - Scaling & normalization techniques
   - Polynomial & interaction features
   - Feature selection methods
   - PCA & dimensionality reduction
   - Categorical encoding

### 4. **Error Analysis & Debugging** (`error_analysis_debugging.py`)
   - Diagnostic decision tree
   - Learning curves interpretation
   - Cross-validation analysis
   - Confusion matrix deep dive
   - Systematic debugging checklist
   - Solution strategies by problem
   - Complete debugging workflow

---

Would you like me to create:
- ✅ **Model Deployment & Production** - Docker, APIs, monitoring
- ✅ **MLOps Pipeline** - CI/CD, model versioning, experiment tracking
- ✅ **Complete End-to-End Projects** - Full ML pipelines
- ✅ **Advanced Techniques** - Ensemble methods, AutoML, Advanced NN

# 🚀 ADVANCED ML PRODUCTION SUITE - PART 6-8

Creating comprehensive guides for Model Deployment, MLOps Pipelines, and Complete End-to-End Projects.

---

## 🐳 CREATE: Model Deployment & Production Guide

````python
"""
Complete Model Deployment & Production Guide
Docker, APIs, Monitoring, and Best Practices
"""

import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

print("=" * 100)
print("MODEL DEPLOYMENT & PRODUCTION GUIDE")
print("=" * 100)

# ============================================================================
# PART 1: MODEL PERSISTENCE & SERIALIZATION
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  MODEL PERSISTENCE & SERIALIZATION")
print("=" * 100)

print("""
SERIALIZATION FORMATS:

1. JOBLIB (Recommended for sklearn)
   ✓ Best for sklearn models
   ✓ Fast, efficient
   ✓ Handles numpy arrays well
   ✗ Python-only
   Use: Production sklearn models
   
2. PICKLE (Standard Python)
   ✓ Native Python
   ✓ Handles most objects
   ✗ Security issues (arbitrary code execution)
   ✗ Version compatibility issues
   Use: Quick prototyping (NOT production!)
   
3. ONNX (Open Neural Network Exchange)
   ✓ Cross-platform
   ✓ Language-agnostic
   ✓ Production-ready
   ✗ Limited model support
   Use: Production deployment, model exchange
   
4. SavedModel (TensorFlow)
   ✓ Complete serialization
   ✓ Serves in TensorFlow Serving
   ✗ TensorFlow-specific
   Use: Deep learning models
   
5. PyTorch Format
   ✓ Flexible
   ✓ Lightweight
   ✗ Python-dependent
   Use: PyTorch models

BEST PRACTICE APPROACH:
   1. Save model with joblib/pickle
   2. Save preprocessing pipeline
   3. Save metadata (training date, version)
   4. Save hyperparameters
   5. Version everything
""")

# Train example model
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\n📊 Model Performance:")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Save model using joblib
print(f"\n💾 Saving Model with JOBLIB...")
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Saved: model.pkl, scaler.pkl")

# Save metadata
metadata = {
    'model_name': 'cancer_classifier',
    'model_type': 'RandomForestClassifier',
    'training_date': datetime.now().isoformat(),
    'training_accuracy': float(train_score),
    'test_accuracy': float(test_score),
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'features': list(cancer.feature_names),
    'n_features': len(cancer.feature_names),
    'classes': [0, 1],
    'version': '1.0.0'
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: model_metadata.json")

# Load and verify
print(f"\n🔄 Loading Model...")
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_metadata = json.load(open('model_metadata.json'))

print(f"✅ Loaded model version: {loaded_metadata['version']}")
print(f"✅ Model type: {loaded_metadata['model_type']}")

# ============================================================================
# PART 2: FLASK API FOR MODEL SERVING
# ============================================================================

print("\n" + "=" * 100)
print("2️⃣  FLASK API FOR MODEL SERVING")
print("=" * 100)

print("""
FLASK API ARCHITECTURE:

1. Load model on startup
2. Create REST endpoints
3. Handle input validation
4. Make predictions
5. Return JSON responses
6. Log requests/predictions

ENDPOINTS:

POST /predict
├─ Input: JSON with features
├─ Process: Preprocess, predict, confidence
└─ Output: JSON with prediction, confidence, timestamp

GET /health
├─ Check if service is running
└─ Return status

GET /metadata
└─ Return model metadata

POST /batch_predict
├─ Predict multiple samples
└─ Return array of predictions
""")

# Create Flask app code (save as app.py)
flask_app_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/app.py

"""
Production Flask API for ML Model
Handles predictions, monitoring, and health checks
"""

import joblib
import json
import numpy as np
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ============ SETUP ============

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model and preprocessing
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
METADATA_PATH = 'model_metadata.json'

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

logger.info(f"Model loaded: {metadata['model_name']} v{metadata['version']}")
logger.info(f"Model accuracy (test): {metadata['test_accuracy']:.4f}")

# ============ HELPER FUNCTIONS ============

def validate_input(data, expected_features):
    """Validate input features"""
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    
    if 'features' not in data:
        raise ValueError("Missing 'features' key")
    
    features = data['features']
    if not isinstance(features, list):
        raise ValueError("'features' must be a list")
    
    if len(features) != len(expected_features):
        raise ValueError(f"Expected {len(expected_features)} features, got {len(features)}")
    
    return np.array(features).reshape(1, -1)

def make_prediction(X):
    """Make prediction with confidence"""
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    
    return {
        'prediction': int(prediction),
        'confidence': float(confidence),
        'probabilities': {
            'class_0': float(probabilities[0]),
            'class_1': float(probabilities[1])
        }
    }

# ============ API ENDPOINTS ============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_version': metadata['version']
    }), 200

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Return model metadata"""
    return jsonify(metadata), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Make single prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        X = validate_input(data, metadata['features'])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        result = make_prediction(X_scaled)
        
        # Log prediction
        logger.info(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.4f}")
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make multiple predictions"""
    try:
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'status': 'error', 'message': "Missing 'samples' key"}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({'status': 'error', 'message': "'samples' must be a list"}), 400
        
        results = []
        for sample in samples:
            try:
                X = validate_input({'features': sample}, metadata['features'])
                X_scaled = scaler.transform(X)
                result = make_prediction(X_scaled)
                results.append({'status': 'success', 'result': result})
            except Exception as e:
                results.append({'status': 'error', 'message': str(e)})
        
        logger.info(f"Batch prediction: {len(results)} samples processed")
        
        return jsonify({
            'status': 'success',
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback for model improvement"""
    try:
        data = request.get_json()
        
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'features': data.get('features'),
            'prediction': data.get('prediction'),
            'actual': data.get('actual'),
            'correct': data.get('correct'),
            'notes': data.get('notes', '')
        }
        
        # Log feedback
        with open('feedback.log', 'a') as f:
            f.write(json.dumps(feedback) + '\\n')
        
        logger.info(f"Feedback recorded: Correct={feedback['correct']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded'
        }), 201
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to record feedback'}), 500

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'status': 'error', 'message': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============ MAIN ============

if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # NEVER debug=True in production
        threaded=True
    )
'''

print("\n💾 Flask API Code (saved as app.py)")
print("Key features:")
print("  ✓ Input validation")
print("  ✓ Error handling")
print("  ✓ Logging")
print("  ✓ Batch predictions")
print("  ✓ Feedback collection")
print("  ✓ Health checks")

# Save Flask app
with open('app.py', 'w') as f:
    f.write(flask_app_code)
print("✅ Saved: app.py")

# ============================================================================
# PART 3: DOCKER CONTAINERIZATION
# ============================================================================

print("\n" + "=" * 100)
print("3️⃣  DOCKER CONTAINERIZATION")
print("=" * 100)

print("""
WHY DOCKER?
  ✓ Consistency: Same environment everywhere
  ✓ Isolation: Dependencies don't conflict
  ✓ Reproducibility: Easy deployment
  ✓ Scaling: Easy horizontal scaling
  ✓ CI/CD: Automate build and deployment

DOCKERFILE:
  Base image → Install dependencies → Copy code → Expose port → Run app
""")

# Create Dockerfile
dockerfile_content = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .
COPY model_metadata.json .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
'''

with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)
print("✅ Saved: Dockerfile")

# Create requirements.txt
requirements = """flask==2.3.0
flask-cors==3.0.10
scikit-learn==1.2.0
joblib==1.2.0
numpy==1.23.0
pandas==1.5.0
gunicorn==20.1.0
requests==2.28.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)
print("✅ Saved: requirements.txt")

# Create docker-compose for easier orchestration
docker_compose = """# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/docker-compose.yml

version: '3.8'

services:
  ml_api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./feedback.log:/app/feedback.log
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  # Optional: Monitoring service
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Optional: Log aggregation
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.13.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
"""

with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose)
print("✅ Saved: docker-compose.yml")

print("""
DOCKER COMMANDS:

1. Build image:
   $ docker build -t ml-api:latest .

2. Run container:
   $ docker run -p 5000:5000 ml-api:latest

3. Using docker-compose:
   $ docker-compose up -d

4. View logs:
   $ docker-compose logs -f ml_api

5. Stop container:
   $ docker-compose down

6. Push to registry:
   $ docker tag ml-api:latest username/ml-api:latest
   $ docker push username/ml-api:latest
""")

# ============================================================================
# PART 4: API CLIENT & TESTING
# ============================================================================

print("\n" + "=" * 100)
print("4️⃣  API CLIENT & TESTING")
print("=" * 100)

# Create API client code
client_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/api_client.py

"""
Python client for ML model API
"""

import requests
import json
from typing import Dict, List, Any
import numpy as np

class MLModelClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        response = self.session.get(f"{self.base_url}/metadata")
        response.raise_for_status()
        return response.json()
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make single prediction"""
        payload = {"features": features}
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, samples: List[List[float]]) -> Dict[str, Any]:
        """Make multiple predictions"""
        payload = {"samples": samples}
        response = self.session.post(
            f"{self.base_url}/batch_predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def send_feedback(self, features: List[float], prediction: int, 
                     actual: int, notes: str = "") -> Dict[str, Any]:
        """Send feedback for model improvement"""
        payload = {
            "features": features,
            "prediction": prediction,
            "actual": actual,
            "correct": prediction == actual,
            "notes": notes
        }
        response = self.session.post(
            f"{self.base_url}/feedback",
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    client = MLModelClient()
    
    # Check health
    print("Health check:", client.health_check())
    
    # Get metadata
    metadata = client.get_metadata()
    print(f"Model: {metadata['model_name']} v{metadata['version']}")
    
    # Single prediction
    sample = np.random.randn(30).tolist()
    result = client.predict(sample)
    print(f"Prediction: {result['result']['prediction']}, Confidence: {result['result']['confidence']:.4f}")
    
    # Batch prediction
    samples = [np.random.randn(30).tolist() for _ in range(5)]
    results = client.batch_predict(samples)
    print(f"Batch processed: {results['count']} samples")
    
    # Send feedback
    feedback = client.send_feedback(sample, 1, 1, "Test feedback")
    print("Feedback status:", feedback['status'])
'''

with open('api_client.py', 'w') as f:
    f.write(client_code)
print("✅ Saved: api_client.py")

# Create test suite
test_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/test_api.py

"""
Test suite for ML API
Run with: pytest test_api.py -v
"""

import pytest
import json
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app import app, model, scaler, metadata
import numpy as np

@pytest.fixture
def client():
    """Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'model_version' in data

def test_metadata(client):
    """Test metadata endpoint"""
    response = client.get('/metadata')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_name' in data
    assert 'version' in data

def test_single_prediction(client):
    """Test single prediction"""
    features = np.random.randn(30).tolist()
    payload = {"features": features}
    
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'result' in data
    assert 'prediction' in data['result']
    assert 'confidence' in data['result']

def test_batch_prediction(client):
    """Test batch prediction"""
    samples = [np.random.randn(30).tolist() for _ in range(5)]
    payload = {"samples": samples}
    
    response = client.post(
        '/batch_predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['count'] == 5

def test_invalid_features(client):
    """Test with invalid features"""
    payload = {"features": [1, 2, 3]}  # Wrong number
    
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'

def test_missing_features(client):
    """Test with missing features key"""
    payload = {}
    
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 400

def test_not_found(client):
    """Test 404 error"""
    response = client.get('/nonexistent')
    assert response.status_code == 404

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''

with open('test_api.py', 'w') as f:
    f.write(test_code)
print("✅ Saved: test_api.py")

# ============================================================================
# PART 5: MONITORING & LOGGING
# ============================================================================

print("\n" + "=" * 100)
print("5️⃣  MONITORING & LOGGING")
print("=" * 100)

print("""
MONITORING METRICS:

1. BUSINESS METRICS:
   • Prediction accuracy
   • Model performance (daily, weekly)
   • User feedback
   • Model drift detection

2. OPERATIONAL METRICS:
   • API response time
   • Request volume
   • Error rate
   • CPU/memory usage
   • Model inference time

3. DATA METRICS:
   • Input feature distributions
   • Missing values
   • Outliers
   • Data drift

LOG LEVELS:
  DEBUG: Detailed debugging info
  INFO: General information
  WARNING: Warning messages
  ERROR: Error messages
  CRITICAL: Critical failures

TOOLS:
  • Prometheus: Metrics collection
  • Grafana: Visualization
  • ELK Stack: Log aggregation
  • Sentry: Error tracking
""")

# Create monitoring script
monitoring_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/monitor.py

"""
Model monitoring and drift detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, baseline_stats: dict = None):
        """Initialize monitor with baseline statistics"""
        self.baseline_stats = baseline_stats or {}
        self.history = []
    
    def log_prediction(self, features: np.ndarray, prediction: int, 
                      confidence: float, actual: int = None):
        """Log prediction for monitoring"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'features_mean': float(features.mean()),
            'features_std': float(features.std()),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'actual': actual,
            'correct': None if actual is None else int(prediction == actual)
        }
        self.history.append(record)
        return record
    
    def detect_data_drift(self, window: int = 100) -> dict:
        """Detect distribution drift in recent data"""
        if len(self.history) < window:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        recent = self.history[-window:]
        recent_means = [r['features_mean'] for r in recent]
        
        if not self.baseline_stats:
            return {'drift_detected': False, 'reason': 'No baseline'}
        
        baseline_mean = self.baseline_stats.get('mean', 0)
        recent_avg = np.mean(recent_means)
        
        # Simple drift detection: if mean shifted >20%
        drift_threshold = abs(baseline_mean) * 0.2
        drift_detected = abs(recent_avg - baseline_mean) > drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'baseline_mean': baseline_mean,
            'recent_mean': recent_avg,
            'drift_magnitude': abs(recent_avg - baseline_mean)
        }
    
    def detect_prediction_drift(self, window: int = 100) -> dict:
        """Detect drift in model predictions"""
        if len(self.history) < window:
            return {'drift_detected': False}
        
        recent = self.history[-window:]
        confidences = [r['confidence'] for r in recent]
        
        avg_confidence = np.mean(confidences)
        
        return {
            'average_confidence': avg_confidence,
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'uncertainty_increasing': np.mean(confidences[-50:]) < np.mean(confidences[:50])
        }
    
    def calculate_performance(self, window: int = 100) -> dict:
        """Calculate recent model performance"""
        if len(self.history) < window:
            return {}
        
        recent = self.history[-window:]
        correct = [r['correct'] for r in recent if r['correct'] is not None]
        
        if not correct:
            return {}
        
        accuracy = np.mean(correct)
        
        return {
            'recent_accuracy': accuracy,
            'sample_size': len(correct),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_summary(self) -> dict:
        """Get comprehensive monitoring summary"""
        data_drift = self.detect_data_drift()
        pred_drift = self.detect_prediction_drift()
        performance = self.calculate_performance()
        
        return {
            'data_drift': data_drift,
            'prediction_drift': pred_drift,
            'performance': performance,
            'total_predictions': len(self.history),
            'monitoring_since': self.history[0]['timestamp'] if self.history else None
        }
    
    def save_history(self, filepath: str):
        """Save monitoring history to file"""
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.history)} records to {filepath}")
    
    def alert_if_drift(self, threshold: float = 0.2):
        """Alert if drift detected"""
        summary = self.get_summary()
        
        if summary['data_drift']['drift_detected']:
            logger.warning(f"Data drift detected: {summary['data_drift']}")
        
        if summary['prediction_drift'].get('uncertainty_increasing'):
            logger.warning("Prediction uncertainty is increasing")
        
        if summary['performance']:
            perf = summary['performance']
            if perf['recent_accuracy'] < (1 - threshold):
                logger.error(f"Model accuracy dropped below {1-threshold}: {perf['recent_accuracy']:.4f}")
'''

with open('monitor.py', 'w') as f:
    f.write(monitoring_code)
print("✅ Saved: monitor.py")

# ============================================================================
# PART 6: DEPLOYMENT CHECKLIST
# ============================================================================

print("\n" + "=" * 100)
print("6️⃣  PRODUCTION DEPLOYMENT CHECKLIST")
print("=" * 100)

checklist = """
🔍 PRE-DEPLOYMENT CHECKS:

DATA & MODEL:
  ☐ Model validated on test set
  ☐ Cross-validation scores good
  ☐ Confusion matrix analyzed
  ☐ Test metrics documented
  ☐ Model version control set up
  ☐ Training data preserved
  ☐ Model card created (metadata)
  ☐ Feature importance documented

CODE QUALITY:
  ☐ Code reviewed
  ☐ Unit tests written (>80% coverage)
  ☐ Integration tests written
  ☐ API tests passing
  ☐ Error handling implemented
  ☐ Logging implemented
  ☐ Documentation complete
  ☐ Code formatted (black/pylint)

SECURITY:
  ☐ Input validation implemented
  ☐ Rate limiting set up
  ☐ Authentication required
  ☐ No credentials in code
  ☐ Secrets management configured
  ☐ HTTPS/TLS enabled
  ☐ CORS properly configured
  ☐ SQL injection/XSS protection (if applicable)

DEPLOYMENT:
  ☐ Docker image built and tested
  ☐ docker-compose tested locally
  ☐ Environment variables documented
  ☐ Scaling strategy defined
  ☐ Rollback plan ready
  ☐ Infrastructure as code ready
  ☐ Database migrations ready (if needed)
  ☐ DNS/load balancing configured

MONITORING:
  ☐ Monitoring dashboards created
  ☐ Alerts configured
  ☐ Logging aggregation set up
  ☐ Health check endpoint ready
  ☐ Metrics collection configured
  ☐ Baseline metrics documented
  ☐ SLO/SLA defined
  ☐ On-call rotation assigned

DOCUMENTATION:
  ☐ API documentation (Swagger/OpenAPI)
  ☐ Deployment guide written
  ☐ Runbook created
  ☐ Troubleshooting guide
  ☐ Performance baseline documented
  ☐ Model limitations documented
  ☐ Data requirements documented
  ☐ Dependencies documented

PRODUCTION OPERATION:
  ☐ Gradual rollout plan (canary/blue-green)
  ☐ Feature flags implemented
  ☐ A/B testing setup
  ☐ User feedback collection
  ☐ Performance monitoring
  ☐ Data drift monitoring
  ☐ Prediction drift monitoring
  ☐ Retraining schedule planned

🚀 DEPLOYMENT STRATEGY:

1. CANARY DEPLOYMENT (Recommended):
   • Deploy to 5% of traffic
   • Monitor metrics for 24 hours
   • If good, gradually increase to 100%
   • If bad, rollback immediately
   • Minimizes risk

2. BLUE-GREEN DEPLOYMENT:
   • Two identical production environments
   • Deploy to "green" environment
   • Test completely
   • Switch traffic to green
   • Easy rollback by switching back to blue

3. ROLLING DEPLOYMENT:
   • Update servers one at a time
   • Gradual transition
   • Can have multiple versions running
   • More complex monitoring

4. SHADOW DEPLOYMENT:
   • Run new model alongside old
   • Send same data to both
   • Compare predictions
   • No user impact
   • Best for testing

✅ POST-DEPLOYMENT:

First 24 hours:
  ☐ Monitor all metrics closely
  ☐ Check for errors in logs
  ☐ Verify predictions look reasonable
  ☐ Collect user feedback
  ☐ Monitor infrastructure usage

First week:
  ☐ Monitor performance trends
  ☐ Check for data drift
  ☐ Analyze prediction distribution
  ☐ Review error patterns
  ☐ Validate against business metrics

Ongoing:
  ☐ Weekly performance reviews
  ☐ Monthly model evaluation
  ☐ Quarterly retraining cycles
  ☐ Continuous monitoring
  ☐ Plan next version

"""

print(checklist)

print("\n" + "=" * 100)
print("✅ MODEL DEPLOYMENT GUIDE COMPLETE")
print("=" * 100)
````

---

## 🔄 CREATE: MLOps Pipeline & CI/CD Guide

````python
"""
Complete MLOps Pipeline Guide
CI/CD, Model Versioning, Experiment Tracking, Retraining
"""

import json
import os
import yaml
from datetime import datetime
from pathlib import Path

print("=" * 100)
print("MLOps PIPELINE & CI/CD GUIDE")
print("=" * 100)

# ============================================================================
# PART 1: EXPERIMENT TRACKING
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  EXPERIMENT TRACKING - MLflow, Weights & Biases")
print("=" * 100)

print("""
WHY EXPERIMENT TRACKING?
  ✓ Compare multiple model versions
  ✓ Reproduce results
  ✓ Track hyperparameters
  ✓ Visualize metrics
  ✓ Collaborate with team

TOOLS:
  1. MLflow (Open-source, local/remote)
  2. Weights & Biases (Cloud-based, advanced features)
  3. Kubeflow (Kubernetes-native)
  4. Neptune (Enterprise)

WHAT TO TRACK:
  • Hyperparameters
  • Metrics (train/val/test)
  • Artifacts (models, plots)
  • Code version (git commit)
  • Data version
  • Environment (dependencies)
  • Execution time
  • System metrics (GPU, memory)
""")

# MLflow example code
mlflow_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/train_with_mlflow.py

"""
Model training with MLflow experiment tracking
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set experiment
mlflow.set_experiment("cancer_classifier")

# Hyperparameters
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

with mlflow.start_run(run_name="rf_baseline"):
    logger.info(f"Starting MLflow run with params: {params}")
    
    # Log parameters
    mlflow.log_params(params)
    
    # Load and prepare data
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Log metrics
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        logger.info(f"{name}: {value:.4f}")
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': cancer.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log dataset info
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", X.shape[0])
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    
    logger.info(f"MLflow run completed. Check http://localhost:5000")

# View results
print("Run 'mlflow ui' to view results")
'''

with open('train_with_mlflow.py', 'w') as f:
    f.write(mlflow_code)
print("✅ Saved: train_with_mlflow.py")

# ============================================================================
# PART 2: MODEL VERSIONING & REGISTRY
# ============================================================================

print("\n" + "=" * 100)
print("2️⃣  MODEL VERSIONING & REGISTRY")
print("=" * 100)

print("""
VERSION CONTROL STRATEGIES:

1. SEMANTIC VERSIONING (MAJOR.MINOR.PATCH)
   • 1.0.0 → First production version
   • 1.1.0 → New feature
   • 1.0.1 → Bug fix
   Use: Production models

2. GIT-BASED VERSIONING
   • Tag each trained model with commit hash
   • Full reproducibility
   Use: Research/development

3. MODEL REGISTRY
   • Central repository
   • Track production/staging/archived
   • Promote to production
   Tools: MLflow Registry, Kubernetes

METADATA TO TRACK:
  • Version number
  • Training date
  • Training data version
  • Git commit hash
  • Hyperparameters
  • Metrics
  • Status (development/staging/production)
  • Author
  • Comments
""")

# Create model registry code
registry_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/model_registry.py

"""
Model registry for version management
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class ModelRegistry:
    def __init__(self, registry_dir: str = "./model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
    
    def register_model(self, model_path: str, version: str, 
                      metrics: Dict, metadata: Dict) -> Dict:
        """Register a new model version"""
        
        registry = self._load_registry()
        
        model_info = {
            'version': version,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata,
            'status': 'registered'  # registered, staging, production, archived
        }
        
        if version in registry:
            raise ValueError(f"Version {version} already exists")
        
        registry[version] = model_info
        self._save_registry(registry)
        
        print(f"✅ Model v{version} registered")
        return model_info
    
    def promote_to_staging(self, version: str):
        """Promote model to staging"""
        registry = self._load_registry()
        if version not in registry:
            raise ValueError(f"Version {version} not found")
        
        # Demote current staging
        for v, info in registry.items():
            if info['status'] == 'staging':
                info['status'] = 'registered'
        
        # Promote new version
        registry[version]['status'] = 'staging'
        registry[version]['staging_date'] = datetime.now().isoformat()
        
        self._save_registry(registry)
        print(f"✅ Model v{version} promoted to staging")
    
    def promote_to_production(self, version: str):
        """Promote model to production"""
        registry = self._load_registry()
        if version not in registry:
            raise ValueError(f"Version {version} not found")
        
        # Demote current production
        for v, info in registry.items():
            if info['status'] == 'production':
                info['status'] = 'archived'
                info['archived_date'] = datetime.now().isoformat()
        
        # Promote new version
        registry[version]['status'] = 'production'
        registry[version]['production_date'] = datetime.now().isoformat()
        
        self._save_registry(registry)
        print(f"✅ Model v{version} promoted to production")
    
    def get_production_model(self) -> Dict:
        """Get current production model"""
        registry = self._load_registry()
        for version, info in registry.items():
            if info['status'] == 'production':
                return info
        return None
    
    def get_model_info(self, version: str) -> Dict:
        """Get model info by version"""
        registry = self._load_registry()
        return registry.get(version)
    
    def list_models(self, status: str = None) -> List[Dict]:
        """List all models, optionally filtered by status"""
        registry = self._load_registry()
        models = list(registry.values())
        
        if status:
            models = [m for m in models if m['status'] == status]
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def _load_registry(self) -> Dict:
        """Load registry from file"""
        if not self.registry_file.exists():
            return {}
        
        with open(self.registry_file, 'r') as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

# Example usage
if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Register models
    registry.register_model(
        model_path="models/model_v1.pkl",
        version="1.0.0",
        metrics={'accuracy': 0.95, 'f1': 0.93},
        metadata={'author': 'data_team', 'description': 'Initial baseline'}
    )
    
    registry.register_model(
        model_path="models/model_v2.pkl",
        version="1.1.0",
        metrics={'accuracy': 0.96, 'f1': 0.95},
        metadata={'author': 'data_team', 'description': 'Improved features'}
    )
    
    # Promote
    registry.promote_to_staging("1.1.0")
    registry.promote_to_production("1.1.0")
    
    # View
    print(f"Production: {registry.get_production_model()['version']}")
    print(f"All models: {len(registry.list_models())}")
'''

with open('model_registry.py', 'w') as f:
    f.write(registry_code)
print("✅ Saved: model_registry.py")

# ============================================================================
# PART 3: CI/CD WITH GITHUB ACTIONS
# ============================================================================

print("\n" + "=" * 100)
print("3️⃣  CI/CD PIPELINE - GitHub Actions")
print("=" * 100)

print("""
CI/CD PIPELINE STAGES:

1. TRIGGER (on push/PR)
   └─ Run on every code change

2. BUILD
   └─ Build Docker image
   └─ Install dependencies

3. TEST
   └─ Unit tests
   └─ Integration tests
   └─ API tests
   └─ Model validation

4. TRAIN (optional)
   └─ Retrain model with new data
   └─ Validate performance

5. DEPLOY
   └─ Push Docker image
   └─ Deploy to staging/production
   └─ Run smoke tests

6. MONITOR
   └─ Collect metrics
   └─ Alert on failures
""")

# Create GitHub Actions workflow
github_workflow = """# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/.github/workflows/ml_pipeline.yml

name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly retraining

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8
    
    - name: Code quality
      run: |
        black --check .
        flake8 . --max-line-length=100
    
    - name: Unit tests
      run: |
        pytest tests/ -v --cov=app
    
    - name: API tests
      run: |
        pytest test_api.py -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: |
        docker build -t ml-api:${{ github.sha }} .
        docker tag ml-api:${{ github.sha }} ml-api:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ml-api:${{ github.sha }}
        docker push ml-api:latest

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to staging
      run: |
        curl -X POST ${{ secrets.STAGING_WEBHOOK }} \\
          -H 'Content-Type: application/json' \\
          -d '{"image": "ml-api:${{ github.sha }}"}'
    
    - name: Run smoke tests
      run: |
        python smoke_tests.py ${{ secrets.STAGING_URL }}

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Approve deployment
      run: |
        echo "Manual approval required for production deployment"
    
    - name: Deploy to production
      run: |
        curl -X POST ${{ secrets.PROD_WEBHOOK }} \\
          -H 'Content-Type: application/json' \\
          -d '{"image": "ml-api:${{ github.sha }}"}'

  retrain:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Retrain model
      run: |
        python train_pipeline.py --data-path data/production/
    
    - name: Validate model
      run: |
        python validate_model.py
    
    - name: Push to registry
      run: |
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add models/
        git commit -m "Automated retraining"
        git push
"""

with open('.github/workflows/ml_pipeline.yml', 'w') as f:
    f.write(github_workflow)
print("✅ Saved: .github/workflows/ml_pipeline.yml")

# ============================================================================
# PART 4: AUTOMATED RETRAINING
# ============================================================================

print("\n" + "=" * 100)
print("4️⃣  AUTOMATED RETRAINING PIPELINE")
print("=" * 100)

print("""
RETRAINING TRIGGERS:

1. SCHEDULED (Weekly/Monthly)
   • Regular updates
   • Fresh data ingestion
   • Use: Most common

2. PERFORMANCE DEGRADATION
   • Accuracy drops below threshold
   • Data drift detected
   • Prediction drift detected
   • Use: Reactive

3. MANUAL TRIGGER
   • Data scientist initiated
   • New hyperparameters
   • New features
   • Use: Development

RETRAINING PIPELINE:
  1. Collect new data
  2. Validate data quality
  3. Train new model
  4. Validate performance
  5. Compare with current
  6. Promote if better
  7. Monitor in production
""")

# Create retraining script
retrain_code = '''
# filepath: /Users/kishorkumarparoi/Desktop/mlops-project/retrain_pipeline.py

"""
Automated retraining pipeline
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import json

logger = logging.getLogger(__name__)

class RetrainingPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Run complete retraining pipeline"""
        
        logger.info("=" * 50)
        logger.info("RETRAINING PIPELINE STARTED")
        logger.info("=" * 50)
        
        try:
            # Step 1: Load data
            logger.info("Loading data...")
            X_train, y_train, X_test, y_test = self._load_data()
            
            # Step 2: Validate data
            logger.info("Validating data...")
            self._validate_data(X_train, y_train)
            
            # Step 3: Preprocess
            logger.info("Preprocessing...")
            X_train_scaled, X_test_scaled, scaler = self._preprocess(X_train, X_test)
            
            # Step 4: Train
            logger.info("Training model...")
            model = self._train(X_train_scaled, y_train)
            
            # Step 5: Evaluate
            logger.info("Evaluating model...")
            train_metrics = self._evaluate(model, X_train_scaled, y_train, "training")
            test_metrics = self._evaluate(model, X_test_scaled, y_test, "test")
            
            # Step 6: Compare with baseline
            logger.info("Comparing with baseline...")
            baseline_metrics = self._load_baseline()
            
            if self._is_improvement(test_metrics, baseline_metrics):
                logger.info("✅ New model is better! Saving...")
                self._save_model(model, scaler, test_metrics)
                logger.info("=" * 50)
                logger.info("RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
                logger.info("=" * 50)
                return True
            else:
                logger.warning("❌ New model not better than baseline. Discarding...")
                return False
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False
    
    def _load_data(self):
        """Load training and test data"""
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
        return X_train, y_train, X_test, y_test
    
    def _validate_data(self, X, y):
        """Validate data quality"""
        assert X.shape[0] == y.shape[0], "Data size mismatch"
        assert not np.isnan(X).any(), "NaN values found"
        assert not np.isinf(X).any(), "Inf values found"
        logger.info("✅ Data validation passed")
    
    def _preprocess(self, X_train, X_test):
        """Preprocess data"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    def _train(self, X_train, y_train):
        """Train model"""
        model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 10),
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def _evaluate(self, model, X, y, phase):
        """Evaluate model"""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        logger.info(f"{phase.upper()} Metrics: {metrics}")
        return metrics
    
    def _load_baseline(self):
        """Load baseline model metrics"""
        try:
            with open(self.models_dir / 'baseline_metrics.json', 'r') as f:
                return json.load(f)
        except:
            logger.warning("No baseline found, treating as first training")
            return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}
    
    def _is_improvement(self, new_metrics, baseline_metrics):
        """Check if new model improves over baseline"""
        improvement_threshold = self.config.get('improvement_threshold', 0.01)
        
        accuracy_improvement = new_metrics['accuracy'] - baseline_metrics['accuracy']
        
        is_better = accuracy_improvement > improvement_threshold
        
        if is_better:
            logger.info(f"Improvement: {accuracy_improvement:.4f}")
        else:
            logger.info(f"No sufficient improvement: {accuracy_improvement:.4f}")
        
        return is_better
    
    def _save_model(self, model, scaler, metrics):
        """Save model and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.models_dir / f"model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        
        scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Update baseline
        with open(self.models_dir / 'baseline_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Model saved: {model_path}")
    
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            logger.warning("Config file not found, using defaults")
            return {}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pipeline = RetrainingPipeline()
    success = pipeline.run()
    exit(0 if success else 1)
'''

with open('retrain_pipeline.py', 'w') as f:
    f.write(retrain_code)
print("✅ Saved: retrain_pipeline.py")

# ============================================================================
# PART 5: MLOPS BEST PRACTICES
# ============================================================================

print("\n" + "=" * 100)
print("5️⃣  MLOPS BEST PRACTICES")
print("=" * 100)

best_practices = """
✅ DO:
   1. Version everything (code, data, models)
   2. Automate testing and deployment
   3. Monitor model and data drift
   4. Track experiments and metrics
   5. Document decision making
   6. Implement feature flags
   7. Use containerization (Docker)
   8. Implement health checks
   9. Setup alerting and monitoring
   10. Plan for rollbacks

❌ DON'T:
   1. Manual deployments
   2. No version control
   3. No monitoring in production
   4. Ignore data quality
   5. Train on test set
   6. No logging
   7. Uncontrolled dependencies
   8. No documentation
   9. Skip integration tests
   10. Assume model won't drift

🎯 ML DEVELOPMENT LIFECYCLE:

1. PROBLEM DEFINITION
   • Understand business problem
   • Define success metrics
   • Data availability assessment

2. DATA ENGINEERING
   • Data collection
   • Data cleaning
   • Feature engineering
   • Data versioning

3. MODEL DEVELOPMENT
   • EDA and analysis
   • Feature selection
   • Model selection
   • Hyperparameter tuning
   • Experiment tracking

4. MODEL VALIDATION
   • Cross-validation
   • Test set evaluation
   • Error analysis
   • Bias/fairness checks

5. MODEL DEPLOYMENT
   • Containerization
   • API creation
   • Monitoring setup
   • Documentation

6. PRODUCTION MONITORING
   • Performance tracking
   • Data drift detection
   • Prediction drift detection
   • User feedback
   • Cost tracking

7. RETRAINING
   • Automated retraining
   • Performance comparison
   • Model promotion
   • Feedback incorporation

8. ITERATION
   • Identify improvements
   • Collect feedback
   • Start new cycle

📊 MLOPS METRICS:

Model Metrics:
  • Accuracy, Precision, Recall, F1
  • AUC-ROC, AUC-PR
  • Latency, Throughput
  • Feature importance

Data Metrics:
  • Data quality score
  • Missing values %
  • Outlier rate
  • Feature correlation

System Metrics:
  • API response time
  • Model inference time
  • CPU/Memory usage
  • Error rate
  • Uptime %

Business Metrics:
  • Revenue impact
  • Cost savings
  • Customer satisfaction
  • Model utilization

🔄 GITOPS FOR ML:

1. Single source of truth (Git)
2. Everything as code (configs, pipelines)
3. Automated deployment
4. Audit trail
5. Easy rollbacks

Example directory structure:
```
ml-project/
├── models/
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   └── current → model_v2.pkl (symlink)
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_manifest.json
├── pipelines/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── config/
│   ├── train.yaml
│   ├── deploy.yaml
│   └── monitor.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .github/workflows/
│   └── ml_pipeline.yml
└── docs/
    ├── model_card.md
    ├── deployment.md
    └── monitoring.md
```
"""

print(best_practices)

print("\n" + "=" * 100)
print("✅ MLOPS PIPELINE GUIDE COMPLETE")
print("=" * 100)
````

---

## 🎯 CREATE: End-to-End ML Projects

````python
# Complete End-to-End ML Projects

## Project 1: Customer Churn Prediction

### Problem Statement
Predict which customers will churn (leave) in the next 30 days

### Business Impact
- Identify at-risk customers
- Enable proactive retention campaigns
- Save revenue

### Data
- Customer demographics
- Account history
- Usage patterns
- Support interactions

### Approach

1. **EDA & Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Detect outliers
   - Check class balance

2. **Feature Engineering**
   - Days since last interaction
   - Account age
   - Usage trends
   - Support ticket patterns

3. **Model Selection**
   - Baseline: Logistic Regression
   - Candidate: XGBoost
   - Ensemble: VotingClassifier

4. **Evaluation**
   - Primary: ROC-AUC (focus on identifying churners)
   - Secondary: Precision (minimize false positives)
   - Business metric: Cost-benefit analysis

5. **Deployment**
   - REST API for real-time predictions
   - Batch predictions (daily)
   - Dashboard for retention teams

6. **Monitoring**
   - Churn rate tracking
   - Model prediction distribution
   - Retention campaign effectiveness

---

## Project 2: Time Series Forecasting - Sales Prediction

### Problem Statement
Forecast weekly sales for next 8 weeks

### Business Impact
- Inventory planning
- Revenue forecasting
- Resource allocation

### Data
- Historical sales data
- Promotions calendar
- Seasonal events
- Competitor information

### Approach

1. **Time Series Analysis**
   - Stationarity testing (ADF test)
   - Autocorrelation analysis (ACF/PACF)
   - Decomposition (trend, seasonality, residuals)

2. **Model Selection**
   - ARIMA: Baseline
   - Prophet: Handle seasonality and holidays
   - LSTM: For complex non-linear patterns

3. **Evaluation**
   - Train/validation/test splits (respect time order!)
   - Metrics: RMSE, MAPE, MAE
   - Backtesting: Walk-forward validation

4. **Deployment**
   - Weekly predictions
   - Confidence intervals
   - Anomaly detection (unusual sales)

5. **Monitoring**
   - Forecast accuracy tracking
   - Residual analysis
   - Seasonal pattern changes

---

## Project 3: Image Classification - Product Recognition

### Problem Statement
Classify product images for e-commerce platform

### Business Impact
- Automated tagging
- Better search results
- Improved user experience

### Data
- ~50K labeled product images
- 100+ product categories
- Various image sizes/qualities

### Approach

1. **Data Preparation**
   - Image preprocessing (resize, normalize)
   - Augmentation (rotation, flip, brightness)
   - Train/validation/test split

2. **Model Selection**
   - Transfer Learning: EfficientNet (pretrained)
   - Fine-tuning on product images
   - Custom head for 100 categories

3. **Training**
   - Learning rate scheduling
   - Batch normalization
   - Dropout regularization
   - Early stopping

4. **Evaluation**
   - Per-class accuracy
   - Confusion matrix
   - Top-5 accuracy
   - Error analysis

5. **Deployment**
   - TensorFlow Serving
   - GPU inference
   - Batch processing
   - Real-time API

6. **Monitoring**
   - Prediction confidence distribution
   - Image quality issues
   - New categories detection
   - Retraining schedule

---

## Project 4: NLP - Sentiment Analysis

### Problem Statement
Classify customer reviews as positive/negative

### Business Impact
- Monitor customer satisfaction
- Identify issues
- Track brand sentiment

### Data
- 100K labeled reviews
- Various lengths
- Informal language

### Approach

1. **Text Preprocessing**
   - Tokenization
   - Lowercasing
   - Remove special characters
   - Stemming/lemmatization

2. **Vectorization**
   - TF-IDF: Baseline
   - Word2Vec/GloVe: Dense embeddings
   - BERT: Contextual embeddings

3. **Model Selection**
   - Logistic Regression on TF-IDF
   - LSTM on embeddings
   - Fine-tuned BERT

4. **Evaluation**
   - Accuracy, Precision, Recall
   - Per-class metrics
   - Confusion matrix
   - Error analysis

5. **Deployment**
   - Batch processing for historical data
   - Real-time API for new reviews
   - Dashboard for sentiment trends

6. **Monitoring**
   - Sentiment distribution changes
   - Language evolution
   - New slang/terms
   - Model drift

---

## Project 5: Anomaly Detection - Fraud Detection

### Problem Statement
Detect fraudulent transactions in real-time

### Business Impact
- Prevent fraud
- Reduce chargebacks
- Protect customers

### Data
- Transaction history
- Highly imbalanced (0.1% fraud)
- Multiple features

### Approach

1. **Handling Imbalance**
   - SMOTE: Synthetic oversampling
   - Class weights
   - Threshold tuning

2. **Feature Engineering**
   - Transaction velocity
   - Merchant patterns
   - Device fingerprinting
   - Behavioral profiles

3. **Model Selection**
   - Isolation Forest: Unsupervised baseline
   - XGBoost with class weights
   - Ensemble of multiple models

4. **Evaluation**
   - Precision: Minimize false positives (customer friction)
   - Recall: Catch as much fraud as possible
   - ROC-AUC for threshold tuning
   - Cost-benefit analysis

5. **Deployment**
   - Real-time streaming predictions
   - Sub-100ms latency
   - Explainability (why flagged?)
   - A/B testing thresholds

6. **Monitoring**
   - False positive rate
   - False negative rate
   - Detection latency
   - Fraud patterns evolution

---

## Project Structure Template

```
my-ml-project/
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned data
│   └── external/         # External sources
│
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
│
├── models/
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   └── current → model_v2.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_selection.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── deploy_config.yaml
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── app.py
│   └── requirements.txt
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── retrain.py
│
├── monitoring/
│   ├── monitor.py
│   ├── alert_rules.yaml
│   └── dashboards.json
│
└── docs/
    ├── model_card.md
    ├── deployment.md
    ├── api.md
    └── monitoring.md
```

---

## Development Workflow

### 1. Setup
```bash
# Clone repository
git clone <repo>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install
```

### 2. Exploration
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/01_eda.ipynb
# - Load data
# - Exploratory analysis
# - Data quality assessment
```

### 3. Development
```bash
# Implement in src/ modules
# Write tests
# Run tests locally
pytest tests/ -v

# Check code quality
black src/
flake8 src/
```

### 4. Training & Evaluation
```bash
# Run training pipeline
python scripts/train.py --config configs/model_config.yaml

# Evaluate
python scripts/evaluate.py --model models/model_v1.pkl

# Compare metrics
python scripts/compare_models.py
```

### 5. Deployment
```bash
# Build Docker image
docker build -t my-model:latest .

# Run locally
docker run -p 5000:5000 my-model:latest

# Push and deploy
docker push my-model:latest
# Deploy using CI/CD
```

### 6. Monitoring
```bash
# Monitor in production
python monitoring/monitor.py

# Check dashboards
# Review alerts
# Plan retraining if needed
```

---

## Code Examples

### Complete Training Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import yaml

# Load config
with open('configs/model_config.yaml') as f:
    config = yaml.safe_load(f)

# Load and prepare data
df = pd.read_csv('data/raw/data.csv')

# Preprocessing
df = df.dropna()
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(**config['model_params'])
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, 'models/model_v1.pkl')
joblib.dump(scaler, 'models/scaler_v1.pkl')
"""

---

## Resources

### Learning
- Fast.ai: Practical deep learning
- Coursera ML Engineering
- StatQuest: Statistics explanations
- Papers with Code

### Tools
- Scikit-learn: Classical ML
- XGBoost/LightGBM: Gradient boosting
- PyTorch/TensorFlow: Deep learning
- Spark: Large-scale processing
- Airflow: Workflow orchestration
- Kubernetes: Container orchestration

### Deployment
- Docker: Containerization
- Kubernetes: Orchestration
- Cloud: AWS/GCP/Azure
- MLflow: Model registry
- Seldon: Model serving

### Monitoring
- Prometheus: Metrics
- Grafana: Dashboards
- ELK: Logs
- DataDog: APM

````

---

## 📋 CREATE: Complete Project Checklist

````python
# Complete ML Project Checklist

## Phase 1: Problem Definition & Planning

### Business Understanding
- [ ] Clear problem statement written
- [ ] Business success metrics defined
- [ ] Stakeholders identified
- [ ] Timeline and resources allocated
- [ ] Risk assessment completed
- [ ] ROI analysis done
- [ ] Constraints identified (latency, memory, cost)

### Data Assessment
- [ ] Data sources identified
- [ ] Data availability confirmed
- [ ] Data quality assessment done
- [ ] Privacy/compliance review
- [ ] Labeling strategy defined
- [ ] Data volume estimated
- [ ] Baseline established

### Project Setup
- [ ] Git repository created
- [ ] Development environment setup
- [ ] Documentation template created
- [ ] Team roles assigned
- [ ] Communication channels set
- [ ] Code review process defined
- [ ] Version control strategy

---

## Phase 2: Data Engineering & EDA

### Data Collection & Preparation
- [ ] Raw data loaded
- [ ] Data profiling completed
- [ ] Data quality report
- [ ] Missing value analysis
- [ ] Duplicates identified/removed
- [ ] Data errors fixed
- [ ] Data dictionary created
- [ ] Data sample verified

### Exploratory Analysis
- [ ] Distributions analyzed
- [ ] Correlations examined
- [ ] Outliers identified
- [ ] Class balance checked
- [ ] Feature relationships explored
- [ ] Statistical tests performed
- [ ] Visualizations created
- [ ] Key insights documented

### Feature Engineering
- [ ] Domain features created
- [ ] Categorical encoding done
- [ ] Numerical scaling applied
- [ ] Feature interactions added
- [ ] Feature selection performed
- [ ] Feature importance calculated
- [ ] Multicollinearity checked
- [ ] Feature validation completed

### Data Versioning
- [ ] Data hash/version recorded
- [ ] Processing pipeline documented
- [ ] Preprocessing saved
- [ ] Train/val/test split saved
- [ ] Data manifest created

---

## Phase 3: Model Development

### Model Selection
- [ ] Problem type clarified (classification/regression)
- [ ] Multiple baselines tried
- [ ] Model candidates selected
- [ ] Algorithm complexity analyzed
- [ ] Computational requirements estimated
- [ ] Interpretability needs addressed
- [ ] Model limitations documented

### Model Training
- [ ] Training pipeline created
- [ ] Cross-validation setup (5-fold minimum)
- [ ] Hyperparameter ranges defined
- [ ] Hyperparameter tuning performed
- [ ] Learning curves analyzed
- [ ] Convergence verified
- [ ] Training metrics logged
- [ ] Model artifacts saved

### Model Evaluation
- [ ] All metrics calculated
- [ ] Confusion matrix created
- [ ] Error analysis performed
- [ ] Per-class metrics evaluated
- [ ] Threshold optimization done
- [ ] Comparison with baseline
- [ ] Statistical significance tested
- [ ] Evaluation report created

### Model Validation
- [ ] Test set evaluation
- [ ] Out-of-sample performance
- [ ] Cross-validation stability
- [ ] Reproducibility verified
- [ ] Fairness analysis done
- [ ] Bias detection completed
- [ ] Edge cases tested
- [ ] Model assumptions validated

---

## Phase 4: Model Deployment

### Containerization & Packaging
- [ ] Model serialized (joblib/pickle)
- [ ] Scaler/preprocessor saved
- [ ] Dependencies documented (requirements.txt)
- [ ] Dockerfile created
- [ ] Docker image built and tested
- [ ] Docker compose setup (if needed)
- [ ] Model artifacts versioned
- [ ] Metadata file created

### API Development
- [ ] Flask/FastAPI app created
- [ ] Input validation implemented
- [ ] Error handling implemented
- [ ] Logging implemented
- [ ] API documentation (Swagger)
- [ ] Health check endpoint
- [ ] Request/response schemas
- [ ] Rate limiting configured

### Testing
- [ ] Unit tests written (>80% coverage)
- [ ] Integration tests written
- [ ] API tests written
- [ ] Load testing performed
- [ ] Edge case testing
- [ ] Security testing
- [ ] All tests passing
- [ ] CI/CD pipeline setup

### Deployment Strategy
- [ ] Canary deployment plan
- [ ] Rollback strategy
- [ ] Feature flags implemented
- [ ] A/B testing setup
- [ ] Monitoring alerts configured
- [ ] Deployment checklist
- [ ] Go/no-go criteria
- [ ] Stakeholder sign-off

---

## Phase 5: Production Monitoring

### System Monitoring
- [ ] API response time monitored
- [ ] CPU/memory usage tracked
- [ ] Error rates monitored
- [ ] Request volume tracked
- [ ] Health checks configured
- [ ] Uptime SLA defined
- [ ] Alerting rules set
- [ ] Dashboard created

### Model Monitoring
- [ ] Prediction accuracy tracked
- [ ] Metric thresholds set
- [ ] Data drift detection
- [ ] Prediction drift detection
- [ ] Feature importance tracked
- [ ] Confidence distribution monitored
- [ ] Error patterns analyzed
- [ ] Alert triggers configured

### Data Monitoring
- [ ] Input distribution tracked
- [ ] Missing values monitored
- [ ] Outlier detection
- [ ] Data quality metrics
- [ ] Schema validation
- [ ] Anomaly alerts
- [ ] Data freshness checks

### Business Monitoring
- [ ] Business metrics tracked
- [ ] ROI calculated
- [ ] User feedback collected
- [ ] Model usage tracked
- [ ] Cost analysis
- [ ] Performance vs baseline
- [ ] Customer impact measured

---

## Phase 6: Continuous Improvement

### Feedback Collection
- [ ] User feedback mechanism
- [ ] Prediction correctness tracking
- [ ] Error patterns documented
- [ ] Improvement suggestions
- [ ] Feedback loop automation

### Retraining Strategy
- [ ] Trigger criteria defined
- [ ] Retraining schedule set
- [ ] Data collection pipeline
- [ ] Automated retraining
- [ ] Model comparison
- [ ] Promotion strategy
- [ ] Version management

### Experimentation
- [ ] New feature ideas tracked
- [ ] Experiment design template
- [ ] A/B testing framework
- [ ] Statistical testing rigor
- [ ] Results documentation
- [ ] Learning captured

### Iteration Planning
- [ ] Roadmap created
- [ ] Prioritization framework
- [ ] Resource allocation
- [ ] Sprint planning
- [ ] Stakeholder updates
- [ ] Success metrics review

---

## Phase 7: Documentation & Knowledge

### Model Documentation
- [ ] Model card created
- [ ] Assumptions documented
- [ ] Limitations documented
- [ ] Training data documented
- [ ] Feature descriptions
- [ ] Performance benchmarks
- [ ] Use cases and don'ts
- [ ] Ethical considerations

### Deployment Documentation
- [ ] API documentation
- [ ] Installation guide
- [ ] Configuration guide
- [ ] Troubleshooting guide
- [ ] Performance tuning guide
- [ ] Monitoring guide
- [ ] Rollback procedure

### Code Documentation
- [ ] Code commented
- [ ] Functions documented
- [ ] README complete
- [ ] Examples provided
- [ ] Dependencies listed
- [ ] Architecture diagram
- [ ] Data flow diagram

### Knowledge Base
- [ ] Lessons learned documented
- [ ] Common issues tracked
- [ ] FAQ created
- [ ] Team training completed
- [ ] Runbook created
- [ ] Escalation procedures

---

## Success Criteria

- ✅ Model meets business requirements
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Monitoring in place
- ✅ Team trained
- ✅ Stakeholders satisfied
- ✅ Performance baseline established
- ✅ Retraining strategy defined
- ✅ Rollback plan ready
- ✅ Production deployment successful

````

---

## ✅ Summary

I've created **3 comprehensive production guides**:

### 1. **Model Deployment & Production** (`model_deployment_guide.py`)
   - Model serialization (joblib, pickle, ONNX)
   - Flask API for model serving
   - Docker containerization
   - API client & testing
   - Monitoring & logging
   - Complete deployment checklist

### 2. **MLOps Pipeline & CI/CD** (`mlops_pipeline_guide.py`)
   - Experiment tracking (MLflow)
   - Model versioning & registry
   - GitHub Actions CI/CD pipeline
   - Automated retraining pipeline
   - MLOps best practices
   - ML development lifecycle

### 3. **End-to-End Projects** (`end_to_end_projects.md`)
   - 5 complete project examples:
     - Customer churn prediction
     - Time series forecasting
     - Image classification
     - NLP sentiment analysis
     - Anomaly detection (fraud)
   - Project structure template
   - Development workflow
   - Complete checklist

Plus **Project Checklist** covering all 7 phases from planning to production!

---

## 🎉 COMPLETE SUITE SUMMARY

You now have **COMPREHENSIVE PRODUCTION-READY GUIDES** covering:

### **Part 1-2: Foundation**
- ✅ Complete ML algorithm selection guide
- ✅ Implementation checklist

### **Part 3-4: Advanced Techniques**
- ✅ Hyperparameter tuning (4 methods)
- ✅ Model evaluation metrics
- ✅ Feature engineering cookbook
- ✅ Error analysis & debugging

### **Part 5-8: Production**
- ✅ Model deployment & APIs
- ✅ MLOps & CI/CD pipelines
- ✅ End-to-end projects
- ✅ Complete project checklist

---

## 📚 **Total Coverage**:
- **8 major guides**
- **30+ code examples**
- **100+ best practices**
- **Ready for production deployment**

Would you like me to create:
- ✅ **Advanced Topics**: Explainability, Fairness, Optimization
- ✅ **Specific Use Cases**: Healthcare, Finance, E-commerce
- ✅ **Interactive Dashboard**: Monitoring templates
- ✅ **Video Walkthroughs**: Step-by-step tutorials

What's your priority? 🚀
