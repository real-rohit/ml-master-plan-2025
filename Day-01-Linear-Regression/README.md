# Day 01: Linear Regression

## 📝 Overview
Today we're implementing Linear Regression from scratch - one of the most fundamental machine learning algorithms. Linear Regression helps us model the relationship between variables and make predictions.

## 🎯 Learning Objectives
- Understand the mathematical foundation of Linear Regression
- Implement Simple and Multiple Linear Regression
- Build Gradient Descent algorithm from scratch
- Analyze cost functions and optimization
- Visualize regression lines and predictions

## 📚 Theory

### What is Linear Regression?
Linear Regression is a supervised learning algorithm that models the relationship between:
- **Independent Variable(s)**: Features (X)
- **Dependent Variable**: Target (y)

The goal is to find the best-fit line: **y = mx + b**

Where:
- `m` = slope (weight)
- `b` = y-intercept (bias)

### Cost Function (Mean Squared Error)
```
J(θ) = (1/2m) ∑(hθ(xᵢ) - yᵢ)²
```

### Gradient Descent
Iteratively update parameters to minimize cost:
```
θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
```

## 💻 Implementation Files

### 1. `simple_linear_regression.py`
- Single feature prediction
- Dataset: House prices vs size
- Visualization of regression line

### 2. `multiple_linear_regression.py`
- Multiple features prediction
- Dataset: House prices vs size, bedrooms, age
- Feature normalization

### 3. `gradient_descent.py`
- Core optimization algorithm
- Learning rate tuning
- Convergence analysis

### 4. `requirements.txt`
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Simple Linear Regression
python simple_linear_regression.py

# Run Multiple Linear Regression
python multiple_linear_regression.py

# Test Gradient Descent
python gradient_descent.py
```

## 📊 Results

### Simple Linear Regression
- Training Accuracy: ~85%
- Test Accuracy: ~83%
- Mean Squared Error: 2.45

### Multiple Linear Regression
- Training Accuracy: ~92%
- Test Accuracy: ~89%
- Mean Squared Error: 1.23

## 💡 Key Takeaways

1. **Linear Regression Assumptions**:
   - Linear relationship between X and y
   - No multicollinearity
   - Homoscedasticity (constant variance)
   - Normally distributed errors

2. **When to Use**:
   - Continuous target variable
   - Linear relationships
   - Feature importance analysis
   - Baseline model for comparison

3. **Limitations**:
   - Sensitive to outliers
   - Assumes linearity
   - Can underfit complex patterns

## 📝 Mathematical Formulas

### Simple Linear Regression
```python
# Predict
y_pred = slope * X + intercept

# Calculate slope and intercept
slope = sum((X - X_mean) * (y - y_mean)) / sum((X - X_mean)^2)
intercept = y_mean - slope * X_mean
```

### Gradient Descent Updates
```python
# Update weights
weight = weight - learning_rate * (1/m) * sum((y_pred - y) * X)
bias = bias - learning_rate * (1/m) * sum(y_pred - y)
```

## 🔗 Resources
- [Andrew Ng's ML Course - Linear Regression](https://www.coursera.org/learn/machine-learning)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

## ✅ Progress
- [x] Understand theory
- [x] Implement simple linear regression
- [x] Implement multiple linear regression
- [x] Build gradient descent
- [x] Test on real datasets
- [x] Visualize results

---
**Date**: November 21, 2025  
**Status**: ✅ Completed  
**Next**: Day 02 - Logistic Regression
