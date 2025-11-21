"""Multiple Linear Regression from Scratch

This module implements Multiple Linear Regression using NumPy and Gradient Descent.
Predicts a continuous target variable using multiple features.

Author: Rohit
Date: November 21, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional


class MultipleLinearRegression:
    """Multiple Linear Regression Model using Gradient Descent
    
    Implements: y = X @ weights + bias
    where X has multiple features
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: Optional[str] = None, lambda_param: float = 0.01):
        """Initialize the model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
            regularization: Type of regularization ('l1', 'l2', or None)
            lambda_param: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.feature_means = None
        self.feature_stds = None
    
    def _normalize_features(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Normalize features using z-score normalization.
        
        Args:
            X: Input features
            training: If True, compute and store mean/std. If False, use stored values.
            
        Returns:
            Normalized features
        """
        if training:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the cost function (MSE with optional regularization).
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Cost value
        """
        m = len(y)
        predictions = self.predict(X, normalize=False)
        
        # Mean Squared Error
        mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        
        # Add regularization term
        if self.regularization == 'l2':
            mse += (self.lambda_param / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            mse += (self.lambda_param / m) * np.sum(np.abs(self.weights))
        
        return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Train the model using Gradient Descent.
        
        Args:
            X: Training features (m samples, n features)
            y: Training targets (m samples)
            verbose: Print training progress
        """
        # Normalize features
        X_normalized = self._normalize_features(X, training=True)
        
        # Initialize parameters
        m, n = X_normalized.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient Descent
        for iteration in range(self.n_iterations):
            # Forward pass
            predictions = X_normalized @ self.weights + self.bias
            
            # Compute gradients
            error = predictions - y
            dw = (1 / m) * (X_normalized.T @ error)
            db = (1 / m) * np.sum(error)
            
            # Add regularization gradient
            if self.regularization == 'l2':
                dw += (self.lambda_param / m) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / m) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Record cost
            cost = self._compute_cost(X_normalized, y)
            self.cost_history.append(cost)
            
            # Print progress
            if verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration:4d}: Cost = {cost:.4f}")
        
        print(f"\nTraining Complete!")
        print(f"Final Cost: {self.cost_history[-1]:.4f}")
        print(f"Learned {len(self.weights)} weights + 1 bias term")
    
    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            normalize: Whether to normalize features
            
        Returns:
            Predicted values
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions!")
        
        if normalize:
            X_normalized = self._normalize_features(X, training=False)
        else:
            X_normalized = X
        
        return X_normalized @ self.weights + self.bias
    
    def calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Calculate performance metrics.
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Get feature importance based on absolute weight values.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.weights is None:
            raise ValueError("Model must be trained first!")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.weights))]
        
        importance = np.abs(self.weights)
        importance_pct = 100 * importance / np.sum(importance)
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': self.weights,
            'Importance': importance,
            'Importance_%': importance_pct
        })
        
        return df.sort_values('Importance', ascending=False)
    
    def plot_cost_history(self) -> None:
        """Visualize the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, linewidth=2, color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Cost Function Over Training Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, X: np.ndarray, y: np.ndarray, title: str = "Predictions vs Actual") -> None:
        """Plot predicted values vs actual values.
        
        Args:
            X: Features
            y: True values
            title: Plot title
        """
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def generate_sample_data(n_samples: int = 200, n_features: int = 3, noise: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Amount of random noise
        
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features) * 10
    
    # True weights
    true_weights = np.array([2.5, -1.5, 3.0])[:n_features]
    true_bias = 10
    
    # Generate target with noise
    y = X @ true_weights + true_bias + np.random.normal(0, noise, n_samples)
    
    return X, y


def main():
    """Main function to demonstrate Multiple Linear Regression."""
    print("="*60)
    print("Multiple Linear Regression - From Scratch")
    print("="*60)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=200, n_features=3, noise=10.0)
    feature_names = ['Size (sqft)', 'Bedrooms', 'Age (years)']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print()
    
    # Train model
    print("Training the model...")
    print("-" * 60)
    model = MultipleLinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        lambda_param=0.01
    )
    model.fit(X_train, y_train, verbose=True)
    print("-" * 60)
    print()
    
    # Evaluate
    print("Training Set Performance:")
    train_metrics = model.calculate_metrics(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    print("Test Set Performance:")
    test_metrics = model.calculate_metrics(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Feature importance
    print("Feature Importance:")
    print("-" * 60)
    importance_df = model.get_feature_importance(feature_names)
    print(importance_df.to_string(index=False))
    print("-" * 60)
    print()
    
    # Sample predictions
    print("Sample Predictions:")
    print("-" * 60)
    sample_X = np.array([
        [150, 3, 5],   # 150 sqft, 3 bedrooms, 5 years old
        [200, 4, 2],   # 200 sqft, 4 bedrooms, 2 years old
        [100, 2, 10]   # 100 sqft, 2 bedrooms, 10 years old
    ])
    predictions = model.predict(sample_X)
    for i, (features, pred) in enumerate(zip(sample_X, predictions), 1):
        print(f"Sample {i}: {features} -> Predicted: {pred:.2f}")
    print("-" * 60)
    print()
    
    # Visualizations
    print("Generating visualizations...")
    model.plot_cost_history()
    model.plot_predictions_vs_actual(X_test, y_test, "Test Set: Predictions vs Actual")
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
