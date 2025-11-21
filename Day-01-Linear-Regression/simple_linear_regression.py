"""Simple Linear Regression from Scratch

This module implements Simple Linear Regression using only NumPy.
Predicts a continuous target variable using a single feature.

Author: Rohit
Date: November 21, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SimpleLinearRegression:
    """Simple Linear Regression Model
    
    Implements y = mx + b where:
    - m is the slope (weight)
    - b is the y-intercept (bias)
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.training_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using the closed-form solution.
        
        Args:
            X: Training features (1D array)
            y: Training targets (1D array)
        """
        # Calculate means
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (m)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.slope = numerator / denominator
        
        # Calculate intercept (b)
        self.intercept = y_mean - self.slope * X_mean
        
        print(f"Training Complete!")
        print(f"Slope (m): {self.slope:.4f}")
        print(f"Intercept (b): {self.intercept:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be trained before making predictions!")
        
        return self.slope * X + self.intercept
    
    def calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Calculate performance metrics.
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Dictionary containing MSE, RMSE, MAE, and R²
        """
        y_pred = self.predict(X)
        
        # Mean Squared Error
        mse = np.mean((y - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y - y_pred))
        
        # R² Score (Coefficient of Determination)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    def plot_regression_line(self, X: np.ndarray, y: np.ndarray, title: str = "Linear Regression") -> None:
        """Visualize the regression line with data points.
        
        Args:
            X: Features
            y: True values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data', s=50)
        
        # Plot regression line
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        
        # Labels and title
        plt.xlabel('X (Feature)', fontsize=12)
        plt.ylabel('y (Target)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add equation to plot
        equation = f'y = {self.slope:.2f}x + {self.intercept:.2f}'
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def generate_sample_data(n_samples: int = 100, noise: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of random noise to add
        
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(42)
    
    # Generate X values
    X = np.linspace(0, 100, n_samples)
    
    # True relationship: y = 2.5x + 15
    true_slope = 2.5
    true_intercept = 15
    
    # Add noise
    y = true_slope * X + true_intercept + np.random.normal(0, noise, n_samples)
    
    return X, y


def main():
    """Main function to demonstrate Simple Linear Regression."""
    print("="*60)
    print("Simple Linear Regression - From Scratch")
    print("="*60)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=100, noise=10.0)
    print(f"Dataset size: {len(X)} samples")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Split data into train and test sets (80-20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Create and train the model
    print("Training the model...")
    print("-" * 60)
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    print("-" * 60)
    print()
    
    # Evaluate on training data
    print("Training Set Performance:")
    train_metrics = model.calculate_metrics(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Evaluate on test data
    print("Test Set Performance:")
    test_metrics = model.calculate_metrics(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Make sample predictions
    print("Sample Predictions:")
    print("-" * 60)
    sample_X = np.array([10, 25, 50, 75, 90])
    predictions = model.predict(sample_X)
    for x_val, pred in zip(sample_X, predictions):
        print(f"X = {x_val:5.1f} -> Predicted y = {pred:7.2f}")
    print("-" * 60)
    print()
    
    # Visualize results
    print("Generating visualization...")
    model.plot_regression_line(X_train, y_train, "Simple Linear Regression - Training Data")
    
    print("\n" + "="*60)
    print("Model training and evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
