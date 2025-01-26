Unit 8: Gradient Cost Function

In the following Python code I attempted an implementation of gradient descent for fitting a linear regression model to a dataset.

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0  # Initialize slope and intercept
    iterations = 100  # Number of updates
    n = len(x)  # Number of data points
    learning_rate = 0.08  # Step size for parameter updates
    
    # To store historical data for visualization
    m_history = []
    b_history = []
    cost_history = []
    
    for i in range(iterations):
        # Calculate predicted values based on current slope and intercept
        y_predicted = m_curr * x + b_curr
        
        # Compute the cost function (mean squared error)
        cost = (1/n) * sum((y - y_predicted) ** 2)
        
        # Calculate gradients for slope (m) and intercept (b)
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        # Update parameters using the gradients
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        # Save progress for later visualization
        m_history.append(m_curr)
        b_history.append(b_curr)
        cost_history.append(cost)
    
    return m_curr, b_curr, m_history, b_history, cost_history

# Input data: x (independent variable) and y (dependent variable)
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Perform gradient descent
m_final, b_final, m_history, b_history, cost_history = gradient_descent(x, y)

# Visualization
plt.figure(figsize=(14, 5))

# Plot the data points and fitted line
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m_final * x + b_final, color='red', label=f'Fitted line: y = {m_final:.2f}x + {b_final:.2f}')
plt.title('Data and Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot cost function progression
plt.subplot(1, 2, 2)
plt.plot(range(len(cost_history)), cost_history, color='green')
plt.title('Cost Function over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.tight_layout()
plt.show()


In this exercise I tried to understand the effects of changing parameter values during a gradient descent optimization process. The goal here was to fit a linear model to a set of 2D data points by minimizing a cost function using gradient descent.

The number of iterations determines how often the parameters (slope and intercept) are updated. More iterations provide additional opportunities for convergence to the global minimum of the cost function. However if there are too few iterations, the model can not capture the underlying data structure which can cause underfitting. On the other hand, too many iterations can result in unnecessary computations, especially when the parameters have already converged.

The learning rate parameter sets the step size during each update of the parameters. A Small learning rate results in gradual updates which provides a stable but slow convergence. Large learning rates speed up convergence by taking larger steps but this can lead to overshooting the minimum, which can potentially cause divergence or oscillations.

