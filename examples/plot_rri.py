"""
=================================
Plotting ReflectiveRandomIndexing
=================================

An example plot of :class:`sklearn_rri.rri.ReflectiveRandomIndexing`
"""
from sklearn.random_projection import gaussian_random_matrix
from sklearn_rri import ReflectiveRandomIndexing
from matplotlib import pyplot as plt

X = gaussian_random_matrix(100, 2, random_state=42)
X.sort(0)
estimator = ReflectiveRandomIndexing(random_state=42)
X_transformed = estimator.fit_transform(X)
X_inverted = estimator.inverse_transform(X_transformed)

plt.plot(X.flatten(), alpha=0.5, label='Original Data')
plt.plot(X_transformed.flatten(), alpha=0.5, label='Transformed Data')
plt.plot(X_inverted.flatten(), alpha=0.5, label='Inverse Transformed Data')
plt.title('Plots of original and transformed data')

plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Value of Data')

plt.show()
