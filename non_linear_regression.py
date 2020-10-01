import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Linear
x = np.arange(-5.0, 5.0, 0.1)
y = 2 * (x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise

plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Non-Linear
x = np.arange(-5.0, 5.0, 0.1)
y = (x**3) + (x**2) + x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise

plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Quadratic
x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise

plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Exponential
X = np.arange(-5.0, 5.0, 0.1)
Y= np.exp(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Logarithmic

X = np.arange(-5.0, 5.0, 0.1)
Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Logistic/Sigmoid
X = np.arange(-5.0, 5.0, 0.1)
Y = 1 - 4 / (1 + np.power(3, X-2))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Getting the data
df = pd.read_csv("csv_dataframes/ChinaGDP.csv")

x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Building the model
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(- Beta_1 * (x - Beta_2)))
     return y

# Viewing a simple sigmoid curve and comparing to the data
beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1 , beta_2)

plt.plot(x_data, Y_pred * 15000000000000)
plt.plot(x_data, y_data, 'ro')

# Normalizing the data
x_data_norm = x_data / max(x_data)
y_data_norm = y_data / max(y_data)

# Training the model
popt, pcov = curve_fit(sigmoid, x_data_norm, y_data_norm)

print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1])) # popt are the final parameters

# Now we plot our resulting regression model.

x = np.linspace(1960, 2015, 55)
x = x / max(x)

y = sigmoid(x, *popt)

plt.plot(x_data_norm, y_data_norm, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')

plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
