import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42) #to ensure the reproducbility 
n_sample = 100

X = np.random.rand(n_sample,1)

true_m = 5.00
true_b = 3.00

noise = np.random.randn(n_sample,1)*0.5

Y = true_m * X+ true_b
# print(Y)

X_bias = np.c_[np.ones((n_sample,1)),X]

# print(X_bias)


# plt.scatter(X,Y, alpha = 0.5)
# plt.title("synthetic data")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

theta_al = np.linalg.inv(X_bias.T@X_bias)@(X_bias.T@Y)
print(theta_al)

# hyperparameter 
lr = 0.01
epochs = 100

theta_gd  = np.random.rand(2,1)
print(theta_gd)

losses =[]

for epoch in range(epochs):
    y_pred =  X_bias@ theta_gd
    error  = y_pred-Y
    loss = (error**2).mean()
    losses.append(loss)

    gradient = (2/n_sample)*(X_bias.T@error)
    theta_gd = -lr*(gradient)

print(f"GD m : m is {theta_gd[1][0]} and b: is {theta_gd[0][0]}")

plt.scatter(X,Y , label = "Data",color = 'b', alpha = 0.6)
plt.plot(X,X_bias@theta_al, color ='y', linestyle ='-.',label = 'analytic')
plt.plot(X,X_bias@theta_gd, color ='g',linestyle = '--', label ='GD')
plt.legend()
plt.show()