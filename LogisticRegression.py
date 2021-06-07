#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('ggplot')
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 12,8
# %%
data = pd.read_csv(r'C:\Users\Kaushik\Desktop\MLProjects\LogisticRegression\DMV_Written_Tests.csv')
data.head()
data.info()
# %%
scores = data[['DMV_Test_1','DMV_Test_2']].values
results = data['Results'].values
# %%
passed = (results == 1).reshape(100, 1)
failed = (results == 0).reshape(100, 1)
ax = sns.scatterplot(x = scores[passed[:,0],0], 
                    y = scores[passed[:,0],1],
                    marker = '^', color = 'green',s=60)
sns.scatterplot(x= scores[failed[:,0],0], 
                y = scores[failed[:,0],1],
                marker = 'X', color = 'red',s=60)
ax.set(xlabel='DMV Written Test 1 Scores',
        ylabel = 'DMV Written Test 2 Scores')
ax.legend(['Passed','Failed'])
# %%
def logistic_function(x):
    return 1/(1+np.exp(-x))
logistic_function(0)
# %%
def compute_cost(theta,x,y):
    m = len(y)
    y_pred = logistic_function(np.dot(x,theta))
    error = (y*np.log(y_pred)) + ((1-y)) * np.log(1-y_pred)
    cost = -1/m * sum(error)
    gradient = 1/m * np.dot(x.transpose(),(y_pred-y))
    return cost[0],gradient
# %%
mean_scores  = np.mean(scores,axis=0)
std_scores = np.std(scores,axis=0)
scores = (scores-mean_scores)/std_scores
rows = scores.shape[0]
cols = scores.shape[1]

X = np.append(np.ones((rows,1)),scores,axis=1)
y = results.reshape(rows,1)

theta_init = np.zeros((cols+1,1))
cost,gradient = compute_cost(theta_init,X,y)
print(cost)
print(gradient)
# %%
def gradient_descent(x,y,theta,alpha,iterations):
    costs = []
    for i in range(iterations):
        cost,gradient = compute_cost(theta,x,y)
        theta -= (alpha*gradient)
        costs.append(cost)
    return theta,costs
# %%
theta,costs = gradient_descent(X,y,theta_init,1,250)
print(theta)
print(costs[-1])
# %%
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('$J(\Theta)$')
plt.title('Values of cost Function Over Iterations of Gradient Descent')
# %%
sns.scatterplot(x = X[passed[:,0],1],
                y = X[passed[:,0],2],
                marker= '^',color = 'green',s=60)
ax = sns.scatterplot(x = X[failed[:,0],1],
                    y = X[failed[:,0],2], marker='X',color='red',s=60)
ax.legend(['Passed','Failed'])
ax.set(xlabel = 'DMV Written Tests 1 scores',
        ylabel = 'DMV Witten Test 2 Scores')
x_boundary = np.array([np.min(X[:,-1]), np.max(X[:,1])])
y_boundary = -(theta[0] + theta[1] * x_boundary)/theta[2]
sns.lineplot(x= x_boundary,y=y_boundary,color='blue')
plt.show()
# %%
def predict(theta,x):
    results = x.dot(theta)
    return results > 0

p = predict(theta,X)
print(sum(p==y)[0],"%")
# %%
test = np.array([50,79])
test = (test-mean_scores)/std_scores
test = np.append(np.ones(1),test)
probability = logistic_function(test.dot(theta))
print(np.round(probability[0],2))
# %%
