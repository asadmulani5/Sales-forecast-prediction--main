#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("/Users/anchalbhondekar/Desktop/project/_data_.csv")


# In[3]:


df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day


# In[4]:


df = pd.get_dummies(df, columns=['Category'])


# In[5]:


X = df[['year', 'month', 'day'] + [col for col in df.columns if col.startswith('Category_')]].values
y = df['Quantity'].values


# In[6]:


print(df.head())


# In[58]:


class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        best_split = self._find_best_split(X, y)
        if best_split['gain'] == 0:
            return np.mean(y)

        left_tree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)

        return {
            'split_feature': best_split['split_feature'],
            'split_value': best_split['split_value'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        best_split = {'gain': 0}
        m, n = X.shape
        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = self._calculate_gain(y, left_indices, right_indices)
                if gain > best_split['gain']:
                    best_split = {
                        'split_feature': feature,
                        'split_value': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'gain': gain
                    }
        return best_split

    def _calculate_gain(self, y, left_indices, right_indices):
        left_y = y[left_indices]
        right_y = y[right_indices]
        p = len(left_y) / len(y)
        gain = self._variance(y) - p * self._variance(left_y) - (1 - p) * self._variance(right_y)
        return gain

    def _variance(self, y):
        return np.var(y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['split_feature']] <= tree['split_value']:
                return self._predict_one(x, tree['left'])
            else:
                return self._predict_one(x, tree['right'])
        else:
            return tree
  


# In[59]:


class XGBoostRegressor:
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=float)  # Ensure y_pred is float
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=float)  # Ensure y_pred is float
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# In[60]:


model = XGBoostRegressor(n_estimators=10, max_depth=3, learning_rate=0.1)
model.fit(X, y)


# In[61]:


predictions = model.predict(X)
print(predictions)


# In[62]:


predictions = np.round(predictions).astype(int)
print(predictions)


# In[63]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[64]:


mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[66]:


plt.figure(figsize=(12, 6))
df_yearly = df.groupby('year')['Quantity'].sum().reset_index()
sns.barplot(data=df_yearly, x='year', y='Quantity')
plt.title('Total Sales of All Categories by Year')
plt.show()


# In[67]:


plt.figure(figsize=(12, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()


# In[68]:


residuals = y - predictions
plt.figure(figsize=(12, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()


# In[69]:


metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}
plt.figure(figsize=(12, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Performance Metrics')
plt.show()


# In[70]:


plt.figure(figsize=(12, 6))
sns.histplot(predictions, kde=True)
plt.title('Distribution of Predicted Values')
plt.xlabel('Predicted Quantity')
plt.ylabel('Frequency')
plt.show()


# In[71]:


df['Predicted'] = predictions
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values(by='Date')
plt.plot(df_sorted['Date'], df_sorted['Quantity'], label='Actual', alpha=0.75)
plt.plot(df_sorted['Date'], df_sorted['Predicted'], label='Predicted', alpha=0.75)
plt.legend()
plt.title('Time Series of Actual vs Predicted Quantities')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.show()


# In[72]:


df_sorted['Cumulative_Actual'] = df_sorted['Quantity'].cumsum()
df_sorted['Cumulative_Predicted'] = df_sorted['Predicted'].cumsum()
plt.figure(figsize=(12, 6))
plt.plot(df_sorted['Date'], df_sorted['Cumulative_Actual'], label='Actual Cumulative Sales')
plt.plot(df_sorted['Date'], df_sorted['Cumulative_Predicted'], label='Predicted Cumulative Sales')
plt.legend()
plt.title('Cumulative Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.show()


# In[73]:


df_pivot = df.pivot_table(values='Quantity', index='year', columns='month', aggfunc='sum')
plt.figure(figsize=(12, 6))
sns.heatmap(df_pivot, annot=True, fmt=".1f", cmap='coolwarm')
plt.title('Heatmap of Sales by Year and Month')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


# In[74]:


plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[76]:


import pickle


# In[77]:


with open('_model_.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




