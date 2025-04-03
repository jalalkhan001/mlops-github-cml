# This code creates a sample dataset using the Iris dataset from sklearn and saves it as a CSV file.

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('iris.csv', index=False)
print("Sample data created and saved to data/iris.csv")

