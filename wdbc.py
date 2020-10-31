#%%
import pandas as pd
import numpy as np

#%%
names = ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]
variants = ["mean", "se", "worst"]
names = [x + " " + y for y in variants for x in names ]
names = ["id", "class"] + names
names

#%%
df = pd.read_csv(r"data\wdbc.data", names=names)
df

#%%
np.unique(df["class"].values, return_counts=True)

#%%
#X = df[["texture mean", "concave points mean", "radius worst", "smoothness worst"]]
X = df.iloc[:, 2:]
#X = df.loc[:, df.columns != "class"]
y = df["class"]
y[y=="B"] = False
y[y=="M"] = True
y = y.astype("int")

X, y

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, stratify=y)
X_train.shape

#%%
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(500, 500, 500), learning_rate_init=1e-2, tol=1e-6, verbose=True))
    ],
    verbose=True
)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
# %%
