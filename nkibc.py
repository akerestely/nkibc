#%%
import pandas as pd
import numpy as np

#%%
df = pd.read_csv("data/deviramanan2016-nki-breast-cancer-data/NKI_cleaned.csv")
df.head()

#%%
df.shape

#%%
print(df.columns)

# %%
df.info()

# %%
X = df.drop(["Patient", "ID", "eventdeath"], axis=1)
y = df['eventdeath']

display(X.shape)
display(y.shape)

#%%
import matplotlib.pyplot as plt
from yellowbrick.target import ClassBalance

_, y_counts = np.unique(y, return_counts=True)
class_labels = ["survived", "deceased"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4.5))
ax1.pie(y_counts, explode=(0, 0.05), labels = class_labels)

visualizer = ClassBalance(labels = class_labels, ax = ax2)
visualizer.fit(y)
visualizer.finalize()

plt.show()

#%%
print("Number of missing values:", X.isna().sum().sum())

#%%
X["timerecurrence"].describe()

#%%
# for column in X.columns[2:16]:
#     plt.scatter(X[column], y)
#     plt.xlabel(column)
#     plt.show()

#%%
from yellowbrick.features.radviz import RadViz 

features = X.columns[:13]
visualizer = RadViz(classes=class_labels, features=features)

visualizer.fit(X[features], y)
visualizer.transform(df[features])
visualizer.show()

#%%
from yellowbrick.target import FeatureCorrelation

visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X[features], y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#%%
from yellowbrick.features import JointPlotVisualizer

visualizer = JointPlotVisualizer()

visualizer.fit_transform(X["grade"], y)        # Fit and transform the data
visualizer.show()                     # Finalize and render the figure

#%%
from yellowbrick.features import Manifold

viz = Manifold(manifold="tsne", classes=class_labels)

viz.fit_transform(X[features], y)  # Fit the data to the visualizer
viz.show()               # Finalize and render the figure

#%%
from yellowbrick.features import Rank2D

visualizer = Rank2D(algorithm='pearson')

visualizer.fit(df[np.append(features, ["eventdeath"])], y)           # Fit the data to the visualizer
visualizer.transform(df[np.append(features, ["eventdeath"])])        # Transform the data
visualizer.show()              # Finalize and render the figure

#%%
from scipy.stats.stats import pearsonr
p_values = X.apply(lambda col: pearsonr(col, y)[0])

#%%
treshold = 0.35
features = p_values[(p_values > treshold) | (p_values < -treshold)].index.values
features = ["grade", "angioinv", "posnodes", "survival", "age", "diam"]
features = ["grade", "angioinv", "posnodes"]
#best 70 -> accuracy
features = ['AF067420', 'NM_001958', 'NM_000168', 'NM_001993', 'NM_006103',
       'NM_000597', 'Contig37946_RC', 'Contig48043_RC', 'NM_001627',
       'NM_006096', 'Contig51486_RC', 'NM_002343', 'NM_017954', 'AF070536',
       'Contig37873', 'NM_004566', 'NM_001216', 'Contig33260_RC', 'U06715',
       'Contig5403_RC', 'D25328', 'NM_000129', 'NM_018208', 'Contig12593_RC',
       'NC_001807', 'NM_001438', 'Contig56160_RC', 'AF097021', 'NM_003226',
       'NM_002426', 'NM_002416', 'NM_016569', 'Contig41413_RC', 'NM_004484',
       'Contig46937_RC', 'Contig42011_RC', 'Contig58260_RC', 'NM_004950',
       'NM_017422', 'Contig58301_RC', 'Contig23211_RC', 'Contig63748_RC',
       'NM_012067', 'Contig53411_RC', 'NM_005842', 'NM_016249',
       'Contig27749_RC', 'NM_004456', 'Contig10961_RC', 'NM_002989',
       'NM_000849', 'Contig37483_RC', 'Contig55725_RC', 'NM_017852',
       'Contig50979_RC', 'NM_016359', 'Contig57903_RC', 'lymphinfil',
       'Contig54325_RC', 'grade', 'NM_001102', 'NM_005010', 'NM_001124',
       'NM_021069', 'timerecurrence', 'AL137517', 'NM_000363',
       'Contig39226_RC', 'NM_005139', 'NM_005132']
#features = X.columns[:13]
#features = X.columns
features

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42, shuffle = True, stratify=y)
X_train.shape

#%%
# model training
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

#%%
# compute accuracy score
from sklearn.metrics import accuracy_score
logreg_acc_score = accuracy_score(y_test, logreg_pred)
print(logreg_acc_score)

#%%
from sklearn.metrics import f1_score
logreg_f1_score = f1_score(y_test, logreg_pred)
print(logreg_f1_score)

#%%
from yellowbrick.classifier import confusion_matrix
confusion_matrix(logreg, X_train, y_train, X_test, y_test, classes=class_labels)

#%%
from yellowbrick.model_selection import FeatureImportances
viz = FeatureImportances(logreg, labels=X_train.columns, relative=False)
viz.fit(X, y)
viz.show()

#%%
fimp = pd.Series(viz.feature_importances_, index = viz.features_)
fimp.sort_values(ascending=True).head(10).index
#%%
fimp.sort_values().head(10).index

#%%
from sklearn.feature_selection import RFE
rfe = RFE(estimator=logreg, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_
#%%
fimp = pd.Series(ranking, index = viz.features_)
fimp.sort_values(ascending=True).head(20).index

#%%
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=logreg, step=1, cv=2, n_jobs=-1, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#%%
ranking = rfecv.ranking_
fimp = pd.Series(ranking, index = features)
fimp.sort_values(ascending=True).head(rfecv.n_features_).index

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

display(X_train.shape)

mlp = MLPClassifier(hidden_layer_sizes=(100))
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', mlp)
    ],
    verbose=True
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

score = f1_score(y_test, y_pred)
print(score)

#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

score = f1_score(y_test, y_pred)
print(score)

#%%
rfecv = RFECV(estimator=rf, step=1, cv=2, n_jobs=-1, scoring='f1')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

ranking = rfecv.ranking_
fimp = pd.Series(ranking, index = features)
important_features = fimp.sort_values(ascending=True).head(rfecv.n_features_).index
important_features

#%%
rf.fit(X_train[important_features], y_train)
y_pred = rf.predict(X_test[important_features])

score = f1_score(y_test, y_pred)
print(score)