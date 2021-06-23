#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# FONT_SIZE=12
# plt.rcParams.update({'font.size': FONT_SIZE})
# plt.rc('axes', titlesize=FONT_SIZE, labelsize=FONT_SIZE)
# plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

sns.set_context("notebook")
sns.set_style("whitegrid")

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
X = df.drop(["Patient", "ID", "barcode", "eventdeath"], axis=1)
y = df['eventdeath']

display(X.shape)
display(y.shape)

#%%
from yellowbrick.target import ClassBalance

class_labels = ["survived", "deceased"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3.6))
y_count = y.map({0:"survived", 1:"deceased"}).value_counts()
_, _, autotexts = ax1.pie(y_count, labels=y_count.index,  explode=[0.03]*y_count.shape[0], startangle=140, labeldistance = 1.3, autopct=lambda x: f"{x:.1f}% ({int(round(y_count.sum()*x/100))})")
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('semibold')
ax1.yaxis.label.set_visible(False)

visualizer = ClassBalance(labels = class_labels, ax = ax2)
visualizer.fit(y)
visualizer.finalize()
ax2.yaxis.label.set_visible(False)
ax2.title.set_visible(False)
fig.tight_layout()
fig.savefig("nkiClassDist.pdf")

#%%
print("Number of missing values:", X.isna().sum().sum())

#%%
fig, axes = plt.subplots(4, 3, figsize=(8, 8))
for i in range(4):
    for j in range(3):
        ax = axes[i][j]
        column = X.columns[i * 3 + j]
        X[column].plot.hist(ax=ax)
        ax.yaxis.label.set_visible(False)
        ax.set_title(column.capitalize())
        #ax.set_xlabel(column.capitalize())
        
fig.tight_layout()
fig.savefig("featureViz12.pdf")

#%%
X["timerecurrence"].describe()

#%%
from yellowbrick.features.radviz import RadViz

# Method 1: experiment with different ordering of the features on the circle
# features = X.columns[:12]
# np.random.shuffle(features.values)
# display(features)
# Method 2: go with an order you like
features = ['lymphinfil', 'chemo', 'angioinv', 'diam', 'grade',
       'hormonal', 'amputation', 'age', 'survival', 'histtype', 'posnodes',
       'timerecurrence']

fig, ax = plt.subplots(figsize=(6.3, 4))
visualizer = RadViz(classes=class_labels, features=features, ax = ax)
visualizer.fit(X[features], y)
visualizer.transform(X[features])
fig.tight_layout()
ax.title.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
visualizer.show()
fig.savefig("radviz12features.pdf")

#%%
from yellowbrick.target import FeatureCorrelation

visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X[features], y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#%%
from yellowbrick.features import JointPlotVisualizer

visualizer = JointPlotVisualizer()

visualizer.fit_transform(X["age"], y)        # Fit and transform the data
visualizer.show()                     # Finalize and render the figure

#%%
from yellowbrick.features import Manifold

viz = Manifold(manifold="tsne", classes=class_labels)

viz.fit_transform(X[features], y)  # Fit the data to the visualizer
viz.show()               # Finalize and render the figure

#%%
from yellowbrick.features import Rank2D

fig, ax = plt.subplots(figsize=(6, 5))
visualizer = Rank2D(algorithm='pearson', ax=ax)

visualizer.fit(df[np.append(features, ["eventdeath"])], y)           # Fit the data to the visualizer
visualizer.transform(df[np.append(features, ["eventdeath"])])        # Transform the data
fig.tight_layout()
ax.title.set_visible(False)
visualizer.show()              # Finalize and render the figure
fig.savefig("pearsonRanking13.pdf")

#%%
from scipy.stats.stats import pearsonr
p_values = X.apply(lambda col: pearsonr(col, y)[0])
treshold = 0.35
features = p_values[(p_values > treshold) | (p_values < -treshold)].index.values

#%%
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
features = X.columns[:12]
#features = X.columns
features

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42, shuffle = True, stratify=y)
X_train.shape

#%%
# model training
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

#%%
# compute accuracy score
from sklearn.metrics import accuracy_score
logreg_acc_score = accuracy_score(y_test, logreg_pred)
print("acc: ", logreg_acc_score * 100, "%")

#%%
from sklearn.metrics import f1_score
logreg_f1_score = f1_score(y_test, logreg_pred)
print("f1:", logreg_f1_score)

# %%
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(max_iter=1000),
 X[features], y, scoring="accuracy", cv=3, n_jobs=1)

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

pipeline = Pipeline([
    ('scaler', Normalizer()),
    ('logreg', LogisticRegression(max_iter=1000))
    ],
    verbose=True
)
pipeline.fit(X_train, y_train)
logreg_pred = pipeline.predict(X_test)

logreg_acc_score = accuracy_score(y_test, logreg_pred)
print("acc: ", logreg_acc_score * 100, "%")
logreg_f1_score = f1_score(y_test, logreg_pred)
print("f1:", logreg_f1_score)

#%%
from yellowbrick.classifier import ConfusionMatrix
fig, ax = plt.subplots(figsize=(4, 3.2))
cfm = ConfusionMatrix(LogisticRegression(max_iter=1000), classes=class_labels, ax=ax)
cfm.fit(X_train, y_train)
cfm.score(X_test, y_test)
cfm.finalize()
ax.title.set_visible(False)
ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
fig.savefig("confMatrix12.pdf")

#%%
from yellowbrick.model_selection import FeatureImportances
fig, ax = plt.subplots(figsize=(6, 4))
viz = FeatureImportances(logreg, labels=X_train.columns, relative=False, ax=ax)
viz.fit(X, y)
ax.title.set_visible(False)
fig.tight_layout()
viz.show(outpath="featImpLog12.pdf")

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

# %%
def plot_rfecv(rfecv, out_path):
    # Plot number of features VS. cross-validation scores
    fig, ax = plt.subplots(figsize=np.array([6.4, 4.8]) / 1.3) # default figsize = [6.4, 4.8]
    ax.set_xlabel("Number of Features Selected")
    ax.set_ylabel("Score")
    ax.plot(np.arange(1, rfecv.grid_scores_.shape[0] + 1), rfecv.grid_scores_)
    if rfecv.grid_scores_.shape[0] < 30:
        ax.scatter(np.arange(1, rfecv.grid_scores_.shape[0] + 1), rfecv.grid_scores_)
    h = ax.axvline(rfecv.n_features_, color="k", linestyle="--", alpha=0.7)
    ax.legend([h], [f"n_features = {rfecv.n_features_}\nscore = {rfecv.grid_scores_[rfecv.n_features_]:.2f}"], loc="lower center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.show()

def save_features(rfecv, out_path):
    pd.Series(features[rfecv.support_].sort_values(), name="Features").to_csv(out_path, index=False)

#%%
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
rfecvLogAcc = RFECV(estimator=LogisticRegression(max_iter=1000),
 step=1, cv=3, n_jobs=-1, scoring='accuracy')
rfecvLogAcc.fit(X[features], y)
plot_rfecv(rfecvLogAcc, out_path="featSelLogAcc.pdf")
save_features(rfecvLogAcc, "featSelLogAcc.csv")

# %%
rfecvLogF1 = RFECV(estimator=LogisticRegression(max_iter=1000),
 step=1, cv=3, n_jobs=-1, scoring='f1')
rfecvLogF1.fit(X[features], y)
plot_rfecv(rfecvLogF1, out_path="featSelLogF1.pdf")
save_features(rfecvLogF1, "featSelLogF1.csv")

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
from sklearn.ensemble import RandomForestClassifier
rfecvRfAcc = RFECV(estimator=RandomForestClassifier(),
 step=1, cv=3, n_jobs=-1, scoring='accuracy')
rfecvRfAcc.fit(X[features], y)
plot_rfecv(rfecvRfAcc, out_path="featSelRfAcc.pdf")
save_features(rfecvRfAcc, "featSelRfAcc.csv")

# %%
from sklearn.ensemble import RandomForestClassifier
rfecvRfF1 = RFECV(estimator=RandomForestClassifier(),
 step=1, cv=3, n_jobs=-1, scoring='f1')
rfecvRfF1.fit(X[features], y)
plot_rfecv(rfecvRfF1, out_path="featSelRfF1.pdf")
save_features(rfecvRfF1, "featSelRfF1.csv")

#%%
rf.fit(X_train[important_features], y_train)
y_pred = rf.predict(X_test[important_features])

score = f1_score(y_test, y_pred)
print(score)

# %%
from yellowbrick.model_selection.rfecv import RFECV as ybRFECV
fig, ax = plt.subplots(figsize=np.array([6.4, 4.8]) / 1.3) # default figsize = [6.4, 4.8]
visualizer = ybRFECV(RandomForestClassifier(), ax=ax, scoring="accuracy")
visualizer.fit(X[features], y)
ax.title.set_visible(False)
visualizer.show()
fig.tight_layout()
#fig.savefig("featSelRfAcc.pdf")

# %%
from yellowbrick.model_selection.rfecv import RFECV as ybRFECV
fig, ax = plt.subplots(figsize=np.array([6.4, 4.8]) / 1.3) # default figsize = [6.4, 4.8]
visualizer = ybRFECV(RandomForestClassifier(), ax=ax, scoring="f1")
visualizer.fit(X[features], y)
ax.title.set_visible(False)
visualizer.show()
fig.tight_layout()
#fig.savefig("featSelRfF1.pdf")

###################################################################
# %%
table = pd.read_csv("featSelLogAcc.csv")
table = pd.DataFrame(table.values.reshape((8, -1), order="F"))
with open("featSelLogAcc.tex", "w") as f:
    table.to_latex(f, header=False, index=False)

# %%
table1 = pd.read_csv("featSelRfAcc.csv")
table2 = pd.read_csv("featSelRfF1.csv")
table1[table2.columns[0]]=table2.iloc[:,0]
with open("featSelRf.tex", "w") as f:
    table1.to_latex(f, index=False, na_rep="")

