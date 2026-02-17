# %%
# =============================================================================
# K-MEANS CLUSTERING EXAMPLE: House Votes Analysis
# =============================================================================
# This script demonstrates unsupervised learning using K-means clustering
# to analyze voting patterns on Democrat-introduced bills.
#
# Learning Objectives:
# 1. Apply K-means clustering to real-world data
# 2. Determine optimal number of clusters using multiple methods
# 3. Visualize clustering results in 3D space
# 4. Compare models with and without cluster features

# Load libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# %%
# =============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# =============================================================================
# Load voting data for Democrat-introduced bills
# Each row represents a legislator's voting record
# Columns: aye (yes votes), nay (no votes), other (abstentions/absences)
house_votes_Dem = pd.read_csv("house_votes_Dem.csv", encoding='latin')
house_votes_Rep = pd.read_csv("house_votes_Rep.csv")

# %%
# Examine the structure and first few rows of the data
# This helps us understand what features we're working with
print(house_votes_Dem.head())
house_votes_Dem.info()

# %%
# Create summary statistics grouped by party affiliation
# This gives us a baseline understanding of voting patterns by party
# We can see if Democrats and Republicans vote differently on Dem bills
# house_votes_Rep.groupby("party.labels").agg(
#     {"aye": "sum", "nay": "sum", "other": "sum"})
house_votes_Dem.groupby("party.labels").agg(
    {"aye": "sum", "nay": "sum", "other": "sum"})

# %%
# =============================================================================
# STEP 2: INITIAL K-MEANS CLUSTERING
# =============================================================================
# Select only the numerical voting columns for clustering
# We're using unsupervised learning, so we don't use party labels yet
clust_data_Dem = house_votes_Dem[["aye", "nay", "other"]]

# Run K-means with 2 clusters (we'll optimize this number later)
# random_state=1 ensures reproducible results
# K-means algorithm:
#   1. Randomly initialize 2 cluster centers
#   2. Assign each point to nearest center
#   3. Update centers to mean of assigned points
#   4. Repeat steps 2-3 until convergence
kmeans_obj_Dem = KMeans(n_clusters=2, random_state=1).fit(clust_data_Dem)

# %%
# Examine the clustering results
# cluster_centers_: The coordinates of the 2 cluster centers in 3D space
#                   (one center for each cluster)
print("Cluster Centers (aye, nay, other):")
print(kmeans_obj_Dem.cluster_centers_)

# labels_: Which cluster (0 or 1) each legislator was assigned to
print("\nCluster Labels for each observation:")
print(kmeans_obj_Dem.labels_)

# inertia_: Within-cluster sum of squares (lower is better)
#           Measures how tight/compact the clusters are
print("\nWithin-cluster sum of squares (WCSS):")
print(kmeans_obj_Dem.inertia_)

# %%
# =============================================================================
# VISUALIZE CLUSTERS IN 3D SPACE
# =============================================================================
# Create interactive 3D scatter plot showing the two clusters
# Each point is a legislator, colored by their assigned cluster
# This helps us see if the clusters make intuitive sense
fig = px.scatter_3d(
    house_votes_Dem, x="aye", y="nay", z="other",
    color=kmeans_obj_Dem.labels_,
    title="Aye vs. Nay vs. Other votes for Democrat-introduced bills")
fig.show(renderer="browser")

# %%
# =============================================================================
# STEP 3: DETERMINE OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# =============================================================================
# Calculate within-cluster sum of squares (WCSS) for different values of k
# WCSS measures total distance of points from their cluster centers
# We test k from 1 to 10 to see which number of clusters works best
wcss = []
for i in range(1, 11):
    kmeans_obj_Dem = KMeans(n_clusters=i, random_state=1).fit(clust_data_Dem)
    wcss.append(kmeans_obj_Dem.inertia_)

# %%
# Plot the Elbow Curve
# Look for the "elbow" - where adding more clusters doesn't help much
# The elbow indicates optimal k (balance between fit and complexity)
# After the elbow, WCSS decreases slowly, suggesting diminishing returns
elbow_data_Dem = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
fig = px.line(elbow_data_Dem, x="k", y="wcss", title="Elbow Method")
fig.show()

# %%
# Based on the elbow plot, retrain the model with 3 clusters
# (In practice, you'd choose k based on where you see the elbow)
kmeans_obj_Dem = KMeans(n_clusters=3, random_state=1).fit(clust_data_Dem)

# %%
# =============================================================================
# STEP 4: EVALUATE MODEL - VARIANCE EXPLAINED
# =============================================================================
# Calculate total sum of squares (TSS)
# This measures total variance in the data (spread around the grand mean)
# Formula: sum of squared distances from each point to the overall mean
total_sum_squares = np.sum((clust_data_Dem - np.mean(clust_data_Dem))**2)
total = np.sum(total_sum_squares)
print(f"Total Sum of Squares (TSS): {total}")

# %%
# Calculate Between-Cluster Sum of Squares (BSS)
# BSS = TSS - WSS (inertia)
# BSS measures variance BETWEEN clusters (how separated they are)
# WSS (inertia) measures variance WITHIN clusters (how tight they are)
between_SSE = (total - kmeans_obj_Dem.inertia_)
print(f"Between-Cluster Sum of Squares (BSS): {between_SSE}")

# Variance Explained = BSS / TSS
# This is like R² for clustering - what % of variance is explained?
# Higher is better (means clusters capture meaningful patterns)
# Range: 0 to 1, where 1 = perfect clustering
var_explained = between_SSE / total
print(f"Variance Explained: {var_explained:.4f} or {var_explained*100:.2f}%")

# %%
# =============================================================================
# STEP 5: ADVANCED VISUALIZATION - CLUSTERS VS. ACTUAL PARTY LABELS
# =============================================================================
# Create 3D plot showing BOTH cluster assignments and true party labels
# Color = actual party affiliation (the "truth")
# Symbol/shape = cluster assignment (what K-means found)
# This reveals how well unsupervised clustering matches known groups
fig = px.scatter_3d(
    house_votes_Dem, x="aye", y="nay", z="other",
    color="party.labels", symbol=kmeans_obj_Dem.labels_,
    title="Aye vs. Nay vs. Other votes for Democrat-introduced bills")

# Add black markers showing the 3 cluster centers
# These are the "centroids" - the mean position of each cluster
# They represent the "typical" voting pattern for each cluster
fig.add_trace(go.Scatter3d(
    x=kmeans_obj_Dem.cluster_centers_[:, 0],
    y=kmeans_obj_Dem.cluster_centers_[:, 1],
    z=kmeans_obj_Dem.cluster_centers_[:, 2],
    mode="markers",
    marker=dict(size=20, color="black"),
    name="Cluster Centers"))

fig.show(renderer="browser")

# %%
# =============================================================================
# STEP 6: ALTERNATIVE METHOD - SILHOUETTE SCORE
# =============================================================================
# The Silhouette Score is another way to find optimal number of clusters
# URL: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
#
# What does it measure?
# - For each point, it measures how similar it is to its own cluster (a)
#   versus how similar it is to the nearest other cluster (b)
# - Formula: (b - a) / max(a, b)
# - Range: -1 to +1
#   * +1 = perfect clustering (points very close to own cluster)
#   * 0 = point is on the border between two clusters
#   * -1 = point probably assigned to wrong cluster
#
# Advantages over Elbow Method:
# - More objective (less subjective "elbow" identification)
# - Considers both separation AND cohesion
# - Penalizes overlapping clusters

# %%
from sklearn.metrics import silhouette_score

# Calculate silhouette score for k = 2 through 10
# Note: Silhouette requires at least 2 clusters, so we start at k=2
silhouette_scores = []
for k in range(2, 11):
    kmeans_obj = KMeans(n_clusters=k, algorithm="lloyd", random_state=1)
    kmeans_obj.fit(clust_data_Dem)
    # Calculate average silhouette score across all points
    silhouette_scores.append(
        silhouette_score(clust_data_Dem, kmeans_obj.labels_))

# Find k with highest silhouette score (that's our optimal number)
best_nc = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters by Silhouette Score: {best_nc}")

# %%
# Plot silhouette scores across different values of k
# Look for the highest point - that's the best number of clusters
# Unlike elbow method, this gives a clear maximum to choose
fig = go.Figure(data=go.Scatter(
    x=list(range(2, 11)),
    y=silhouette_scores,
    mode='lines+markers'))
fig.update_layout(
    title="Silhouette Score by Number of Clusters",
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Silhouette Score")
fig

# %%
# =============================================================================
# OPTIONAL: GAP STATISTIC (Advanced Topic)
# =============================================================================
# The Gap Statistic is yet another method for choosing k
# Paper: https://web.stanford.edu/~hastie/Papers/gap.pdf
#
# Key Idea:
# - Compare clustering on real data vs. random uniform data
# - If real data has better clustering than random, there's structure
# - Choose k where gap between real and random is largest
#
# This method is more computationally intensive but more rigorous
# (Not implemented here, but worth knowing about!)

# %%
# =============================================================================
# STEP 7: USING CLUSTERS AS FEATURES IN SUPERVISED LEARNING
# =============================================================================
# Now we'll see if cluster membership helps predict party affiliation
# This demonstrates how unsupervised learning can enhance supervised models

# Fit K-means with 3 clusters and add cluster labels to dataset
kmeans_obj_Dem = KMeans(
    n_clusters=3, algorithm="lloyd", random_state=1).fit(clust_data_Dem)
house_votes_Dem['clusters'] = kmeans_obj_Dem.labels_

# Prepare data for decision tree
# Remove Last.Name (not useful for prediction)
tree_data = house_votes_Dem.drop(columns=["Last.Name"])

# Convert categorical variables to category type
# This helps the decision tree handle them properly
tree_data[['party.labels', 'clusters']] = (
    tree_data[['party.labels', 'clusters']].astype('category'))

# Split data: 70% train, 15% tune, 15% test
# We use tune set to evaluate model before final test
train, tune_and_test = train_test_split(
    tree_data, test_size=0.3, random_state=1)
tune, test = train_test_split(tune_and_test, test_size=0.5, random_state=1)

# Separate features (X) from target (y)
features = train.drop(columns=["party.labels"])
target = train["party.labels"]

# Train Decision Tree WITH cluster feature
# This model has 4 features: aye, nay, other, AND cluster assignment
party_dt = DecisionTreeClassifier(random_state=1)
party_dt.fit(features, target)

# Predict on tune set and show confusion matrix
dt_predict_1 = party_dt.predict(tune.drop(columns=["party.labels"]))
print("=" * 60)
print("CONFUSION MATRIX: Decision Tree WITH Clusters")
print("=" * 60)
print(confusion_matrix(dt_predict_1, tune["party.labels"]))
print("\nRows = Predicted, Columns = Actual")

# %%
# =============================================================================
# COMPARISON: MODEL WITHOUT CLUSTER FEATURES
# =============================================================================
# Now train the same model WITHOUT cluster information
# This shows whether clusters add predictive value

tree_data_nc = tree_data.drop(columns=["clusters"])
train, tune_and_test = train_test_split(
    tree_data_nc, test_size=0.3, random_state=1)
tune, test = train_test_split(tune_and_test, test_size=0.5, random_state=1)

# Now features only include: aye, nay, other (no cluster)
features = train.drop(columns=["party.labels"])
target = train["party.labels"]

# Train Decision Tree WITHOUT cluster feature
party_dt = DecisionTreeClassifier(random_state=1)
party_dt.fit(features, target)

# Predict and compare performance
dt_predict_t = party_dt.predict(tune.drop(columns=["party.labels"]))
print("\n" + "=" * 60)
print("CONFUSION MATRIX: Decision Tree WITHOUT Clusters")
print("=" * 60)
print(confusion_matrix(dt_predict_t, tune["party.labels"]))
print("\nRows = Predicted, Columns = Actual")
print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print("Compare the two confusion matrices above:")
print("- Are accuracy rates similar or different?")
print("- Does adding cluster membership improve predictions?")
print("- This shows whether unsupervised learning found useful patterns!")
# %%
