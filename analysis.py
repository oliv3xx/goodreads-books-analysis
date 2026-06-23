import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('books.csv', on_bad_lines='skip')
df.columns = df.columns.str.strip()
df = df.dropna(subset=['average_rating', 'num_pages', 'ratings_count'])

df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')

df = df[df['num_pages'] > 0]
df = df[df['ratings_count'] > 0]

print(f"Clean dataset: {df.shape[0]} books")

# --- ANALYSIS 1: Top 10 Most Reviewed Authors ---
top_authors = (
    df.groupby('authors')['ratings_count']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_authors.columns = ['Author', 'Total Ratings']
print("\nTop 10 Most Reviewed Authors:")
print(top_authors)

plt.figure(figsize=(10, 6))
plt.barh(top_authors['Author'], top_authors['Total Ratings'], color='steelblue')
plt.xlabel('Total Ratings')
plt.title('Top 10 Most Reviewed Authors on GoodReads')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_authors.png')
plt.show()
print("Saved top_authors.png")

# --- ANALYSIS 2: Rating Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(df['average_rating'], bins=20, color='mediumpurple', edgecolor='white')
plt.xlabel('Average Rating')
plt.ylabel('Number of Books')
plt.title('Distribution of Book Ratings on GoodReads')
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.show()
print("Saved rating_distribution.png")

# --- ANALYSIS 3: Predict Rating with Linear Regression ---

X = df[['num_pages', 'ratings_count']]
y = df['average_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & test model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
print(f"\nLinear Regression Results:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# --- ANALYSIS 5: K-Means Clustering ---

cluster_df = df[['average_rating', 'ratings_count']].copy()

# ratings_count doesn't overpower average_rating
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

# Train K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

print("\nK-Means Cluster Summary:")
print(df.groupby('cluster')[['average_rating', 'ratings_count']].mean().round(2))

# Plot
plt.figure(figsize=(9, 5))
colors = ['steelblue', 'mediumpurple', 'coral', 'mediumseagreen']
for i in range(4):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['ratings_count'], cluster_data['average_rating'],
                alpha=0.3, s=10, color=colors[i], label=f'Cluster {i}')

plt.xscale('log')
plt.xlabel('Ratings Count (log scale)')
plt.ylabel('Average Rating')
plt.title('Book Clusters by Popularity and Rating')
plt.legend()
plt.tight_layout()
cluster_labels = {
    0: 'Underperformers',
    1: 'Mainstream Middles',
    2: 'Blockbusters',
    3: 'Hidden Gems'
}
df['cluster_label'] = df['cluster'].map(cluster_labels)

print("\nExample books per cluster:")
for label in cluster_labels.values():
    sample = df[df['cluster_label'] == label][['title', 'average_rating', 'ratings_count']].head(2)
    print(f"\n{label}:")
    print(sample.to_string(index=False))

plt.savefig('clusters.png')
plt.show()
print("Saved clusters.png")

# --- ANALYSIS 6: Classify Highly Rated Books ---

# Highly rated if average_rating >= 4.0
df['highly_rated'] = (df['average_rating'] >= 4.0).astype(int)

print(f"\nHighly rated books: {df['highly_rated'].sum()}")
print(f"Not highly rated: {(df['highly_rated'] == 0).sum()}")

# Features
X3 = df[['num_pages', 'ratings_count', 'text_reviews_count']]
y3 = df['highly_rated']

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X3_train, y3_train)

y3_pred = clf.predict(X3_test)

print("\nClassification Report:")
print(classification_report(y3_test, y3_pred, target_names=['Not Highly Rated', 'Highly Rated']))

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X3.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# Export clustered data for PowerBI
export_df = df[['title', 'authors', 'average_rating', 'ratings_count', 'num_pages', 'cluster_label']].copy()
export_df.to_csv('books_clustered.csv', index=False)
print("Saved books_clustered.csv")