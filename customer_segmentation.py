import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load customer data
df = pd.read_csv("data/customers.csv")

# Select features for clustering
features = [
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency"
]

X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Create customer segments
df["Segment"] = kmeans.fit_predict(X_scaled)

# Display results
print("\nCustomer Segmentation Results:")
print(df[[
    "CustomerID",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "Segment"
]].to_string(index=False))

# Display average characteristics of each segment
print("\nSegment Characteristics:")
print(
    df.groupby("Segment")[features]
    .mean()
    .round(2)
)

# Save results
df.to_csv("customer_segments.csv", index=False)

print("\nSegmentation completed successfully!")
print("Results saved to data/customer_segments.csv")