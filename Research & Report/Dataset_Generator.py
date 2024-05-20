import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for healthcare insurance fraud detection
np.random.seed(42)

# Number of samples (rows)
n_samples = 1000

# Define features and their distributions
n_features = 10  # Number of features
n_informative = 8  # Number of informative features
n_redundant = 2  # Number of redundant features

# Generate synthetic dataset
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_clusters_per_class=2,
    weights=[0.95, 0.05],  # 95% non-fraudulent (0) and 5% fraudulent (1) samples
    flip_y=0.01,  # Probability of noisy labels
    random_state=42
)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create DataFrame
columns = [f"feature_{i+1}" for i in range(n_features)]
columns.append("is_fraudulent")
data = np.column_stack((X, y))
df = pd.DataFrame(data, columns=columns)

# Convert target variable to integer
df['is_fraudulent'] = df['is_fraudulent'].astype(int)

# Save DataFrame to CSV file
csv_file_path = 'healthcare_insurance_fraud_dataset.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset saved to '{csv_file_path}'")
