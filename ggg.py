import pandas as pd
import numpy as np

# Load the dataset from the Excel file
file_path = 'Dry_Bean_Dataset.csv'
df = pd.read_csv(file_path)

# Assuming you have a DataFrame 'df' with a 'Class' column
# Map the two chosen classes
df['Class'] = df['Class'].map({'Cali': 1, 'Sira': -1})

# Fill NAN values
df.interpolate(method='linear', inplace=True)
# df = df.fillna(df.mean())

# Assuming df contains your dataset, and the first 5 columns are the features
features = df.iloc[:, :5]  # Extracting the first 5 columns as features

# Separate rows with Class values of -1 and 1 after mapping
class1_df = df[df['Class'] == 1]
class2_df = df[df['Class'] == -1]

# Set a random seed for reproducibility
np.random.seed(42)

# Shuffle each class rows
class1_df = class1_df.sample(frac=1, random_state=42).reset_index(drop=True)
class2_df = class2_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and target separately for each class
X1 = class1_df[['Area', 'Perimeter']].values
X2 = class2_df[['Area', 'Perimeter']].values
y1 = class1_df['Class'].values
y2 = class2_df['Class'].values

X1_train = X1[:30]
X2_train = X2[:30]
X1_test = X1[30:50]
X2_test = X2[30:50]
X_train = np.concatenate((X1_train, X2_train), axis=0)
X_test = np.concatenate((X1_test, X2_test), axis=0)
x = np.concatenate((X_train, X_test), axis=0)

y1_train = y1[:30]
y2_train = y2[:30]
y1_test = y1[30:50]
y2_test = y2[30:50]
y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print(type(X_train))

