import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel('tracking_results_with_descriptors.xlsx')

# Group by frame and class to see the number of keypoints over time for each class
grouped = df.groupby(['Frame', 'Class'])['Keypoints'].sum().unstack().fillna(0)
grouped = grouped.reset_index().melt(id_vars='Frame', var_name='Class', value_name='Keypoints')

# Use seaborn lineplot for a line chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='Frame', y='Keypoints', hue='Class', marker='o')
plt.title('Number of Keypoints Detected Over Frames')
plt.xlabel('Frame')
plt.ylabel('Number of Keypoints')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='Class')
plt.show()

# Distribution of Confidence Levels using seaborn
plt.figure(figsize=(8, 4))
sns.histplot(df['Confidence'], bins=20, kde=True, alpha=0.7)
plt.title('Distribution of Detection Confidence Levels')
plt.xlabel('Confidence Level')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to your descriptors folder
descriptors_folder = 'descriptors_folder'

# Get a list of all descriptor files
descriptor_files = os.listdir(descriptors_folder)

# Initialize lists to store the features
features = []

# Loop over the descriptor files and append the features
for file in descriptor_files:
    descriptor_path = os.path.join(descriptors_folder, file)
    descriptor = np.load(descriptor_path)
    features.append(descriptor[0, :3])  # Assuming each descriptor has at least 3 features

# Convert the list of features to a numpy array for easier slicing
features = np.array(features)

# Now we'll plot these features in a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the first three features, color-coded by the index which could represent time or frame number
scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=range(len(features)), cmap='viridis')

# Creating a colorbar to represent the frame numbers or index
cbar = plt.colorbar(scatter, shrink=0.5, aspect=5)
cbar.set_label('Index/Frame Number')

# Labeling axes
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.title('3D Scatter Plot of First Three Descriptor Features')
plt.show()

