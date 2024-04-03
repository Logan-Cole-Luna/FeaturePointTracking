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

from mpl_toolkits.mplot3d import Axes3D

# Example: Load the first descriptor file for illustration
# Adjust the path as necessary to point to an actual descriptor file
descriptor = np.load('descriptors_folder/first_descriptor.npy')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Taking only the first three features of each descriptor for plotting
ax.scatter(descriptor[:, 0], descriptor[:, 1], descriptor[:, 2])

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D Scatter Plot of Descriptor Features')

plt.show()
