import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA



# Make sure to adjust these paths to where your data is actually stored
tracking_results_path = 'tracking_results_with_descriptors.xlsx'
keypoint_tracking_results_path = 'keypoint_tracking_results.xlsx'

# Load the Excel file
df = pd.read_excel(tracking_results_path)

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

# Path to your descriptors folder
descriptors_folder = 'descriptors_folder'

# Get a list of all descriptor files
descriptor_files = [f for f in os.listdir(descriptors_folder) if f.endswith('.npy')]

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

# Load your DataFrames (make sure the paths are correct)
df = pd.read_excel(tracking_results_path)
kp_df = pd.read_excel(keypoint_tracking_results_path)

# 1. YOLO Detection Confidence Over Frames
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Frame', y='Confidence', hue='Class', legend='full')
plt.title('YOLO Detection Confidence Over Frames')
plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.legend(title='Detected Class')
plt.show()

# 2. Number of Keypoints Detected Over Frames
plt.figure(figsize=(12, 6))
sns.lineplot(data=kp_df.groupby(['Frame', 'Class']).size().reset_index(name='Keypoint Count'),
             x='Frame', y='Keypoint Count', hue='Class', legend='full')
plt.title('Number of Keypoints Detected Over Frames')
plt.xlabel('Frame')
plt.ylabel('Keypoint Count')
plt.legend(title='Detected Class')
plt.show()

# 3. Histogram of Keypoint Matches Per Frame
plt.figure(figsize=(12, 6))
sns.histplot(kp_df['Frame'], bins=len(df['Frame'].unique()))
plt.title('Histogram of Keypoint Matches Per Frame')
plt.xlabel('Frame')
plt.ylabel('Matches Count')
plt.show()

# 4. 3D Scatter Plot of Keypoint Positions for a Given Class and Frame
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Filter the DataFrame for a specific class and frame, e.g., 'airplane' in frame '00001.jpg'
class_filter = 'airplane'
frame_filter = '00001.jpg'
filtered_df = kp_df[(kp_df['Class'] == class_filter) & (kp_df['Frame'] == frame_filter)]

import ast

# Inside the plotting section where you're encountering the error, use the following:
# Convert string representations of tuples to actual tuples
filtered_df['Keypoint_Position'] = filtered_df['Keypoint_Position'].apply(ast.literal_eval)

# Now proceed with the 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    filtered_df['Keypoint_Position'].apply(lambda x: x[0]),
    filtered_df['Keypoint_Position'].apply(lambda x: x[1]),
    filtered_df['Confidence']
)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Confidence')
plt.title(f'3D Scatter Plot of Keypoints for {class_filter} in Frame {frame_filter}')
plt.show()

# 5. 3D Scatter Plot of Descriptor Space Reduced by PCA
# Run PCA on descriptors to reduce dimensionality
descriptors = np.stack(kp_df['Descriptor'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')))
pca = PCA(n_components=3)
pca_result = pca.fit_transform(descriptors)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
plt.title('3D Scatter Plot of Descriptor Space by PCA')
plt.show()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import ast

# Load data from Excel files
tracking_results_path = 'tracking_results_with_descriptors.xlsx'
keypoint_tracking_results_path = 'keypoint_tracking_results.xlsx'
df = pd.read_excel(tracking_results_path)
kp_df = pd.read_excel(keypoint_tracking_results_path)

# Correcting potential SettingWithCopyWarning and ensuring operation on actual DataFrame
if 'Keypoint_Position' in kp_df.columns:
    kp_df.loc[:, 'Keypoint_Position'] = kp_df['Keypoint_Position'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Assuming Descriptor is a stringified numpy array; convert back to numpy array
if 'Descriptor' in kp_df.columns:
    kp_df.loc[:, 'Descriptor'] = kp_df['Descriptor'].apply(lambda x: np.fromstring(x[1:-1], sep=' ') if isinstance(x, str) else x)
    descriptors = np.stack(kp_df['Descriptor'].values)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(descriptors)

    # 3D Scatter Plot of PCA-reduced Descriptor Space
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
    ax.set_title('PCA Reduced Descriptor Space')
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your DataFrames (make sure the paths are correct)
df = pd.read_excel('tracking_results_with_descriptors.xlsx')
kp_df = pd.read_excel('keypoint_tracking_results.xlsx')

# 1. YOLO Detection Confidence Over Frames
# Create the first plot for YOLO confidence with a primary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df, x='Frame', y='Confidence', ax=ax1, color='blue', label='YOLO Confidence')
ax1.set_xlabel('Frame')
ax1.set_ylabel('YOLO Confidence', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('YOLO Detection Confidence and Keypoint Matches Over Frames')

# Create the second plot for Keypoint matches with a secondary y-axis
ax2 = ax1.twinx()
sns.barplot(data=kp_df.groupby('Frame').size().reset_index(name='Keypoint Count'),
            x='Frame', y='Keypoint Count', ax=ax2, color='orange', alpha=0.5, label='Keypoint Matches')
ax2.set_ylabel('Keypoint Matches Count', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Show the plot
fig.tight_layout()
plt.show()



# Group data by frame and class for plotting
grouped = df.groupby(['Frame', 'Class'])['Keypoints'].sum().unstack().fillna(0)
grouped = grouped.reset_index().melt(id_vars='Frame', var_name='Class', value_name='Keypoints')

# Line plot of keypoints detected over frames
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='Frame', y='Keypoints', hue='Class', marker='o')
plt.title('Number of Keypoints Detected Over Frames')
plt.xlabel('Frame')
plt.ylabel('Number of Keypoints')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='Class')
plt.show()

# Histogram of detection confidence levels
plt.figure(figsize=(8, 4))
sns.histplot(df['Confidence'], bins=20, kde=True, alpha=0.7)
plt.title('Distribution of Detection Confidence Levels')
plt.xlabel('Confidence Level')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Ensure your descriptors_folder exists and contains .npy files
descriptors_folder = 'descriptors_folder'
descriptor_files = [f for f in os.listdir(descriptors_folder) if f.endswith('.npy')]

# Plotting 3D scatter plot of descriptor features
features = []
for file in descriptor_files:
    descriptor_path = os.path.join(descriptors_folder, file)
    descriptor = np.load(descriptor_path)
    features.append(descriptor[0, :3])  # Assuming each descriptor has at least 3 features

features = np.array(features)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=range(len(features)), cmap='viridis')
plt.title('3D Scatter Plot of First Three Descriptor Features')
plt.show()

# Additional plotting code remains as previously provided
# Make sure 'Matched_Keypoint_ID' exists before attempting to plot it
if 'Matched_Keypoint_ID' in kp_df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=kp_df.dropna(subset=['Matched_Keypoint_ID']), x='Frame', y='Matched_Keypoint_ID', hue='Class', legend='full', markers=True)
    plt.title('Keypoint Matches Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Matched Keypoint ID')
    plt.show()
else:
    print("'Matched_Keypoint_ID' column not found in kp_df DataFrame.")



