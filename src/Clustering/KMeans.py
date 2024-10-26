import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Function to convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(distance, azimuth_deg, elevation_deg):
    # Convert degrees to radians
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)

    # Calculate Cartesian coordinates
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    return x, y, z

# Step 1: Data Preprocessing
# data = pd.read_csv('../../data/event_1/raw_tracks_3.csv', skiprows=1)
data = pd.read_csv('../materials/raw_tracks_1.csv', skiprows=1)
data.columns = ['time', 'distance', 'azimuth', 'elevation', 'radial_speed', 'circle']

# Step 2: Initial Detection and Clustering
first_circle_data = data[data['circle'] == data['circle'].min()]
possible_n_values = range(2, 10)
best_n = None
best_score = -1
best_labels = None
position_features = first_circle_data[['distance', 'azimuth', 'elevation']].values

for n_clusters in possible_n_values:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(position_features)
    if len(set(labels)) == 1:
        continue  # Skip if all data is assigned to one cluster
    score = silhouette_score(position_features, labels)
    if score > best_score:
        best_score = score
        best_n = n_clusters
        best_labels = labels

print(f"Estimated number of drones: {best_n}")

# Step 3: Track Initialization
class Track:
    def __init__(self, id, initial_position):
        self.id = id
        self.positions = [initial_position]
        self.x = np.array(initial_position)  # State vector
        self.P = np.eye(3) * 1000  # Covariance matrix
        self.F = np.eye(3)  # State transition model
        self.Q = np.eye(3) * 0.1  # Process noise covariance
        self.R = np.eye(3) * 1  # Measurement noise covariance

tracks = []
kmeans = KMeans(n_clusters=best_n, random_state=42)
kmeans.fit(position_features)
centroids = kmeans.cluster_centers_

for i, centroid in enumerate(centroids):
    track = Track(id=i, initial_position=centroid)
    tracks.append(track)

# Initialize a list to store centers for each circle
temp_traj = []
circle_centers = []

# Step 4: Tracking Over Time and Computing Centers
circles = sorted(data['circle'].unique())

for circle_number in circles[1:]:
    circle_data = data[data['circle'] == circle_number]
    measurements = circle_data[['distance', 'azimuth', 'elevation']].values
    time_stamps = circle_data['time'].values

    # Predict the next position for each track
    for track in tracks:
        # Predict
        track.x = track.F @ track.x
        track.P = track.F @ track.P @ track.F.T + track.Q

    # Data association
    cost_matrix = np.zeros((len(tracks), len(measurements)))

    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            cost = np.linalg.norm(track.x - measurement)
            cost_matrix[i, j] = cost

    # Solve the assignment problem
    track_indices, measurement_indices = linear_sum_assignment(cost_matrix)

    assigned_tracks = set()
    assigned_measurements = set()

    for t_idx, m_idx in zip(track_indices, measurement_indices):
        cost = cost_matrix[t_idx, m_idx]
        if cost < 10.0:  # Threshold for gating
            track = tracks[t_idx]
            measurement = measurements[m_idx]
            y = measurement - track.x
            S = track.P + track.R
            K = track.P @ np.linalg.inv(S)
            track.x = track.x + K @ y
            track.P = (np.eye(3) - K) @ track.P
            track.positions.append(track.x)
            assigned_tracks.add(t_idx)
            assigned_measurements.add(m_idx)
        else:
            pass  # Detection is considered noise

    # **Compute the center of drones for the current circle**

    # Collect current positions of all tracks
    cartesian_positions = []
    for track in tracks:
        distance, azimuth, elevation = track.x  # Current estimated position
        x, y, z = spherical_to_cartesian(distance, azimuth, elevation)
        cartesian_positions.append([x, y, z])

    cartesian_positions = np.array(cartesian_positions)

    # Compute the center (mean position)
    center_x = np.mean(cartesian_positions[:, 0])
    center_y = np.mean(cartesian_positions[:, 1])
    center_z = np.mean(cartesian_positions[:, 2])
    center_position = [center_x, center_y, center_z]

    # Store the center position with the circle number
    circle_centers.append((circle_number, center_position))
    
    temp_traj.append((circle_number, cartesian_positions[2,:]))

# After processing all circles, you can print or analyze the centers
for circle_number, center_pos in circle_centers:
    print(f"Circle {circle_number}: Center Position (X, Y, Z) = {center_pos}")
    
for circle_number, pos in temp_traj:
    print(f"Circle {circle_number}: Center Position (X, Y, Z) = {pos}")

# save the centers into csv file for further analysis
center_df = pd.DataFrame(circle_centers, columns=['circle', 'center_position'])
# add header to the csv file
center_df.to_csv('./tracks_centers_0.csv', index=False)

# plot the centers for further analysis
# centers = np.array([center_pos for circle_number, center_pos in circle_centers])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2])

pos = np.array([pos for circle_number, pos in temp_traj])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

