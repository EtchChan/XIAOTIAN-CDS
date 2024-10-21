import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_data(file_path):
    return np.load(file_path, allow_pickle=True)


def extract_features(tracks):
    drone_rcs = []
    drone_radial_velocity = []
    not_drone_rcs = []
    not_drone_radial_velocity = []

    for track in tracks:
        x, y, length = track
        for point in x[:length]:
            # remove NaNs by converting them to 0
            point = [0 if np.isnan(val) else val for val in point]
            _, _, _, radial_velocity, _, rcs = point
            if y == 1:  # Drone
                drone_rcs.append(rcs)
                drone_radial_velocity.append(radial_velocity)
            else:  # Not drone
                not_drone_rcs.append(rcs)
                not_drone_radial_velocity.append(radial_velocity)

    return drone_rcs, drone_radial_velocity, not_drone_rcs, not_drone_radial_velocity


def fit_and_plot(data1, data2, label1, label2, feature_name):
    plt.figure(figsize=(10, 6))

    # Fit and plot for data1
    kde1 = stats.gaussian_kde(data1)
    mu1, std1 = stats.norm.fit(data1)
    x1 = np.linspace(min(data1), max(data1), 100)
    plt.plot(x1, kde1(x1), label=f'{label1} KDE')
    plt.plot(x1, stats.norm.pdf(x1, mu1, std1), label=f'{label1} Gaussian Fit')
    plt.hist(data1, bins=50, density=True, alpha=0.5, label=f'{label1} Histogram')

    # Fit and plot for data2
    kde2 = stats.gaussian_kde(data2)
    mu2, std2 = stats.norm.fit(data2)
    x2 = np.linspace(min(data2), max(data2), 100)
    plt.plot(x2, kde2(x2), label=f'{label2} KDE')
    plt.plot(x2, stats.norm.pdf(x2, mu2, std2), label=f'{label2} Gaussian Fit')
    plt.hist(data2, bins=50, density=True, alpha=0.5, label=f'{label2} Histogram')

    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {feature_name} for {label1} vs {label2}')
    plt.legend()
    plt.savefig(f'./{feature_name}_distribution_contrast.jpg')
    plt.show()


def main():
    file_path = '../../data/event_2/raw_tracks_graph.npy'
    tracks = load_data(file_path)

    drone_rcs, drone_radial_velocity, not_drone_rcs, not_drone_radial_velocity = extract_features(tracks)

    fit_and_plot(drone_rcs, not_drone_rcs, 'Drone', 'Not Drone', 'RCS')
    fit_and_plot(drone_radial_velocity, not_drone_radial_velocity, 'Drone', 'Not Drone', 'Radial Velocity')


def temp_demo_of_confusion_matrix():
    import seaborn as sns
    cm = [[0.81, 0.19], [0.35, 0.65]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix of Ensemble Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("./Ensemble_Model_confusion_matrix.jpg")
    plt.show()


if __name__ == "__main__":
    # main()

    temp_demo_of_confusion_matrix()