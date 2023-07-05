import numpy as np
import csv

def generate_points_on_sphere(num_points):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.column_stack((x, y, z))
    return points

def write_points_to_csv(points, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['x', 'y', 'z'])  # Write header
        writer.writerows(points)

# Usage example
num_points = 1024
points = generate_points_on_sphere(num_points)
outFileName = "sphere_%d.csv" % num_points;
write_points_to_csv(points, outFileName)