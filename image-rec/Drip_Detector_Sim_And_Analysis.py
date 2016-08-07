import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Spatial scale, 1 pix = 0.1 mm


def circle(weight=10.0, center=[0, 0], radii=5, im_size=[500, 100]):

    a, b = center[0], center[1]
    r = radii

    x, y = np.ogrid[-a:im_size[0]-a, -b:im_size[1]-b]
    mask = x*x + y*y <= r*r

    arr = np.zeros((im_size[0], im_size[1]))
    arr[mask] = weight
    arr[np.where(arr == 0)] = 1
    return arr


def drop(center=[5, 49]):
    circ1 = circle(weight=-100, radii=4, center=center)
    circ2 = circle(weight=500.0, radii=5, center=center)
    return circ1+circ2


def drip_images():

    # Simulate camera data at 50 fps
    # Drop falls after 0.5 seconds and then accelerates due to gravity
    # Leaves camera field of view after approx 1.5 second

    t = np.arange(0, 2, 0.02)  # 50 fps
    x = 0.5*9.81*t**2*100*1
    
    imgs = []
    for i in range(25):
        imgs.append(drop(center=[5, 49])+np.random.poisson(drop(center=[5, 49])))
    for dist in x:
        imgs.append(drop(center=[5 + dist, 49])+np.random.poisson(drop(center=[5 + dist, 49])))

    return imgs


def laplacian_rate_of_change():

    # Simulate camera data at 50 fps
    data = drip_images()
    lap_data = []

    # Absolute Laplaican of each image
    for i in data:                
        lap_data.append(abs(ndimage.filters.laplace(i)))

    lap_rate_of_change = []
    for i in range(len(lap_data)-1):
        lap_rate_of_change.append(abs((lap_data[i+1]-lap_data[i]).sum()))

    return np.asarray(lap_rate_of_change)


def plot_laplacian_rate_of_change():

    data = laplacian_rate_of_change()
    t = np.arange(0.02, 2.5, 0.02)
    plt.figure()
    print(t.shape)
    plt.step(t, data/data.max(), linewidth=2)
    plt.xlabel("Time / [Seconds]", size=14)
    plt.ylabel("Absolute Laplacian Rate of Change", size=14)
    plt.show()


if __name__ == "__main__":
    plot_laplacian_rate_of_change()
