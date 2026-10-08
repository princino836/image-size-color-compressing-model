from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
image = Image.open("BEIP2786.JPG").convert("RGB")

# Convert image to NumPy array
pixels = np.array(image)

print("Original shape:", pixels.shape)

# Get height, width and channels
height, width, channels = pixels.shape

# Flatten pixels
pixels_2d = pixels.reshape(-1, 3)

# K-Means
kmeans = KMeans(n_clusters=16, random_state=42, n_init=10)
kmeans.fit(pixels_2d)

# Get compressed colors
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Convert to uint8
compressed_pixels = np.clip(
    compressed_pixels, 0, 255
).astype(np.uint8)

# Reshape back into image
compressed_image = compressed_pixels.reshape(
    height, width, 3
)

# Display
# compressed image
plt.imshow(compressed_image)
plt.axis("off")
plt.show()