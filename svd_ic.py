import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MAX_RANK = 100
FNAME = 'car.jpg'

image = Image.open(FNAME).convert("L")
img_mat = np.asarray(image)

U, s, V = np.linalg.svd(img_mat, full_matrices=True)
s = np.diag(s)

for k in range(MAX_RANK + 1):
  approx = U[:, :k] @ s[0:k, :k] @ V[:k, :]
  img = plt.imshow(approx, cmap='gray')
  plt.title(f'SVD approximation with degree of {k}')
  plt.plot()
  pause_length = 0.0001 if k < MAX_RANK else 5
  plt.pause(pause_length)
  plt.clf()

plt.figure(1)
plt.semilogy(s)
plt.title("Singular vaues")
plt.show()

plt.figure(2)
plt.plot(np.cumsum(s)/np.sum(s))
plt.title("Cumulative sum of singular values")
plt.show()

# Compression ratio calculation
compression_ratio = np.sum(img_mat)/np.sum(s)
print(f'Compression ratio: {compression_ratio}')