
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 1. Membaca citra awal
gambar = iio.imread("gambar.jpg")

# 2. Mengubah nilai piksel menjadi rentang 0-1
gambar_float = gambar.astype(np.float32) / 255.0

# 3. Mengurangi noise menggunakan Gaussian Filter
gambar_halus = ndimage.gaussian_filter(
    gambar_float, sigma=(1, 1, 0)
)

# 4. Meningkatkan kontras citra
nilai_min = np.percentile(gambar_halus, 2)
nilai_maks = np.percentile(gambar_halus, 98)

citra_tingkat = np.clip(
    (gambar_halus - nilai_min) /
    (nilai_maks - nilai_min),
    0, 1
)

# 5. Mengembalikan nilai piksel ke rentang 0-255
citra_tingkat = (citra_tingkat * 255).astype(np.uint8)

# 6. Menampilkan perbandingan citra
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gambar)
plt.title("Citra Awal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(citra_tingkat)
plt.title("Citra Setelah Peningkatan")
plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_program.png", dpi=150)
plt.show()

# 7. Menyimpan citra hasil peningkatan
iio.imwrite("gambar_ditingkatkan.png", citra_tingkat)
