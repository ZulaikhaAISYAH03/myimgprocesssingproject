import cv2
import numpy as np
import matplotlib.pyplot as plt

paths = [
    r"C:\Users\Zulaikha\Documents\spyder\assignment\tiger.jpeg",
    r"C:\Users\Zulaikha\Documents\spyder\assignment\animal.jpg"
]

results = []

for idx, path in enumerate(paths):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # filled
    
    overlay = img.copy()
    overlay[mask == 255] = [0, 0, 255]  # Red in BGR
    alpha = 0.5  # transparency
    highlighted = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    results.append((img, gray, thresh, highlighted))  # include grayscale

fig, axes = plt.subplots(len(paths), 4, figsize=(20, 8))

for i, (img, gray, thresh, highlighted) in enumerate(results):
    axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original {i+1}")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(gray, cmap="gray")
    axes[i, 1].set_title(f"Grayscale {i+1}")
    axes[i, 1].axis("off")
    
    axes[i, 2].imshow(thresh, cmap="gray")
    axes[i, 2].set_title(f"Threshold {i+1}")
    axes[i, 2].axis("off")
    
    axes[i, 3].imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    axes[i, 3].set_title(f"Highlighted {i+1}")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()
