# COMPUTER VISION FUNDAMENTALS

**Subject:** Computer Vision Fundamentals

---

## Section A: Concept Application

---

### Q1. You load an image with cv2.imread() and print its shape, getting (480, 640, 3). What do these three values represent?

When an image is loaded using OpenCV's `cv2.imread()` function, it is returned as a multi-dimensional NumPy array (`numpy.ndarray`). The output tuple `(480, 640, 3)` represents the spatial and channel dimensions of the image in the specific order of **(Height, Width, Channels)**:

1. **480 (Height):** Represents the number of rows in the pixel matrix. There are 480 horizontal lines of pixels arrayed vertically from top to bottom.
2. **640 (Width):** Represents the number of columns in the pixel matrix. There are 640 vertical lines of pixels arrayed horizontally from left to right.
3. **3 (Channels):** Represents the number of distinct color channels. By default, `cv2.imread()` loads color images with 3 channels corresponding to the Blue, Green, and Red (BGR) color spaces. Each channel is represented as a 2D matrix of shape `(480, 640)` containing pixel intensity values.

#### Memory Contiguity and Matrix Operations
NumPy arrays use **row-major (C-contiguous) ordering** in memory, where rows are stored sequentially. As a result, indexing is structured as `image[row, col, channel]` or `image[y, x, c]`. This row-major layout dictates that y-coordinates (height) are indexed first, followed by x-coordinates (width). When designing coordinate-based operations (like bounding boxes or point-plotting), engineers must map `(x, y)` coordinates to the `[y, x]` array indexing format.

---

### Q2. cv2.imread() loads images in BGR order by default. Why does the BGR vs RGB distinction matter when displaying or passing images to other libraries?

The BGR (Blue, Green, Red) vs. RGB (Red, Green, Blue) distinction is a critical factor in computer vision pipelines due to historical standards and library compatibility.

#### Historical Context
OpenCV was originally developed in the early 2000s by Intel when BGR was the prevailing standard for many digital camera manufacturers, hardware graphics accelerators, and software APIs (such as Windows Device Contexts / Device Independent Bitmaps). To preserve backward compatibility, OpenCV maintained BGR as its default memory layout.

#### The Core Issue
Most modern visualization, processing, and deep learning libraries expect images to be ordered in the standard **RGB** format. These include:
- **Plotting/UI libraries:** Matplotlib (`plt.imshow`), PIL (Pillow), Tkinter, web-based visualizers.
- **Deep Learning Frameworks:** PyTorch, TensorFlow, Keras, Hugging Face `transformers`.

#### Consequences of Channel Mismatches
If a BGR image loaded via `cv2.imread()` is passed directly to a library that expects RGB without conversion:
1. **Color Inversion/Distortion:** The Red and Blue channels are swapped. Any red element in the physical world (e.g., blood, fire, red light) will be rendered as blue on the screen, and vice versa. The Green channel remains unaffected because it sits in the middle of both representations, but the overall color fidelity is ruined.
2. **Deep Learning Model Degradation:** Convolutional Neural Networks (CNNs) are pre-trained on large-scale datasets (such as ImageNet) that are formatted in RGB. The first convolutional layer trains specific filters to capture features based on red, green, and blue components. If a BGR image is fed into the network, the red features are matched against the blue filters, severely compromising the model's accuracy, resulting in garbage inference outputs or poor training convergence.

#### Programmatic Conversion
To safely transition between color spaces, standard conversion modules must be invoked prior to exporting data to other libraries:
```python
# Convert BGR to RGB for Matplotlib or Deep Learning libraries
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
```

---

### Q3. You resize the same image using INTER_LINEAR and INTER_AREA interpolation. When is INTER_AREA the preferred choice over INTER_LINEAR?

Image interpolation methods determine how pixel values are estimated when scaling an image's coordinate grid. Selecting the proper interpolation mode depends on whether the transformation is **downsampling (shrinking)** or **upsampling (enlarging)**.

#### Interpolation Mechanics
- **`INTER_LINEAR` (Bilinear Interpolation):** Computes the target pixel's value by performing a linear interpolation in both the horizontal and vertical directions using a $2 \times 2$ pixel neighborhood around the source coordinates. It is computationally efficient and provides smooth results.
- **`INTER_AREA` (Area-based Interpolation):** Resamples the image by calculating the pixel area relation. It averages the intensity values of all source pixels that fall within the footprint of a single destination pixel.

#### When to Prefer `INTER_AREA`
`INTER_AREA` is the preferred and mathematically superior choice during **downsampling (shrinking)**.

| Metric / Aspect | `INTER_LINEAR` (Bilinear) for Downsampling | `INTER_AREA` (Area-based) for Downsampling |
|---|---|---|
| **Sampling Method** | Point-sampling over a localized $2 \times 2$ window. | Box-filtering/averaging over the entire pixel overlap area. |
| **Nyquist-Shannon Violation** | High. Skips intermediate pixels, leading to high-frequency aliasing. | Low. Naturally operates as a low-pass filter, respecting sampling bounds. |
| **Visual Artifacts** | Moiré patterns, jagged edges, pixel dropouts. | Smooth transitions, sharp legible structures, no aliasing. |
| **Best Use Case** | Zooming/Upsampling ($2 \times 2$ bilinear smoothing). | Shrinking/Downsampling (averaging area pixels). |

#### Why Bilinear Downsampling Fails
When downsampling a high-resolution image using bilinear interpolation, many original source pixels are completely ignored. This point-sampling approach violates the **Nyquist-Shannon sampling theorem**. Fine details, such as text, thin lines, or complex textures, turn into jagged, distorted patterns (aliasing).

`INTER_AREA` solves this by acting as a localized box filter. It averages all pixels within the source footprint that maps to a single destination pixel. This ensures that no pixel information is completely discarded, preserving color gradients, high-frequency structures, and edge integrity without creating artificial high-frequency noise.

*Note:* When **upsampling (enlarging)**, `INTER_AREA` behaves similarly to nearest-neighbor interpolation, resulting in pixelated, blocky, or stepped artifacts. For upsampling, bilinear (`INTER_LINEAR`) or bicubic (`INTER_CUBIC`) interpolations are preferred.

---

### Q4. You convert an image from BGR to HSV before processing. What does the HSV colour space represent that BGR does not?

The BGR and HSV color spaces represent image data under different coordinate systems. BGR is oriented around hardware reproduction, whereas HSV is modeled after human perception and feature separation.

#### The Problem with BGR
In the BGR color space, the three channels (Blue, Green, Red) are **highly correlated**. Chrominance (color) and luminance (brightness) are mixed across all three components.
- If the lighting condition or shadow across an object changes, the values of B, G, and R all shift simultaneously.
- Consequently, segmenting or isolating a specific colored object (e.g., a green surgical tool in an endoscopy video) based on BGR values is extremely fragile, as shadows and highlights distort the BGR range.

#### The Advantage of HSV
The HSV color space resolves this coupling by separating chromaticity from brightness, decomposing the signal into three independent components:

1. **Hue (H):** Represents the pure color type or dominant wavelength (e.g., Red, Yellow, Green, Cyan, Blue, Magenta). It corresponds to an angle on a color wheel ranging from $0^\circ$ to $360^\circ$.
   - *OpenCV mapping:* To fit Hue into an 8-bit unsigned integer (`uint8` max value 255), the angle is halved, mapping $[0, 360^\circ)$ to $[0, 180)$.
2. **Saturation (S):** Represents the purity, vibrancy, or shade of the color, ranging from 0 (muted grayscale/white) to 255 (fully saturated, vivid color).
3. **Value (V):** Represents the brightness or intensity of the light, ranging from 0 (pure black) to 255 (maximum brightness).

```
   HSV COLOR SPACE DECOUPLING:
   
          [Hue] (Color frequency/wavelength) ---> Decoupled from illumination
          
          [Saturation] (Purity/Vibrancy)    ---> Decoupled from illumination
          
          [Value] (Brightness/Intensity)    ---> Captures shadowing and highlights
```

#### Practical Applications in Computer Vision
Because Hue is isolated from Saturation and Value, HSV enables **robust color-based thresholding and segmentation**:
- **Illumination Invariance:** A green ball under a bright light and in a shadow will have widely different BGR values. However, in HSV, they will share almost the same **Hue** value; only the **Value** (brightness) and **Saturation** will shift.
- **Threshold Simplicity:** Engineers can isolate a specific color by setting a simple range on the Hue channel (e.g., `cv2.inRange(hsv, lower_bound, upper_bound)`), creating a mask that is highly resilient to shadows, spotlights, and dynamic lighting.

---

### Q5. You need to extract a specific region from a medical scan. What is the conceptual difference between cropping an image and resizing it?

In medical image preprocessing, extracting anatomical regions of interest must balance spatial resolution with data integrity. Cropping and resizing achieve spatially distinct results.

```
   CROPPING (Subsetting)
   +-----------------------+      +-------+
   |                       |      |       |  - Aspect Ratio preserved
   |     +-------+         | ---> |  ROI  |  - Pixel values unchanged
   |     |  ROI  |         |      |       |  - Physical scale preserved
   |     +-------+         |      +-------+
   |                       |
   +-----------------------+
   
   RESIZING (Resampling)
   +-----------------------+      +---------------+
   |                       |      |               |  - Scales coordinate grid
   |                       | ---> |  Resized Scan |  - Computes new pixel values
   |                       |      |               |  - Interpolation artifacts
   |                       |      +---------------+
   +-----------------------+
```

#### Cropping (Spatial Subsetting / Slicing)
- **Concept:** Cropping extracts a localized spatial subset (Region of Interest - ROI) from an image, discarding everything outside the defined bounding box.
- **Data Integrity:** The spatial resolution, pixel pitch, and aspect ratio of the extracted region are preserved **100% unchanged**. No new pixel values are synthesized, and no original pixels within the ROI are discarded or averaged.
- **Mathematical Operation:** In Python, this is executed using efficient array slicing:
  ```python
  cropped_roi = image[y_start:y_end, x_start:x_end]
  ```
- **Medical Context:** Cropping is used to isolate a specific physical structure (e.g., a tumor in an MRI or a single vertebra in an X-ray) without altering the scale or physical calibration of the pixels. The relationship between pixel distance and physical size (e.g., mm per pixel) remains completely accurate.

#### Resizing (Spatial Resampling / Interpolation)
- **Concept:** Resizing alters the total pixel grid dimensions of the image (or ROI), scaling it to fit a new width and height.
- **Data Integrity:** This process maps coordinates from the original grid to a new grid, recalculating pixel values using an interpolation kernel (such as bilinear or bicubic). It **loses information** during downsampling (aliasing/smoothing) and **synthesizes information** during upsampling (blurring/interpolation artifacts).
- **Mathematical Operation:** In OpenCV, this is executed via coordinate mapping:
  ```python
  resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
  ```
- **Medical Context:** Resizing is used to standardize heterogeneous scans from varying hospital scanners (each producing different image resolutions) to a uniform dimension (e.g., $512 \times 512$ pixels) so they can be processed by a convolutional neural network's fixed-size input layer.

---

---

## Section B: Practical Tasks

---

### B1. Task 1: Load multiple JPG and PNG images from the OpenCV samples dataset and validate their H x W x C structure.

This Python script programmatically downloads sample images from the official OpenCV raw repository, loads them using `cv2.imread()`, and validates their spatial and channel structure.

```python
import os
import urllib.request
import cv2

def download_and_validate_images():
    """
    Downloads sample BGR images from the OpenCV raw repository, loads them,
    and validates their Height x Width x Channels (H x W x C) structure.
    """
    # Create directory to store downloaded sample files
    output_dir = "opencv_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of images to fetch
    samples = {
        "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "messi5.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
        "sudoku.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
    }
    
    # Download images programmatically
    for filename, url in samples.items():
        save_path = os.path.join(output_dir, filename)
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, save_path)
            print(f"Successfully saved to {save_path}")
        else:
            print(f"Image {filename} already exists locally.")
            
    print("\n--- Validating Image H x W x C Structure ---")
    
    # Load and validate structural properties of each image
    for filename in samples.keys():
        file_path = os.path.join(output_dir, filename)
        
        # cv2.imread loads color images in BGR format with 3 channels by default
        img = cv2.imread(file_path)
        
        if img is None:
            print(f"Error: Failed to load image {filename}")
            continue
            
        # Get shape dimensions
        shape_info = img.shape
        height = shape_info[0]
        width = shape_info[1]
        
        # Check if the image has a third channel dimension (grayscale loaded as color has 3)
        channels = shape_info[2] if len(shape_info) == 3 else 1
        
        print(f"File Name   : {filename}")
        print(f"  Shape     : {shape_info}")
        print(f"  Height (H): {height} pixels (Rows)")
        print(f"  Width (W) : {width} pixels (Columns)")
        print(f"  Channels  : {channels} (Color Planes)")
        print(f"  Data Type : {img.dtype}")
        print(f"  Total Size: {img.size} elements")
        print("-" * 45)

if __name__ == "__main__":
    download_and_validate_images()
```

---

### B2. Task 2: Perform image resizing using INTER_LINEAR and INTER_CUBIC interpolation methods and visualize the pixel-level differences.

This script downsamples a sample image using Bilinear (`INTER_LINEAR`) and Bicubic (`INTER_CUBIC`) interpolations, calculates the pixel-level absolute difference using `cv2.absdiff()`, prints statistical metrics, and saves a side-by-side comparison plot.

```python
import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_analyze_differences():
    """
    Performs image resizing using INTER_LINEAR and INTER_CUBIC methods,
    quantifies the pixel-level differences, and generates a visual plot.
    """
    # Programmatic download of sample image
    image_dir = "opencv_samples"
    os.makedirs(image_dir, exist_ok=True)
    img_path = os.path.join(image_dir, "lena.jpg")
    
    if not os.path.exists(img_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        print("Downloading lena.jpg...")
        urllib.request.urlretrieve(url, img_path)
        
    # Read the original high-resolution image
    original = cv2.imread(img_path)
    if original is None:
        raise FileNotFoundError("Could not load original image.")
        
    print(f"Original image resolution: {original.shape}")
    
    # Define a downsampled target shape (e.g., 256x256, down from 512x512)
    target_dims = (256, 256)
    
    # Perform resizing using Bilinear Interpolation
    resized_linear = cv2.resize(original, target_dims, interpolation=cv2.INTER_LINEAR)
    
    # Perform resizing using Bicubic Interpolation (uses a 4x4 pixel neighborhood)
    resized_cubic = cv2.resize(original, target_dims, interpolation=cv2.INTER_CUBIC)
    
    # Calculate the pixel-level absolute difference
    # abs_diff(x, y) = |x - y|
    diff = cv2.absdiff(resized_linear, resized_cubic)
    
    # Extract statistics to quantify the difference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    max_diff = np.max(diff)
    
    print("\n--- Quantitative Interpolation Differences ---")
    print(f"Mean Absolute Difference per channel: {mean_diff:.5f}")
    print(f"Standard Deviation of Differences    : {std_diff:.5f}")
    print(f"Maximum Absolute Difference observed : {max_diff}")
    
    # Amplify the difference image for visualization clarity
    # Scales the minimum value to 0 and maximum value to 255
    amplified_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert BGR to RGB for correct display in Matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    linear_rgb = cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB)
    cubic_rgb = cv2.cvtColor(resized_cubic, cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(amplified_diff, cv2.COLOR_BGR2RGB)
    
    # Plotting using Matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original (512x512)")
    axes[0].axis("off")
    
    axes[1].imshow(linear_rgb)
    axes[1].set_title("INTER_LINEAR (256x256)")
    axes[1].axis("off")
    
    axes[2].imshow(cubic_rgb)
    axes[2].set_title("INTER_CUBIC (256x256)")
    axes[2].axis("off")
    
    axes[3].imshow(diff_rgb)
    axes[3].set_title("Abs Diff (Normalized)")
    axes[3].axis("off")
    
    plt.tight_layout()
    plot_save_path = os.path.join(image_dir, "interpolation_comparison.png")
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    
    print(f"\nVisual comparison successfully saved to: {plot_save_path}")

if __name__ == "__main__":
    resize_and_analyze_differences()
```

---

### B3. Task 3: Implement a 45-degree and 90-degree rotation using a computed transformation matrix without cropping the corners.

Standard rotation around an image center drops corner details because the bounding box dimensions remain fixed. This script dynamically calculates the new bounding box dimensions based on the rotation angle ($\theta$) and adjusts the affine transformation matrix's translation coordinates to prevent cropping.

```python
import os
import urllib.request
import cv2
import numpy as np

def rotate_image_no_crop(image, angle):
    """
    Rotates an image by the specified angle (in degrees) around its center,
    calculating a new bounding box to prevent corner cropping.
    """
    # Get original image dimensions
    h, w = image.shape[:2]
    
    # Convert angle to radians for trigonometric bounding box adjustments
    # Positive values in cv2.getRotationMatrix2D rotate counter-clockwise
    theta = np.radians(angle)
    cos_val = np.abs(np.cos(theta))
    sin_val = np.abs(np.sin(theta))
    
    # Calculate the new spatial bounds to encapsulate the rotated corners
    new_w = int((w * cos_val) + (h * sin_val))
    new_h = int((w * sin_val) + (h * cos_val))
    
    # Compute the standard rotation matrix centered at the original image's midpoint
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust translation factors in the 2x3 affine matrix:
    # M[0, 2] controls horizontal translation (T_x)
    # M[1, 2] controls vertical translation (T_y)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    
    # Perform warpAffine transformation
    # Using high-quality bicubic interpolation for the transformation
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Pad empty corners with pure black
    )
    return rotated

def run_rotation_pipeline():
    """Downloads sample image, rotates it 45 and 90 degrees without clipping."""
    image_dir = "opencv_samples"
    os.makedirs(image_dir, exist_ok=True)
    img_path = os.path.join(image_dir, "lena.jpg")
    
    if not os.path.exists(img_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        print("Downloading lena.jpg...")
        urllib.request.urlretrieve(url, img_path)
        
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load sample image.")
        
    print(f"Original shape: {img.shape}")
    
    # Execute 45-degree rotation
    rotated_45 = rotate_image_no_crop(img, 45)
    print(f"Rotated 45 degrees shape: {rotated_45.shape}")
    cv2.imwrite(os.path.join(image_dir, "lena_rotated_45.jpg"), rotated_45)
    
    # Execute 90-degree rotation
    rotated_90 = rotate_image_no_crop(img, 90)
    print(f"Rotated 90 degrees shape: {rotated_90.shape}")
    cv2.imwrite(os.path.join(image_dir, "lena_rotated_90.jpg"), rotated_90)
    
    print("\nRotations completed! Images saved to 'opencv_samples' folder.")

if __name__ == "__main__":
    run_rotation_pipeline()
```

---

### B4. Task 4: Execute a localized region-of-interest (ROI) crop and apply horizontal and vertical flips to the extracted segment.

This Python script extracts a specific Region of Interest (ROI) from an image and applies spatial flipping operations along the horizontal, vertical, and combined axes.

```python
import os
import urllib.request
import cv2

def crop_and_flip_roi():
    """
    Crops a localized Region of Interest (ROI) and performs horizontal,
    vertical, and double-axis flips on the extracted segment.
    """
    image_dir = "opencv_samples"
    os.makedirs(image_dir, exist_ok=True)
    img_path = os.path.join(image_dir, "lena.jpg")
    
    if not os.path.exists(img_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        print("Downloading lena.jpg...")
        urllib.request.urlretrieve(url, img_path)
        
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Could not load image.")
        
    # Define ROI bounding coordinates for localized crop (e.g. Lena's face area)
    # Coordinate indexing format: [y_start:y_end, x_start:x_end]
    y_start, y_end = 200, 360
    x_start, x_end = 200, 360
    
    # Slicing the numpy array crops the region with zero memory overhead
    roi = img[y_start:y_end, x_start:x_end]
    print(f"Original shape: {img.shape} | Extracted ROI shape: {roi.shape}")
    
    # Apply flipping operations:
    # cv2.flip(src, flipCode)
    # flipCode = 1  --> Horizontal flip (mirror along y-axis)
    # flipCode = 0  --> Vertical flip (mirror along x-axis)
    # flipCode = -1 --> Simultaneous horizontal and vertical flip (180 degree rotation)
    flip_horizontal = cv2.flip(roi, 1)
    flip_vertical = cv2.flip(roi, 0)
    flip_both = cv2.flip(roi, -1)
    
    # Save the output images
    cv2.imwrite(os.path.join(image_dir, "roi_original.jpg"), roi)
    cv2.imwrite(os.path.join(image_dir, "roi_flip_h.jpg"), flip_horizontal)
    cv2.imwrite(os.path.join(image_dir, "roi_flip_v.jpg"), flip_vertical)
    cv2.imwrite(os.path.join(image_dir, "roi_flip_hv.jpg"), flip_both)
    
    print("\nROI crops and flips saved successfully in the 'opencv_samples' folder.")

if __name__ == "__main__":
    crop_and_flip_roi()
```

---

### B5. Task 5: Develop a color-space conversion module to transform BGR source images into Grayscale and HSV formats for feature analysis.

This modular script converts BGR images to Grayscale and HSV formats and demonstrates feature analysis by thresholding skin tones using HSV color channels.

```python
import os
import urllib.request
import cv2
import numpy as np

class ImageColorConverter:
    """
    A modular utility class to handle color space transformations
    and color-based feature extraction.
    """
    def __init__(self, bgr_image):
        if bgr_image is None:
            raise ValueError("Input image cannot be None.")
        self.bgr = bgr_image

    def to_grayscale(self):
        """Converts BGR to 8-bit single-channel Grayscale."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    def to_hsv(self):
        """Converts BGR to 3-channel Hue-Saturation-Value."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

    def segment_color_range(self, lower_hsv, upper_hsv):
        """
        Creates a binary mask and segments a specific HSV range.
        Useful for tracking colored features or isolating regions.
        """
        hsv_image = self.to_hsv()
        # Create a binary mask where pixels inside bounds are 255, others are 0
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        # Apply mask using bitwise AND operation
        segmented = cv2.bitwise_and(self.bgr, self.bgr, mask=mask)
        return mask, segmented

def run_conversion_pipeline():
    """Executes color transformations and feature segmentations."""
    image_dir = "opencv_samples"
    os.makedirs(image_dir, exist_ok=True)
    img_path = os.path.join(image_dir, "lena.jpg")
    
    if not os.path.exists(img_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        print("Downloading lena.jpg...")
        urllib.request.urlretrieve(url, img_path)
        
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Could not load image.")
        
    # Initialize conversion module
    converter = ImageColorConverter(img)
    
    # 1. Transform to Grayscale
    gray = converter.to_grayscale()
    
    # 2. Transform to HSV
    hsv = converter.to_hsv()
    
    # 3. Extract skin-like features for demonstration
    # Define broad threshold boundaries for human skin tone in HSV color space
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    mask, segmented = converter.segment_color_range(lower_skin, upper_skin)
    
    # Save outputs
    cv2.imwrite(os.path.join(image_dir, "lena_gray.jpg"), gray)
    cv2.imwrite(os.path.join(image_dir, "lena_hsv.jpg"), hsv)
    cv2.imwrite(os.path.join(image_dir, "lena_skin_mask.jpg"), mask)
    cv2.imwrite(os.path.join(image_dir, "lena_skin_segmented.jpg"), segmented)
    
    print("\nColor transformations and segmentations completed.")
    print("Files saved to 'opencv_samples' folder:")
    print("  - Grayscale conversion: lena_gray.jpg")
    print("  - HSV conversion: lena_hsv.jpg")
    print("  - Binary skin mask: lena_skin_mask.jpg")
    print("  - Segmented skin features: lena_skin_segmented.jpg")

if __name__ == "__main__":
    run_conversion_pipeline()
```

---

---

## Section C: Mini Project

---

### Automated Medical Scan Normalizer

In clinical AI deployment, input scans are highly heterogeneous. Scans originate from different scanners, showing varied spatial resolutions, contrast ranges, and color configurations. 

This production-ready Python script standardizes input scans to a uniform structure:
1. **Programmatic Data Ingestion:** Downloads a sample scan surrogate.
2. **Color Space Standardization:** Standardizes BGR images to a single-channel grayscale layout.
3. **Resolution Standardization:** Resizes scans to a unified $512 \times 512$ footprint using **downsampling anti-aliasing** (`cv2.INTER_AREA`) or **upsampling bicubic interpolation** (`cv2.INTER_CUBIC`) based on source dimensions.
4. **Contrast Normalization (CLAHE):** Applies Contrast Limited Adaptive Histogram Equalization to normalize structural contrast without over-amplifying background noise.
5. **Intensity Normalization:** Rescales pixel intensities from $[0, 255]$ integers to $[0.0, 1.0]$ float32 values.

```python

# Automated Medical Scan Normalizer (production_normalizer.py)
"""
Automated Medical Scan Normalizer
---------------------------------
Purpose: Standardizing heterogeneous X-ray and MRI scans to a unified spatial 
         resolution, contrast range, and color depth for diagnostic AI systems.
"""

import os
import urllib.request
import cv2
import numpy as np

class AutomatedMedicalScanNormalizer:
    """
    Pipeline to ingest, standardise, and normalize diagnostic medical scans.
    """
    def __init__(self, target_resolution=(512, 512)):
        self.target_w, self.target_h = target_resolution
        
        # Initialize CLAHE filter (Contrast Limited Adaptive Histogram Equalization)
        # clipLimit=2.0 restricts noise amplification in uniform dark regions.
        # tileGridSize=(8, 8) divides the scan into an 8x8 grid for local enhancement.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def ingest_scan(self, path):
        """Loads scan from local disk, raising an exception if load fails."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to ingest scan from path: {path}")
        return img

    def normalize_pipeline(self, raw_img):
        """
        Runs the full normalization pipeline: Grayscale Conversion -> Dynamic 
        Resizing -> CLAHE Contrast Normalization -> Float32 Intensity Scaling.
        """
        # --- Step 1: Color Space Standardization ---
        # Medical scans (X-ray, CT, MRI) are single-channel grayscale signals.
        # Ensure any BGR formatting is converted to single-channel Grayscale.
        if len(raw_img.shape) == 3:
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw_img.copy()

        # --- Step 2: Resolution Standardization with Adaptive Interpolation ---
        h_orig, w_orig = gray.shape[:2]
        
        # Select interpolation mode dynamically based on scaling direction
        # If the input image is larger than the target, use INTER_AREA (averaging)
        # to avoid high-frequency aliasing.
        # If the input is smaller, use INTER_CUBIC (4x4 bicubic) for smooth upsampling.
        if h_orig > self.target_h or w_orig > self.target_w:
            interpolation = cv2.INTER_AREA
            interp_used = "INTER_AREA (Downsampling)"
        else:
            interpolation = cv2.INTER_CUBIC
            interp_used = "INTER_CUBIC (Upsampling)"

        resized = cv2.resize(
            gray, 
            (self.target_w, self.target_h), 
            interpolation=interpolation
        )

        # --- Step 3: Contrast Normalization ---
        # Standardize local contrast ranges using CLAHE to correct for varying 
        # illumination conditions across scanner hardware.
        contrast_enhanced = self.clahe.apply(resized)

        # --- Step 4: Intensity Normalization ---
        # Convert intensity integers [0, 255] to a standard [0.0, 1.0] float32 array.
        normalized = contrast_enhanced.astype(np.float32) / 255.0

        metadata = {
            "original_shape": (h_orig, w_orig),
            "target_shape": (self.target_h, self.target_w),
            "interpolation_mode": interp_used,
            "min_intensity": float(np.min(normalized)),
            "max_intensity": float(np.max(normalized)),
            "mean_intensity": float(np.mean(normalized))
        }

        return normalized, metadata

def execute_normalizer_run():
    """
    Downloads sample image, instantiates the normalizer pipeline,
    processes the image, and writes the output scan to the local disk.
    """
    input_dir = "opencv_samples"
    output_dir = "medical_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download sudoku.png (acting as an illustrative structure scan)
    sample_path = os.path.join(input_dir, "sudoku.png")
    if not os.path.exists(sample_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
        print(f"Downloading sample scan from: {url}")
        urllib.request.urlretrieve(url, sample_path)

    # Initialize Pipeline
    normalizer = AutomatedMedicalScanNormalizer(target_resolution=(512, 512))
    
    # Load and process the scan
    raw_scan = normalizer.ingest_scan(sample_path)
    processed_scan, meta = normalizer.normalize_pipeline(raw_scan)
    
    # Output metrics to console
    print("\n=== Medical Scan Normalizer Metadata ===")
    print(f"Original Resolution  : {meta['original_shape'][0]}x{meta['original_shape'][1]}")
    print(f"Standardized Bounds  : {meta['target_shape'][0]}x{meta['target_shape'][1]}")
    print(f"Interpolation Applied: {meta['interpolation_mode']}")
    print(f"Processed Intensity  : Range [{meta['min_intensity']:.4f}, {meta['max_intensity']:.4f}]")
    print(f"Mean Signal Value    : {meta['mean_intensity']:.4f}")
    
    # Convert float32 range [0.0, 1.0] back to [0, 255] uint8 for storage
    export_img = (processed_scan * 255.0).astype(np.uint8)
    save_filepath = os.path.join(output_dir, "normalized_scan.png")
    cv2.imwrite(save_filepath, export_img)
    print(f"Processed diagnostic scan saved to: {save_filepath}")

if __name__ == "__main__":
    execute_normalizer_run()

# TECHNICAL DESIGN REPORT: PREPROCESSING SELECTION IN DIAGNOSTIC SYSTEMS
#
# 1. CRITICAL CHOICE OF INTERPOLATION MECHANISMS
# ---------------------------------------------
# - Downsampling (Shrinking Scans): When clinical inputs are larger than the 
#   standard neural network tensor capacity (e.g., 2048x2048 high-res X-rays 
#   downsampled to 512x512), point-sampling methods like INTER_LINEAR or 
#   INTER_NEAREST skip intermediate pixel columns/rows. This leads to 
#   high-frequency aliasing (e.g., bone marrow textures appearing as artificial 
#   grid patterns). 
#   We use INTER_AREA because it acts as an integrated low-pass box filter, 
#   averaging pixel intensities within the shrinking grid footprint. This 
#   avoids aliasing, prevents micro-fracture line dropouts, and maintains 
#   structural diagnostics.
#
# - Upsampling (Enlarging Scans): When smaller scans must be stretched up to 
#   the standard size, INTER_AREA behaves like nearest-neighbor interpolation, 
#   creating blocky, pixelated, or stepped edges. 
#   We use INTER_CUBIC because it interpolates pixel values over a 4x4 spatial 
#   neighborhood using third-degree polynomials. This produces smoother color 
#   gradients and sharper anatomical boundaries compared to bilinear methods.
#
# 2. LOCAL CONTRAST ENHANCEMENT: CLAHE VS. GLOBAL HISTOGRAM EQUALIZATION
# ----------------------------------------------------------------------
# - Global Histogram Equalization applies a single mapping function across the 
#   entire image. This frequently over-amplifies background noise in uniform 
#   air/soft-tissue regions and washes out details in highly bright (dense bone) 
#   zones, compromising diagnostic safety.
#
# - Contrast Limited Adaptive Histogram Equalization (CLAHE) partitions the 
#   image into contextual grid tiles (8x8 pixels in this design). Histogram 
#   equalization is computed independently on each tile.
#   Importantly, a contrast limit (clipLimit=2.0) restricts the amplification of 
#   any single gray level. If a histogram bin exceeds this limit, the pixels 
#   are clipped and uniformly redistributed. This local control enhances subtle 
#   tissue variations (e.g., nodules, vascular lines) without introducing 
#   artificial contrast boundaries or boosting background scanner hum.
#
# 3. IMPACT OF INTENSITY NORMALIZATION ON DEEP LEARNING SYSTEMS
# -------------------------------------------------------------
# - Standardizing inputs to [0.0, 1.0] floating-point representation is a 
#   prerequisite for training stability in Deep Learning architectures:
#
#   a) Gradient Stability: Feeding unnormalized [0, 255] inputs causes large 
#      intermediate activation values. During backpropagation, this can lead 
#      to exploding gradients, destabilizing parameters in early layers.
#
#   b) Regularization Compatibility: Weight decay (L2 regularization) assumes 
#      uniform scaling. If pixel values are massive, the weight penalty's 
#      impact is neutralized, leading to rapid overfitting.
#
#   c) Optimizer Convergence: Optimization algorithms (e.g., Adam, SGD) 
#      traverse the loss landscape faster when input feature ranges are bounded, 
#      speeding up model convergence.
#
# =============================================================================
