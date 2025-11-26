# Image Processing Project

## Project Summary

A Python 3.11 object-oriented image processing application with a graphical user interface (GUI) that implements the operations and compression techniques listed in the project specification. All core algorithms are implemented from basic operations (filters, sums, min/max, median, etc.) rather than relying on single built-in "do-it-all" functions. The GUI groups related operations, supports image upload and live result display, and includes extra usability features.

## Features (implemented)

The application implements the full set of processes from the project brief. Each bullet below corresponds to a required process and a short note about the approach used.

Image I/O & Info

* Upload colored images from disk and display them in a side frame.
* Display image metadata: resolution (width × height), file size, color type.

Grayscale & Binary

* Convert RGB to grayscale (custom weighted average).
* Convert grayscale to binary using a threshold computed as average pixel intensity; includes an evaluation routine to test threshold quality and optional manual override.

Affine Transformations

* Translation, scaling, rotation, x-shear, y-shear implemented using homogeneous coordinates and custom sampling.

Interpolation (resolution enhancement)

* Resize using nearest-neighbor, bilinear, and bicubic interpolation (custom implementations).

Image Operations and ROI

* Cropping of user-selected region.
* First-derivative gradient filters and second-derivative Laplacian (custom convolution kernels).
* Sobel operator implemented from separable kernels.

Histogram Analysis & Enhancement

* Compute grayscale histogram and provide an assessment helper (e.g., under/over-exposed, low contrast) with justification.
* Histogram equalization implemented from histogram CDF.

Filtering

* Low-pass: 19×19 Gaussian filter with σ = 3 (kernel computed programmatically).
* Median: 7×7 median filter.
* High-pass: Laplacian and other high-frequency emphasizing filters.
  All filters are implemented using explicit neighborhood operations rather than black-box API calls.

Compression Techniques

* Implementations or prototypes for: Huffman, Golomb–Rice, Arithmetic coding (basic), LZW, Run-Length Encoding (RLE), symbol-based coding, bit-plane coding, block transform coding (DCT-based block compression), predictive coding, and a simple wavelet coding prototype. (Each technique includes encoder and decoder modules and a short demonstration.)

GUI & UX

* Clean, consistent UI design with grouped sections (Reading & Info, Geometric, Interpolation, Spatial Filters, Histogram & Enhancement, Compression).
* Buttons for each operation; clicking applies the process and displays the result dynamically.
* Side-by-side original vs processed comparison view and a small preview thumbnail strip for history.
* Additional creative conveniences: drag-to-select crop, slider controls for parameters (e.g., rotation angle, scaling factor, Gaussian σ), and save/export processed images.

Design & Implementation Notes

* Written in Python 3.11 using an object-oriented design (classes for ImageModel, Filter, Transform, Compressor, and GUIController).
* GUI library: assumed GTK6 (you wrote "gt6") — if you actually used PyQt5 or another toolkit, replace references accordingly. The code is modular so switching GUI backends requires only the GUI layer to be adapted.
* Heavy use of NumPy for array operations and file I/O via Pillow (PIL). Custom implementations avoid using built-in high-level image processing routines for the core algorithms.

## Project Structure

```
/project-root
├─ README.md
├─ requirements.txt
├─ main.py            # app launcher
├─ app/
│  ├─ gui.py          # GUI controller and layout
│  ├─ image_model.py  # ImageModel class (load/save/metadata)
│  ├─ transforms.py   # affine transforms & interpolation
│  ├─ filters.py      # low/high-pass, median, sobel, laplacian
│  ├─ histogram.py    # histogram, equalization, analysis
│  ├─ compression/
│  │  ├─ huffman.py
│  │  ├─ lzw.py
│  │  ├─ golomb_rice.py
│  │  ├─ arithmetic.py
│  │  ├─ rle.py
│  │  ├─ dct_block.py
│  │  ├─ wavelet.py
│  │  └─ bitplane.py
```

## Requirements

* Python 3.11
* NumPy
* Pillow (PIL)
* Matplotlib (for plots, optional)
* GUI toolkit: GTK6 (pygobject) or PyQt5 — adjust based on actual implementation
* (Optional) scipy for DCT/wavelet utilities if used for prototypes

Example `requirements.txt` entries:

```
attr==0.3.2
charset_normalizer==3.3.2
ConfigParser==7.2.0
cryptography==46.0.3
Cython==3.2.1
dl==0.1.0
docutils==0.22.3
filelock==3.20.0
HTMLParser==0.0.2
hypothesis==6.148.2
ipython==9.7.0
ipywidgets==8.1.8
Jinja2==3.1.6
jnius==1.1.0
keyring==25.6.0
mtrand==0.1
mypy==1.18.2
numarray==1.5.1
Numeric==24.2
pickle5==0.0.12
protobuf==6.33.1
psutil==7.1.3
PyInstaller==6.17.0
pyOpenSSL==25.3.0
pytest==9.0.1
pytz==2025.2
PyYAML==6.0.3
railroad==0.5.0
redis==7.1.0
scipy_doctest==2.0.1
Sphinx==8.2.3
thread==2.0.5
threadpoolctl==3.6.0
trove_classifiers==2025.11.14.15
urllib3_secure_extra==0.1.0
xmlrpclib==1.0.1
```

## How to run

1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  (or venv\Scripts\activate on Windows)
2. Install dependencies:
   pip install -r requirements.txt
3. Launch the app:
   python gui.py
   The GUI will open; click "Upload Image" or drag and drop to start and use grouped buttons to apply operations. Results show dynamically in the right pane, with options to save.

## Examples and Usage

* Load `examples/lena.png`, click "Grayscale" then "Histogram" to view the histogram and an automated assessment.
* Use "Resize (Bicubic)" with a factor of 2 to demonstrate bicubic interpolation.
* Try "Compress -> Huffman" to encode and then decode; compare original vs reconstructed PSNR printed in the status bar.

## Grading & Deliverables Notes

* The README documents which member implemented which module (add names/IDs in the Team section).
* Include answers to project questions for each team member in `docs/` or append as `TEAM_QUESTIONS.md` (this portion is graded separately).
* Part of the grade depends on the correctness of algorithmic implementations and another on the GUI usability and aesthetics.
