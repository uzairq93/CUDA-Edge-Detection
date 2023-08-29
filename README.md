### Readme.md for Image Processing Experiment Suite
#### Contributors
- [Uzair Qadir](mailto:uzairqadir2020@u.northwestern.edu)
- [Tanmeet Butani](mailto:tanmeetbutani2024@u.northwestern.edu)

#### Course
- Massively Parallel Programs in CUDA: [CS 468]
#### Introduction
This repository contains our final report in PDF form and a set of image processing operations such as Sobel Edge Detection to benchmark the performance difference between CPU and GPU implementations. By executing a simple `Makefile`, you can run various experiments to assess the efficiency of each operation. 

#### Pre-requisites
- C/C++ Compiler
- CUDA Toolkit

#### Installation and Execution
1. Download the tarball and extract it.
2. Navigate to the directory and run `make`. This generates an executable called `executable`.
3. Run an operation with the following command:
    ```
    ./executable <operation>
    ```
#### Operations
The following operations can be performed:
- `brighten [val]`: Increases the brightness level by a specified integer value.
  - Example: `./executable brighten 60`
- `grayscale`: Converts the image to grayscale.
  - Example: `./executable grayscale`
- `dilate`: Applies dilation to the image.
- `erode`: Applies erosion to the image.
- `gauss`: Applies Gaussian blur to the image.
- `sobel`: Applies Sobel filter to the image.



---

