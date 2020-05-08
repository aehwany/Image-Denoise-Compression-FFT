# Image DTFT, Denoising & Compression Script
 
## Usage
```python fft.py [-m mode] [-i image]```

* mode (optional) [int]:
    - [1] (Default) for fast mode where the image is converted into its FFT form and displayed
    - [2] for denoising where the image is denoised by applying an FFT, truncating high
    frequencies and then displayed
    - [3] for compressing and saving the image
    - [4] for plotting the runtime graphs for the report

* image (optional) [string]: 
    - filename of the image we wish to take the DFT of. (Default: './moonlanding.png')