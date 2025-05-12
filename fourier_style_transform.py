import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ------------------- Utility Functions -------------------

def load_image(image_path, size=None):
    """
    Load an image from the specified path and convert it to RGB.
    
    Parameters:
    - image_path: Path to the image file
    - size: Optional tuple (width, height) to resize the image
    
    Returns:
    - RGB image as a numpy array
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size is not None:
        img = cv2.resize(img, size)
    
    return img

def preprocess_image(img):
    """
    Preprocess an image for Fourier transform.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Preprocessed image in range [0, 1]
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_float = img.astype(np.float32) / 255.0
    return img_float

def fourier_transform(img):
    """
    Apply Fourier transform to each channel of an image.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Fourier transform of the image (complex array)
    """
    fft_img = np.zeros_like(img, dtype=np.complex64)
    for i in range(3):
        fft_img[:,:,i] = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
    return fft_img

def inverse_fourier_transform(fft_img):
    """
    Apply inverse Fourier transform to get back to the spatial domain.
    
    Parameters:
    - fft_img: Fourier transform of an image
    
    Returns:
    - Image in spatial domain
    """
    img = np.zeros_like(fft_img, dtype=np.float32)
    for i in range(3):
        img[:,:,i] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,i])))
    return img

def amplitude_style_transfer(content_fft, style_fft, alpha=0.5):
    """
    Transfer the amplitude spectrum from style to content.
    
    Parameters:
    - content_fft: Fourier transform of content image
    - style_fft: Fourier transform of style image
    - alpha: Blending factor for amplitude (0-1)
    
    Returns:
    - Fourier transform with blended amplitude
    """
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:,:,i])
        content_phase = np.angle(content_fft[:,:,i])
        style_amp = np.abs(style_fft[:,:,i])
        blended_amp = (1 - alpha) * content_amp + alpha * style_amp
        result_fft[:,:,i] = blended_amp * np.exp(1j * content_phase)
    return result_fft

def phase_style_transfer(content_fft, style_fft, beta=0.2):
    """
    Transfer the phase spectrum from style to content.
    
    Parameters:
    - content_fft: Fourier transform of content image
    - style_fft: Fourier transform of style image
    - beta: Blending factor for phase (0-1)
    
    Returns:
    - Fourier transform with blended phase
    """
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:,:,i])
        content_phase = np.angle(content_fft[:,:,i])
        style_phase = np.angle(style_fft[:,:,i])
        blended_phase = (1 - beta) * content_phase + beta * style_phase
        result_fft[:,:,i] = content_amp * np.exp(1j * blended_phase)
    return result_fft

def postprocess_image(img):
    """
    Convert image back to 8-bit format after processing.
    
    Parameters:
    - img: Input image in range [0, 1]
    
    Returns:
    - Image in range [0, 255] as uint8
    """
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def visualize_spectrum(fft_img, log_scale=True, color_channels=False):
    """
    Visualize the frequency spectrum of an image.
    
    Parameters:
    - fft_img: Fourier transform of an image
    - log_scale: Whether to use log scale for better visualization
    - color_channels: If True, visualize each RGB channel separately
    
    Returns:
    - Visualization of the frequency spectrum
    """
    if not color_channels:
        # Visualize each RGB channel separately
        vis_spectra = []
        for i in range(3):
            channel_spectrum = np.abs(fft_img[:,:,i])
            if log_scale:
                channel_spectrum = np.log(channel_spectrum + 1e-10)
            eps = 1e-10
            channel_spectrum = (channel_spectrum - channel_spectrum.min()) / (channel_spectrum.max() - channel_spectrum.min() + eps)
            vis_spectra.append((channel_spectrum * 255).astype(np.uint8))
            
        # Create a colored visualization by putting each channel in RGB
        colored_spectrum = np.zeros((fft_img.shape[0], fft_img.shape[1], 3), dtype=np.uint8)
        colored_spectrum[:,:,0] = vis_spectra[0]  # R
        colored_spectrum[:,:,1] = vis_spectra[1]  # G
        colored_spectrum[:,:,2] = vis_spectra[2]  # B
        
        return colored_spectrum
    else:
        # Average across channels (grayscale visualization)
        spectrum = np.mean(np.abs(fft_img), axis=2)
        if log_scale:
            spectrum = np.log(spectrum + 1e-10)
        eps = 1e-10
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + eps)
        return (spectrum * 255).astype(np.uint8)

def is_grayscale(img):
    """
    Check if an image is grayscale.
    
    Parameters:
    - img: Input image
    
    Returns:
    - True if grayscale, False otherwise
    """
    return np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2])

# ------------------- Improved Color Transfer Functions -------------------

def match_histograms(source, reference):
    """
    Adjust the pixel values of source to match the histogram of reference.
    Works on each RGB channel independently.
    
    Parameters:
    - source: Source image
    - reference: Reference image
    
    Returns:
    - Image with matched histograms
    """
    result = np.zeros_like(source)
    
    for i in range(3):  # For each RGB channel
        src_values = source[:,:,i].flatten()
        ref_values = reference[:,:,i].flatten()
        
        # Get the set of unique pixel values and their indices in source
        s_values, bin_idx, s_counts = np.unique(src_values, return_inverse=True, return_counts=True)
        
        # Get the set of unique pixel values and their counts in reference
        r_values, r_counts = np.unique(ref_values, return_counts=True)
        
        # Calculate the normalized cumulative histograms
        s_quantiles = np.cumsum(s_counts) / s_counts.sum()
        r_quantiles = np.cumsum(r_counts) / r_counts.sum()
        
        # Map the source values to reference values based on quantiles
        interp_values = np.interp(s_quantiles, r_quantiles, r_values)
        
        # Apply the mapping to source pixels
        result[:,:,i] = interp_values[bin_idx].reshape(source[:,:,i].shape)
    
    return result

def color_transfer(content_img, style_img, strength=0.5):
    """
    Transfer colors from style to content using LAB color space.
    
    Parameters:
    - content_img: Content image
    - style_img: Style image
    - strength: Strength of color transfer (0-1)
    
    Returns:
    - Image with transferred colors
    """
    # Convert from RGB to LAB color space
    content_lab = cv2.cvtColor((content_img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    style_lab = cv2.cvtColor((style_img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Compute mean and standard deviation of each channel
    content_mean = np.mean(content_lab, axis=(0, 1))
    content_std = np.std(content_lab, axis=(0, 1))
    style_mean = np.mean(style_lab, axis=(0, 1))
    style_std = np.std(style_lab, axis=(0, 1))
    
    # Apply the transformation
    result_lab = np.zeros_like(content_lab)
    
    # For L channel (luminance), use less style influence to preserve content structure
    result_lab[:,:,0] = content_lab[:,:,0]  # Keep L channel (luminance) from content
    
    # For a and b channels (color), transfer the style with given strength
    for i in range(1, 3):
        result_lab[:,:,i] = ((content_lab[:,:,i] - content_mean[i]) * 
                            (style_std[i] / (content_std[i] + 1e-6)) * strength + 
                            (1 - strength) * content_lab[:,:,i] +
                            style_mean[i] * strength)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return result_rgb.astype(np.float32) / 255.0

def fourier_style_transfer(content_img, style_img, method='combined', alpha=0.5, beta=0.2, gamma=None):
    """
    Apply style transfer using Fourier transforms and various methods.
    
    Parameters:
    - content_img: Content image
    - style_img: Style image
    - method: Style transfer method ('amplitude', 'phase', 'combined', 'color', 'histogram')
    - alpha: Blending factor for amplitude (0-1)
    - beta: Blending factor for phase (0-1)
    - gamma: Grayscale blending factor (0-1), None to disable
    
    Returns:
    - Processed image, content FFT, style FFT, result FFT
    """
    print("Preprocessing images...")
    content_img = preprocess_image(content_img)
    style_img = preprocess_image(style_img)

    print(f"Transferring style using method: {method}")
    
    # Always compute the Fourier transforms for visualization
    content_fft = fourier_transform(content_img)
    style_fft = fourier_transform(style_img)
    
    if method == 'color':
        # Use LAB color space transfer
        result_img = color_transfer(content_img, style_img, strength=alpha)
        # Compute FFT of result for visualization
        result_fft = fourier_transform(result_img)
        
    elif method == 'histogram':
        # Use histogram matching
        result_img = match_histograms(content_img, style_img)
        # Compute FFT of result for visualization
        result_fft = fourier_transform(result_img)
        
    else:
        # Apply Fourier transforms-based style transfer
        if method == 'amplitude':
            result_fft = amplitude_style_transfer(content_fft, style_fft, alpha)
        elif method == 'phase':
            result_fft = phase_style_transfer(content_fft, style_fft, beta)
        elif method == 'combined':
            temp_fft = amplitude_style_transfer(content_fft, style_fft, alpha)
            result_fft = phase_style_transfer(temp_fft, style_fft, beta)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        result_img = inverse_fourier_transform(result_fft)

    if gamma is not None:
        print(f"Blending color and grayscale with gamma={gamma}...")
        gray = cv2.cvtColor((result_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_rgb = np.stack([gray]*3, axis=-1)
        result_img = (1 - gamma) * result_img + gamma * gray_rgb
        # Update result FFT after grayscale blending
        result_fft = fourier_transform(result_img)

    return postprocess_image(result_img), content_fft, style_fft, result_fft

# ------------------- Enhanced Visualization -------------------

def plot_all(content_img, style_img, result_img, content_fft, style_fft, result_fft, method, plot_spectra=True, color_spectra=True):
    """
    Plot images and their frequency spectra.
    
    Parameters:
    - content_img, style_img, result_img: Original images
    - content_fft, style_fft, result_fft: Fourier transforms
    - method: Style transfer method used
    - plot_spectra: Whether to plot frequency spectra
    - color_spectra: Whether to visualize spectra in color
    """
    if plot_spectra:
        # Create a 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot original images in the first row
        axes[0, 0].imshow(content_img)
        axes[0, 0].set_title('Content Image')
        axes[0, 1].imshow(style_img)
        axes[0, 1].set_title('Style Image')
        axes[0, 2].imshow(result_img)
        axes[0, 2].set_title(f'Result ({method})')
        
        
        # Plot frequency spectra in the second row
        if color_spectra:
            # Color visualization of frequency spectra
            content_spectrum = visualize_spectrum(content_fft, log_scale=True, color_channels=True)
            style_spectrum = visualize_spectrum(style_fft, log_scale=True, color_channels=True)
            result_spectrum = visualize_spectrum(result_fft, log_scale=True, color_channels=True)
            
            axes[1, 0].imshow(content_spectrum)
            axes[1, 0].set_title('Content Spectrum (RGB)')
            axes[1, 1].imshow(style_spectrum)
            axes[1, 1].set_title('Style Spectrum (RGB)')
            axes[1, 2].imshow(result_spectrum)
            axes[1, 2].set_title('Result Spectrum (RGB)')
        else:
            # Grayscale visualization of frequency spectra
            content_spectrum = visualize_spectrum(content_fft, log_scale=True)
            style_spectrum = visualize_spectrum(style_fft, log_scale=True)
            result_spectrum = visualize_spectrum(result_fft, log_scale=True)
            
            axes[1, 0].imshow(content_spectrum, cmap='viridis')
            axes[1, 0].set_title('Content Spectrum')
            axes[1, 1].imshow(style_spectrum, cmap='viridis')
            axes[1, 1].set_title('Style Spectrum')
            axes[1, 2].imshow(result_spectrum, cmap='viridis')
            axes[1, 2].set_title('Result Spectrum')
    else:
        # Just plot the images without spectra
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(content_img)
        axes[0].set_title('Content Image')
        axes[1].imshow(style_img)
        axes[1].set_title('Style Image')
        axes[2].imshow(result_img)
        axes[2].set_title(f'Result ({method})')
    
    # Turn off axis for all subplots
    for ax in axes.ravel():
        ax.axis('off')
    
    plt.tight_layout()
    # plt.savefig(f"result_{method}_with_spectra.png", dpi=150)
    plt.show()

# ------------------- Additional Spectrum Visualization -------------------

def plot_channel_spectra(fft_img, title="Frequency Spectrum by Channel"):
    """
    Plot the frequency spectrum of each RGB channel separately.
    
    Parameters:
    - fft_img: Fourier transform of an image
    - title: Title of the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        channel_spectrum = np.abs(fft_img[:,:,i])
        log_spectrum = np.log(channel_spectrum + 1e-10)
        normalized = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-10)
        
        im = axes[i].imshow(normalized, cmap='inferno')
        axes[i].set_title(f'{channel_names[i]} Channel')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

# ------------------- Example Run -------------------

def run_style_transfer(content_path, style_path, size=(512, 512), method='color', alpha=0.5, beta=0.2, gamma=None):
    """
    Run the style transfer pipeline.
    
    Parameters:
    - content_path: Path to content image
    - style_path: Path to style image
    - size: Size to resize images to
    - method: Style transfer method ('amplitude', 'phase', 'combined', 'color', 'histogram')
    - alpha: Blending factor for amplitude (0-1) or color strength
    - beta: Blending factor for phase (0-1)
    - gamma: Grayscale blending factor (0-1), None to disable
    
    Returns:
    - Processed image
    """
    # Ensure the paths are valid
    content_path = Path(content_path)
    style_path = Path(style_path)
    
    if not content_path.exists():
        raise FileNotFoundError(f"Content image not found at {content_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"Style image not found at {style_path}")
        
    print(f"Loading images from {content_path} and {style_path}")
    content_img = load_image(str(content_path), size=size)
    style_img = load_image(str(style_path), size=size)

    result_img, content_fft, style_fft, result_fft = fourier_style_transfer(
        content_img, 
        style_img, 
        method=method, 
        alpha=alpha, 
        beta=beta,
        gamma=gamma
    )

    # Plot regular comparison with spectra
    plot_all(content_img, style_img, result_img, content_fft, style_fft, result_fft, method, plot_spectra=True, color_spectra=True)
    
    # Plot detailed spectra for each channel
    plot_channel_spectra(content_fft, "Content Image Frequency Spectrum")
    plot_channel_spectra(style_fft, "Style Image Frequency Spectrum")
    plot_channel_spectra(result_fft, f"Result ({method}) Frequency Spectrum")

    
    # Save result image
    result_path = f"result_{method}.png"
    result_pil = Image.fromarray(result_img)
    result_pil.save(result_path)
    print(f"Result saved as {result_path}")

    return result_img

def main():
    """
    Main function to run the style transfer with command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Fourier-based style transfer')
    parser.add_argument('content_path', type=str, help='Path to content image')
    parser.add_argument('style_path', type=str, help='Path to style image')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512], help='Size to resize images to (width height)')
    parser.add_argument('--method', type=str, default='color', choices=['amplitude', 'phase', 'combined', 'color', 'histogram'], 
                        help='Style transfer method')
    parser.add_argument('--alpha', type=float, default=0.5, help='Blending factor for amplitude or color strength (0-1)')
    parser.add_argument('--beta', type=float, default=0.2, help='Blending factor for phase (0-1)')
    parser.add_argument('--gamma', type=float, default=None, help='Grayscale blending factor (0-1), None to disable')
    
    args = parser.parse_args()
    
    run_style_transfer(
        args.content_path, 
        args.style_path, 
        size=tuple(args.size), 
        method=args.method, 
        alpha=args.alpha, 
        beta=args.beta,
        gamma=args.gamma
    )

# Main execution
if __name__ == "__main__":
    # If no arguments are provided, use default values
    import sys
    if len(sys.argv) == 1:
        content_path = './cm.png'
        style_path = './no_filter.jpg'
        
        # Method 1: LAB color transfer (often gives most natural results)
        result1 = run_style_transfer(
            content_path, 
            style_path, 
            size=(512, 512),
            method='color',
            alpha=0.7  # Strength of color transfer (0.5-0.8 works well)
        )
        
        # Method 2: Fourier amplitude transfer
        # result2 = run_style_transfer(
        #     content_path, 
        #     style_path, 
        #     size=(512, 512),
        #     method='amplitude',
        #     alpha=0.5  # Strength of amplitude transfer
        # )
        
        # # Method 3: Combined amplitude and phase transfer
        # result3 = run_style_transfer(
        #     content_path, 
        #     style_path, 
        #     size=(512, 512),
        #     method='combined',
        #     alpha=0.5,  # Strength of amplitude transfer
        #     beta=0.2    # Strength of phase transfer
        # )
    else:
        main()