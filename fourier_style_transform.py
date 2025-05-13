import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ------------------- Utility Functions -------------------

def load_image(image_path, size=None):
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
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) / 255.0

def postprocess_image(img):
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def is_grayscale(img):
    return np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2])

def visualize_spectrum(fft_img, log_scale=True):
    spectrum = np.mean(np.abs(fft_img), axis=2)
    if log_scale:
        spectrum = np.log(spectrum + 1e-10)
    eps = 1e-10
    spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + eps)
    return (spectrum * 255).astype(np.uint8)

# ------------------- Fourier Domain Functions -------------------

def fourier_transform(img):
    fft_img = np.zeros_like(img, dtype=np.complex64)
    for i in range(3):
        fft_img[:, :, i] = np.fft.fftshift(np.fft.fft2(img[:, :, i]))
    return fft_img

def inverse_fourier_transform(fft_img):
    img = np.zeros_like(fft_img, dtype=np.float32)
    for i in range(3):
        img[:, :, i] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, i])))
    return img

# ------------------- Style Transfer Modes -------------------

def amplitude_only_transfer(content_fft, style_fft, alpha):
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:, :, i])
        content_phase = np.angle(content_fft[:, :, i])
        style_amp = np.abs(style_fft[:, :, i])
        blended_amp = (1 - alpha) * content_amp + alpha * style_amp
        result_fft[:, :, i] = blended_amp * np.exp(1j * content_phase)
    return result_fft

def phase_only_transfer(content_fft, style_fft, beta):
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:, :, i])
        content_phase = np.angle(content_fft[:, :, i])
        style_phase = np.angle(style_fft[:, :, i])
        blended_phase = (1 - beta) * content_phase + beta * style_phase
        result_fft[:, :, i] = content_amp * np.exp(1j * blended_phase)
    return result_fft

def combined_transfer(content_fft, style_fft, alpha, beta):
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:, :, i])
        content_phase = np.angle(content_fft[:, :, i])
        style_amp = np.abs(style_fft[:, :, i])
        style_phase = np.angle(style_fft[:, :, i])
        
        blended_amp = (1 - alpha) * content_amp + alpha * style_amp
        blended_phase = (1 - beta) * content_phase + beta * style_phase
        result_fft[:, :, i] = blended_amp * np.exp(1j * blended_phase)
    return result_fft

def color_transfer_lab(content_img, style_img, strength=0.5):
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

def fourier_color_transfer_lab(content_img, style_img, alpha=0.5):
    """
    Transfer color from style to content using Fourier transform on LAB a/b channels.
    
    Parameters:
    - content_img: RGB image in float32 [0,1]
    - style_img: RGB image in float32 [0,1]
    - alpha: Style strength (0 = content only, 1 = style only)
    
    Returns:
    - RGB image with LAB-based color transfer via Fourier
    """
    # Convert to LAB color space
    content_lab = cv2.cvtColor((content_img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    style_lab = cv2.cvtColor((style_img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    result_lab = np.copy(content_lab)

    # Keep L (lightness) from content — for structure preservation
    result_lab[:,:,0] = content_lab[:,:,0]

    for ch in [1, 2]:  # a and b channels
        content_fft = np.fft.fft2(content_lab[:,:,ch])
        style_fft = np.fft.fft2(style_lab[:,:,ch])

        content_amp = np.abs(content_fft)
        content_phase = np.angle(content_fft)
        style_amp = np.abs(style_fft)

        # Blend amplitude spectrum
        blended_amp = (1 - alpha) * content_amp + alpha * style_amp

        # Reconstruct with content phase and blended amplitude
        result_fft = blended_amp * np.exp(1j * content_phase)
        result_lab[:,:,ch] = np.real(np.fft.ifft2(result_fft))

    # Clip and convert back to RGB
    result_lab[:,:,1:] = np.clip(result_lab[:,:,1:], 0, 255)
    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return result_rgb.astype(np.float32) / 255.0

# ------------------- Style Transfer Dispatcher -------------------

def fourier_style_transfer(content_img, style_img, method='amplitude', alpha=0.5, beta=0.2):
    print("Preprocessing images...")
    content_img = preprocess_image(content_img)
    style_img = preprocess_image(style_img)

    print("Applying Fourier transforms...")
    content_fft = fourier_transform(content_img)
    style_fft = fourier_transform(style_img)

    print(f"Transferring style using method: {method}")
    if method == 'color_lab':
        # Use LAB color space transfer
        result_img = color_transfer_lab(content_img, style_img, alpha=alpha)
        # Compute FFT of result for visualization
        result_fft = fourier_transform(result_img)
    elif method == 'fourier_color_lab':
        # Use LAB color space fourier transfer
        result_img = fourier_color_transfer_lab(content_img, style_img, alpha=alpha)
        # Compute FFT of result for visualization
        result_fft = fourier_transform(result_img)
    else:
        # Apply Fourier transforms-based style transfer
        if method == 'amplitude':
            result_fft = amplitude_only_transfer(content_fft, style_fft, alpha)
        elif method == 'phase':
            result_fft = phase_only_transfer(content_fft, style_fft, beta)
        elif method == 'combined':
            temp_fft = amplitude_only_transfer(content_fft, style_fft, alpha)
            result_fft = phase_only_transfer(temp_fft, style_fft, beta)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        result_img = inverse_fourier_transform(result_fft)

    return postprocess_image(result_img), content_fft, style_fft, result_fft

# ------------------- Visualization -------------------

def plot_all(content_img, style_img, result_img, content_fft, style_fft, result_fft, method, plot_all=False):
    if plot_all:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(content_img)
        axes[0, 0].set_title('Content Image')
        axes[0, 1].imshow(style_img)
        axes[0, 1].set_title('Style Image')
        axes[0, 2].imshow(result_img)
        axes[0, 2].set_title(f'Result ({method})')

        axes[1, 0].imshow(visualize_spectrum(content_fft), cmap='viridis')
        axes[1, 0].set_title('Content Spectrum')
        axes[1, 1].imshow(visualize_spectrum(style_fft), cmap='viridis')
        axes[1, 1].set_title('Style Spectrum')
        axes[1, 2].imshow(visualize_spectrum(result_fft), cmap='viridis')
        axes[1, 2].set_title('Result Spectrum')
    
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(content_img)
        axes[0].set_title('Content Image')
        axes[1].imshow(style_img)
        axes[1].set_title('Style Image')
        axes[2].imshow(result_img)
        axes[2].set_title(f'Result ({method})')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
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

def run_style_transfer(content_path, style_path, size=(512, 512), method='combined', alpha=0.7, beta=0.2):
    content_img = load_image(content_path, size=size)
    style_img = load_image(style_path, size=size)

    result_img, content_fft, style_fft, result_fft = fourier_style_transfer(
        content_img,
        style_img,
        method=method,
        alpha=alpha,
        beta=beta
    )

    plot_all(content_img, style_img, result_img, content_fft, style_fft, result_fft, method, plot_all=True)

    # Plot detailed spectra for each channel
    plot_channel_spectra(content_fft, "Content Image Frequency Spectrum")
    plot_channel_spectra(style_fft, "Style Image Frequency Spectrum")
    plot_channel_spectra(result_fft, f"Result ({method}) Frequency Spectrum")

    return result_img

# Run example
content_path = './cm.png'
style_path = './filter.jpg'

result = run_style_transfer(
    content_path,
    style_path,
    size=(512, 512),
    method='color',  # Options: 'amplitude', 'phase', 'combined', 'color_lab', 'fourier_color_lab'
    alpha=0.7,
    beta=0.1
)
