import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import io
import matplotlib.pyplot as plt

# ------------------- Utility Functions -------------------

def load_image(image_path, size=None):
    """Load an image from a file path or bytes"""
    if isinstance(image_path, (str, bytes)):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        # For PIL Image objects
        img = np.array(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if img is None:
        raise ValueError(f"Could not load image")
    
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
    """Apply Gaussian blur and normalize pixel values"""
    img = cv2.GaussianBlur(img, (3,3), 0)
    img_float = img.astype(np.float32) / 255.0
    return img_float

def fourier_transform(img):
    """Apply 2D Fourier transform to each RGB channel"""
    fft_img = np.zeros_like(img, dtype=np.complex64)
    for i in range(3):
        fft_img[:,:,i] = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
    return fft_img

def inverse_fourier_transform(fft_img):
    """Apply inverse 2D Fourier transform to each RGB channel"""
    img = np.zeros_like(fft_img, dtype=np.float32)
    for i in range(3):
        img[:,:,i] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,i])))
    return img

def amplitude_style_transfer(content_fft, style_fft, alpha=0.5):
    """Transfer the amplitude (magnitude) spectrum from style to content"""
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:,:,i])
        content_phase = np.angle(content_fft[:,:,i])
        style_amp = np.abs(style_fft[:,:,i])
        blended_amp = (1 - alpha) * content_amp + alpha * style_amp
        result_fft[:,:,i] = blended_amp * np.exp(1j * content_phase)
    return result_fft

def phase_style_transfer(content_fft, style_fft, beta=0.2):
    """Transfer the phase spectrum from style to content"""
    result_fft = np.zeros_like(content_fft, dtype=complex)
    for i in range(3):
        content_amp = np.abs(content_fft[:,:,i])
        content_phase = np.angle(content_fft[:,:,i])
        style_phase = np.angle(style_fft[:,:,i])
        blended_phase = (1 - beta) * content_phase + beta * style_phase
        result_fft[:,:,i] = content_amp * np.exp(1j * blended_phase)
    return result_fft

def postprocess_image(img):
    """Clip values to [0,1] range and convert to uint8"""
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
    if color_channels:
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

# ------------------- Color Transfer Functions -------------------

def match_histograms(source, reference):
    """
    Adjust the pixel values of source to match the histogram of reference.
    Works on each RGB channel independently.
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
    Transfer colors from style to content using LAB color space
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

# ------------------- Main Style Transfer Function -------------------

def fourier_style_transfer(content_img, style_img, method='combined', alpha=0.5, beta=0.2, gamma=None):
    """
    Apply style transfer using Fourier transforms and various methods
    """
    # Preprocess images
    content_img = preprocess_image(content_img)
    style_img = preprocess_image(style_img)

    # Compute Fourier transforms
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

    if gamma is not None and gamma > 0:
        # Apply grayscale blending if gamma > 0
        gray = cv2.cvtColor((result_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_rgb = np.stack([gray]*3, axis=-1)
        result_img = (1 - gamma) * result_img + gamma * gray_rgb
        # Update result FFT after grayscale blending
        result_fft = fourier_transform(result_img)

    return postprocess_image(result_img), content_fft, style_fft, result_fft

# ------------------- Streamlit App -------------------

def create_plot_image(content_img, style_img, result_img, content_fft, style_fft, result_fft, 
                     method, show_spectra=True, color_spectra=True):
    """Create matplotlib figure and convert to image for Streamlit display"""
    
    if show_spectra:
        # Create a 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Plot original images in the first row
        axes[0, 0].imshow(content_img)
        axes[0, 0].set_title('Content Image')
        axes[0, 1].imshow(style_img)
        axes[0, 1].set_title('Style Image')
        axes[0, 2].imshow(result_img)
        axes[0, 2].set_title(f'Result ({method})')
        
        # Plot frequency spectra in the second row
        if not color_spectra:
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
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
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
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# ------------------- Streamlit App Main -------------------

def main():
    st.set_page_config(page_title="Fourier Style Transfer", layout="wide")
    
    st.title("🎨 Fourier Style Transfer")
    st.write("""
    This app applies style transfer using Fourier transforms. Upload a content image 
    and a style image, then adjust parameters to create your stylized image.
    """)
    
    # Sidebar for controls
    st.sidebar.title("Upload Images")
    
    content_file = st.sidebar.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
    
    st.sidebar.title("Transfer Parameters")
    
    method = st.sidebar.selectbox(
        "Transfer Method",
        ["color", "amplitude", "phase", "combined", "histogram"],
        index=0,
        help="""
        - color: Transfer colors using LAB color space (often most natural)
        - amplitude: Transfer frequency magnitudes
        - phase: Transfer frequency phases
        - combined: Transfer both amplitude and phase
        - histogram: Match color histograms
        """
    )
    
    alpha = st.sidebar.slider("Alpha (Color/Amplitude influence)", 0.0, 1.0, 0.7, 
                             help="Controls the strength of color or amplitude transfer")
    
    beta = st.sidebar.slider("Beta (Phase influence)", 0.0, 1.0, 0.2,
                            help="Controls the strength of phase transfer (for phase and combined methods)")
    
    gamma = st.sidebar.slider("Gamma (Grayscale Blend)", 0.0, 1.0, 0.0,
                             help="Blend with grayscale version (0.0 = full color, 1.0 = grayscale)")
    
    size_options = {"256x256": (256, 256), "512x512": (512, 512), "1024x1024": (1024, 1024)}
    size_choice = st.sidebar.selectbox("Resize Images to", list(size_options.keys()), index=1)
    size = size_options[size_choice]
    
    show_spectra = st.sidebar.checkbox("Show Frequency Spectra", True, 
                                      help="Display visualizations of the Fourier transforms")
    
    color_spectra = st.sidebar.checkbox("Colored Spectra", True,
                                      help="Show spectra in color instead of grayscale")

    col1, col2 = st.columns(2)
    
    # Load example images if no uploads
    example_images = st.sidebar.checkbox("Use Example Images")
    
    content_img = None
    style_img = None
    
    if example_images:
        st.sidebar.info("Using built-in example images. Upload your own images to replace these.")
        # In a real app, you would include example images with the package
        # For now, let's assume we have example images built-in
        with col1:
            st.subheader("Content Image (Example)")
            st.info("Upload your own content image or use this example")
            # In a real implementation, load actual examples from files
            # Here we're using random noise as a placeholder
            content_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            st.image(content_img, use_container_width=True)
        
        with col2:
            st.subheader("Style Image (Example)")
            st.info("Upload your own style image or use this example")
            # Same placeholder logic
            style_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            st.image(style_img, use_container_width=True)
    else:
        # Display uploaded images
        if content_file:
            with col1:
                st.subheader("Content Image")
                content_img = Image.open(content_file)
                st.image(content_img, use_container_width=True)
                content_img = np.array(content_img)
        
        if style_file:
            with col2:
                st.subheader("Style Image")
                style_img = Image.open(style_file)
                st.image(style_img, use_container_width=True)
                style_img = np.array(style_img)
    
    run_button = st.sidebar.button("Run Style Transfer")
    
    if run_button:
        if content_img is not None and style_img is not None:
            with st.spinner("Applying style transfer..."):
                try:
                    # Resize images
                    content_img_resized = cv2.resize(content_img, size)
                    style_img_resized = cv2.resize(style_img, size)
                    
                    # Run style transfer
                    result_img, content_fft, style_fft, result_fft = fourier_style_transfer(
                        content_img_resized, 
                        style_img_resized, 
                        method=method, 
                        alpha=alpha, 
                        beta=beta,
                        gamma=gamma
                    )
                    
                    # Create visualization
                    plot_buf = create_plot_image(
                        content_img_resized, style_img_resized, result_img,
                        content_fft, style_fft, result_fft,
                        method, show_spectra, color_spectra
                    )
                    
                    # Display results
                    st.subheader("Style Transfer Results")
                    st.image(plot_buf, use_container_width=True)
                    
                    # Create download button for result
                    result_pil = Image.fromarray(result_img)
                    result_buf = io.BytesIO()
                    result_pil.save(result_buf, format="PNG")
                    result_buf.seek(0)
                    
                    st.download_button(
                        label="Download Result Image",
                        data=result_buf,
                        file_name=f"styled_result_{method}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
        else:
            st.warning("Please upload both content and style images, or use example images")
    
    # Add explanation section
    with st.expander("How Fourier Style Transfer Works"):
        st.write("""
        ### The Science Behind Fourier Style Transfer
        
        This app uses the Fourier transform to decompose images into their frequency components.
        The Fourier transform represents an image as a sum of sinusoidal waves of different frequencies.
        
        #### Key Components:
        - **Amplitude spectrum**: Controls the strength of each frequency component, related to texture and contrast
        - **Phase spectrum**: Contains structural information like edges and content placement
        
        #### Transfer Methods:
        1. **Color transfer**: Transfers color characteristics using LAB color space statistics
        2. **Amplitude transfer**: Blends the amplitude spectra, transferring texture patterns
        3. **Phase transfer**: Blends the phase spectra, gently altering structural elements
        4. **Combined transfer**: Applies both amplitude and phase transfer for comprehensive styling
        5. **Histogram matching**: Maps color histograms from style to content
        
        Adjusting the parameters α (alpha), β (beta), and γ (gamma) controls the strength of these effects.
        """)

if __name__ == "__main__":
    main()