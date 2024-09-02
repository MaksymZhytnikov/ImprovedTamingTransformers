import os
import shutil
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import re

def clear_folder(folder_path='sampling'):
    """
    Delete all files and subdirectories in the specified folder.

    Parameters:
    folder_path (str): Path to the folder to be cleared.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f'🗄️ Folder {folder_path} does not exist.')
        return

    # Iterate over all filenames in the specified folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if the path is a file or a symbolic link
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Delete the file or symbolic link
                os.unlink(file_path)
            # Check if the path is a directory
            elif os.path.isdir(file_path):
                # Delete the directory and all its contents
                shutil.rmtree(file_path)
        except Exception as e:
            # Print an error message if deletion fails
            print(f'❌ Failed to delete {file_path}. Reason: {e}')


def show_pil_image(image_path):
    """
    Display an image from the specified file path using PIL and matplotlib.

    Parameters:
    image_path (str): The file path to the image to be displayed.

    This function loads an image from the given file path using the PIL library,
    and then displays the image using matplotlib. The axis is turned off for a cleaner display.
    """
    # Load the image using PIL
    image = Image.open(image_path)

    # Create a new figure with a specified size
    plt.figure(figsize=(8, 8))
    
    # Display the image
    plt.imshow(image)
    
    # Hide the axis for a cleaner look
    plt.axis('off')
    
    # Render the image
    plt.show()
    
    
def create_random_input(image_size, num_categories, device):
    """
    Create a random input tensor for non-conditional generation.

    Args:
    image_size (tuple): The size of the image to generate (height, width).
    num_categories (int): Number of categories for the random input.
    device (str): Device to use for tensor operations.

    Returns:
    torch.Tensor: A random tensor of shape (1, num_categories, height, width).
    """
    height, width = image_size
    random_input = torch.randint(0, num_categories, (1, height, width), device=device)
    one_hot = F.one_hot(random_input, num_classes=num_categories).float()
    return one_hot.permute(0, 3, 1, 2)


def show_segmentation(s):
    """
    Display a segmentation tensor as an image.

    Args:
    s (torch.Tensor): Segmentation tensor to display.
    """
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
    colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
    colorize = colorize / colorize.sum(axis=2, keepdims=True)
    s = s@colorize
    s = s[...,0,:]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    display(Image.fromarray(s))
    

def show_image(s):
    """
    Display an image tensor.

    Args:
    s (torch.Tensor): Image tensor to display.
    """
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    display(Image.fromarray(s))
    
    
def show_decoded_vector(image_tensor):
    # Assuming the tensor is already on the CPU, if not, move it to the CPU
    image_tensor = image_tensor.cpu().numpy()

    # Since the tensor has shape (1, 3, H, W), we need to remove the batch dimension
    image = image_tensor[0]

    # Transpose the dimensions from (C, H, W) to (H, W, C) for plotting
    image = image.transpose(1, 2, 0)

    # Plot the image
    plt.figure(figsize=(12,6))
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis
    plt.show()


def create_gif(output_path='output.gif', folder_path='sampling', duration=100):
    """
    Create a GIF from images in the specified folder.
    Parameters:
    output_path (str): Path to save the output GIF.
    folder_path (str): Path to the folder containing images.
    duration (int): Duration between frames in milliseconds.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Get a list of all PNG files in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith('.png')]
    
    print(f"🔮 Total PNG files found: {len(images)}")

    # Filter and sort images by iteration and step
    def extract_info(filename):
        match = re.search(r'image_iter(\d+)_step(\d+)\.png', filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    images = [img for img in images if extract_info(img) is not None]
    images.sort(key=lambda x: extract_info(x))
    
    # Load images
    frames = []
    for img in images:
        try:
            frame = Image.open(os.path.join(folder_path, img))
            frames.append(frame)
        except (OSError, IOError) as e:
            print(f"❌ Error loading image {img}: {e}")
    
    # Save as GIF
    if frames:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"💾 GIF saved to {output_path}")
        except (OSError, IOError) as e:
            print(f"❌ Error saving GIF: {e}")
    else:
        print("🤔 No images found to create GIF")

    
def save_image(generated_image, step, iteration, sampling_folder='sampling'):
    if not os.path.exists(sampling_folder):
        os.makedirs(sampling_folder)
        
    # Ensure the tensor is on CPU and in the correct format
    generated_image = generated_image.cpu().squeeze(0)
    
    # Adjust the dynamic range
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(generated_image)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(3) 
    
    # Save the image
    filename = f"{sampling_folder}/image_iter{iteration}_step{step}.png"
    pil_image.save(filename, 'PNG', quality=100)
    