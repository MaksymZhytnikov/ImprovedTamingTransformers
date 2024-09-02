import torch

# Define paths for VQGAN configuration and checkpoint files
VQGAN_CONFIG_COCO = "logs/2021-01-20T16-04-20_coco_transformer/configs/2021-02-08T17-18-53-project.yaml"
VQGAN_CHECKPOINT_COCO = "logs/2021-01-20T16-04-20_coco_transformer/model_ckpt/last.ckpt"

# Set the device to CUDA if available, otherwise use CPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Main prompt for the image generation
PROMPT = "A beautiful sunset over a city skyline."

# Dimensions for the generated images
WIDTH = 512
HEIGHT = 512

# Frequency for displaying intermediate results during training
DISPLAY_FREQUENCY = 50

# Initial image to start the generation process (set to None if not used)
INITIAL_IMAGE = None  # "data/coco_segmentations/000000335529.png"

# Target images for guiding the generation (empty string if not used)
TARGET_IMAGES = ""  # @param {type:"string"}

# Random seed for reproducibility (-1 means no specific seed)
SEED = -1  # @param {type:"number"}

# Maximum number of iterations for the generation process
MAX_ITERATIONS = 500  # @param {type:"number"}

# Handle the initial image and target images parameters
if SEED == -1:
    SEED = None

if INITIAL_IMAGE == "None":
    INITIAL_IMAGE = None

if TARGET_IMAGES == "None" or not TARGET_IMAGES:
    TARGET_IMAGES = []
else:
    TARGET_IMAGES = TARGET_IMAGES.split("|")
    TARGET_IMAGES = [image.strip() for image in TARGET_IMAGES]

INPUT_IMAGES = bool(INITIAL_IMAGE or TARGET_IMAGES)

PROMPTS_LIST = [frase.strip() for frase in PROMPT.split("|")]
if PROMPTS_LIST == ['']:
    PROMPTS_LIST = []

# Additional parameters
NOISE_PROMPT_SEEDS = []
NOISE_PROMPT_WEIGHTS = []
INIT_WEIGHT = 0.0
CLIP_MODEL = 'ViT-B/32'
STEP_SIZE = 0.1
CUTN = 64
CUT_POW = 1.0

