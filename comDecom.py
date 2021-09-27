import os
import glob
import torch
import collections
from PIL import Image
from compress import prepare_model, prepare_dataloader, compress_and_save, load_and_decompress, compress_and_decompress

INPUT_DIR = 'files/'
STAGING_DIR = 'stage/'
OUT_DIR = 'out/'
model_path = 'checkpoint/hific-med.pt'

original_sizes = dict()
first_model_init = False
SUPPORTED_EXT = {'.png', '.jpg'}
File = collections.namedtuple('File', ['output_path', 'compressed_path', 'num_bytes', 'bpp'])

all_files = os.listdir(INPUT_DIR)
scale_factor = 2 if len(all_files) == 1 else 4

if not all_files:
    raise ValueError("Please upload/download images!")

def get_bpp(image_dimensions, num_bytes):
    w, h = image_dimensions
    return num_bytes * 8 / (w * h)

for file_name in all_files:
    if os.path.isdir(file_name):
        continue
    if not any(file_name.endswith(ext) for ext in SUPPORTED_EXT):
        print('Skipping non-image', file_name, '...')
        continue
    full_path = os.path.join(INPUT_DIR, file_name)

    file_name, _ = os.path.splitext(file_name)
    original_sizes[file_name] = os.path.getsize(full_path)
    output_path = os.path.join(OUT_DIR, f'{file_name}.png')

if first_model_init is False:
    print('Building model ...')
    model, args = prepare_model(model_path, STAGING_DIR)
    first_model_init = True

data_loader = prepare_dataloader(args, INPUT_DIR, OUT_DIR)
compress_and_save(model, args, data_loader, OUT_DIR)