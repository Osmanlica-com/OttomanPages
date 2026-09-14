from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_model = SamModel(config=model_config).from_pretrained("/workspace/sam_output/epoch_3")
#Update the model by loading the weights from saved file.
# my_mito_model.load_state_dict(torch.load("/content/drive/MyDrive/ColabNotebooks/models/SAM/mito_full_data_10_epochs_model_checkpoint.pth"))

device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)

array_size = 256
grid_size = 10

x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)

xv, yv = np.meshgrid(x, y)

xv_list = xv.tolist()
yv_list = yv.tolist()

input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

#We need to reshape our nxn grid to the expected shape of the input_points tensor
# (batch_size, point_batch_size, num_points_per_image, 2),
# where the last dimension of 2 represents the x and y coordinates of each point.
#batch_size: The number of images you're processing at once.
#point_batch_size: The number of point sets you have for each image.
#num_points_per_image: The number of points in each set.
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

single_patch = Image.open("/workspace/patches_dataset/image_4593_10_3.png").convert('RGB')

inputs = processor(single_patch, input_points=input_points, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}
my_mito_model.eval()

with torch.no_grad():
  outputs = my_mito_model(**inputs, multimask_output=False)

# apply sigmoid
single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)

cv2.imwrite("/workspace/image.png", np.array(single_patch))
cv2.imwrite("/workspace/prediction.png", single_patch_prediction)
cv2.imwrite("/workspace/prob.png", single_patch_prob)

print(single_patch_prob)