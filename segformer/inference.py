# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb#scrollTo=_OI7144HdXNj

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from constants import id2label, label2id
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the trained model checkpoint
model_checkpoint_path = "/Users/batuhanmanav/Desktop/Ottoman-Train-Models/layout/segformer/checkpoint_epoch_0.pth"

# Load the model architecture
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", num_labels=len(id2label), id2label=id2label, label2id=label2id)
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

# Load the model weights from the checkpoint
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))

# Move the model to the appropriate device (GPU or CPU)
model.to(device)

# Load the input image
image_path = "/Users/batuhanmanav/Desktop/Ottoman-Train-Models/dataset/test/0bd7df1f-239e-4fac-a3f9-d9dce9ec9b4a.png"
image = Image.open(image_path).convert("RGB")

# Preprocess the image using the feature extractor
inputs = feature_extractor(images=image, return_tensors="pt")

# Move the inputs to the appropriate device
pixel_values = inputs.pixel_values.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits

# Resize the logits to match the original image size
upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)

# Get the predicted segmentation map
predicted = upsampled_logits.argmax(dim=1).cpu().numpy()

predicted[predicted == 1] = 255

np.savetxt('predicted.txt', predicted.flatten())

# Convert the predicted segmentation map to an image
predicted_image = Image.fromarray(predicted[0].astype(np.uint8))

# Save or display the predicted image
predicted_image.save("predicted_segmentation.png")
predicted_image.show()