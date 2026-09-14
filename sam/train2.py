import wandb
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from dataset import SAMDataset
from datasets import load_from_disk
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import os

wandb_on = True
num_epochs = 5
num_batch = 8

output_dir = "/workspace/sam_output"

# Initialize wandb
if wandb_on:
    wandb.init(project="huggingface", name="sam-vit-base")

# Load dataset
dataset = load_from_disk("/workspace/dataset_bulk")

# Load model and processor
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Freeze vision encoder and prompt encoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Create dataset and dataloader
train_dataset = SAMDataset(dataset=dataset, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True, drop_last=False)

# Define loss function
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# Initialize optimizer
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

global_step = 0  # Initialize the global step counter

# Log hyperparameters to wandb
if wandb_on:
    wandb.config.update({"learning_rate": 1e-5, "epochs": num_epochs, "batch_size": num_batch})

for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        # Forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # Compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss
        epoch_losses.append(loss.item())

        global_step += 1
        
        if global_step % 5 == 0:
            # Log batch loss and other outputs
            if wandb_on:
                wandb.log({
                    "batch_loss": loss.item(),
                    "iou_scores": outputs.iou_scores.mean().item(),
                    "pred_masks_mean": predicted_masks.mean().item(),
                    "global_step": global_step
                })

    mean_epoch_loss = mean(epoch_losses)
    print(f'EPOCH: {epoch} - Mean loss: {mean_epoch_loss}')

    # Log epoch-level metrics
    if wandb_on:
        wandb.log({"epoch_loss": mean_epoch_loss, "epoch": epoch})

     # Save model checkpoint after each epoch
    epoch_checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    model.save_pretrained(epoch_checkpoint_dir)
    processor.save_pretrained(epoch_checkpoint_dir)
    print(f"Model checkpoint saved at {epoch_checkpoint_dir}")

# Save model checkpoint after each epoch
epoch_checkpoint_dir = os.path.join(output_dir, f"final_model")
os.makedirs(epoch_checkpoint_dir, exist_ok=True)
model.save_pretrained(epoch_checkpoint_dir)
processor.save_pretrained(epoch_checkpoint_dir)
print(f"Model checkpoint saved at {epoch_checkpoint_dir}")

# Finish wandb run
if wandb_on:
    wandb.finish()
