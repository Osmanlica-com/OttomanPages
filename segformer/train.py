import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor
from dataset import SemanticSegmentationDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import evaluate
from constants import *
import numpy as np
import time
import json
import os

root_folder = "/workspace/Ottoman-Train-Models/segformer_output"

metric = evaluate.load("mean_iou")

feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

train_dataset = SemanticSegmentationDataset(root_dir="/workspace/Ottoman-Train-Models/dataset/train", feature_extractor=feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", num_labels=len(id2label), id2label=id2label, label2id=label2id)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()

log_file_path = os.path.join(root_folder, "training_logs.jsonl")
log_file = open(log_file_path, "w")

for epoch in range(5):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    epoch_start_time = time.time()  # Track epoch start time
    for idx, batch in enumerate(tqdm(train_dataloader)):
        batch_start_time = time.time()  # Track batch start time
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        
        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            
            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        # np.savetxt("predicted.txt", predicted.detach().cpu().numpy().flatten())
        # np.savetxt("labels.txt", labels.detach().cpu().numpy().flatten())
        # let's print loss and metrics every 100 batches
        batch_time = time.time() - batch_start_time
        if idx % 50 == 0:
            metrics = metric.compute(num_labels=1, 
                                    ignore_index=255,
                                    reduce_labels=False)
            # Prepare log entry
            log_entry = {
                "epoch": epoch,
                "iteration": idx,
                "loss": loss.item(),
                "mean_iou": metrics["mean_iou"],
                "mean_accuracy": metrics["mean_accuracy"],
                "batch_time": batch_time,
                "lr": optimizer.param_groups[0]["lr"],  # Log learning rate
            }

            # Write log entry to file
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()  # Ensure the log is written to disk
            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])
        # Log epoch time
    epoch_time = time.time() - epoch_start_time
    log_entry = {
        "epoch": epoch,
        "epoch_time": epoch_time,
    }
    log_file.write(json.dumps(log_entry) + "\n")
    log_file.flush()

    # After each epoch, save the model
    model_save_path = os.path.join(root_folder, f"checkpoint_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Close the log file after training
log_file.close()