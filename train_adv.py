import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool

def train_adv_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} Adv Train", unit="batch", leave=False)
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # --- 1. Generate Adversarial Examples (On-the-fly) ---
        # DeepFool is slow, so we only attack the first 4 samples of the batch
        # to keep training speed reasonable for this project.
        num_to_attack = min(4, data.size(0)) 
        adv_samples = []
        adv_targets = []

        # Switch to eval mode to generate attacks
        model.eval() 
        for i in range(num_to_attack):
            # Unsqueeze to make it (1, 1, H, W) for the attack function
            clean_sample = data[i].unsqueeze(0)
            target_label = target[i].unsqueeze(0)
            
            try:
                # Run DeepFool
                adv_sample, _, _ = deepfool(model, clean_sample, device, max_iter=10)
                adv_samples.append(adv_sample)
                adv_targets.append(target_label)
            except Exception:
                # If attack fails, skip
                pass
        
        model.train() # Switch back to train mode
        
        # --- 2. Combine Data ---
        if adv_samples:
            adv_batch = torch.cat(adv_samples, dim=0)
            adv_tgts = torch.cat(adv_targets, dim=0)
            
            # Combine clean batch + generated adversarial samples
            combined_data = torch.cat([data, adv_batch], dim=0)
            combined_target = torch.cat([target, adv_tgts], dim=0)
        else:
            combined_data = data
            combined_target = target

        # --- 3. Standard Training Step ---
        optimizer.zero_grad()
        output = model(combined_data)
        loss = criterion(output, combined_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(combined_target).sum().item()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader), correct / len(loader.dataset) # Approx accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_set = SubsetSC("training")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Load Baseline Model
    model = SimpleAudioCNN(n_classes=35).to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth"))
    print("Loaded baseline model. Starting Adversarial Fine-tuning...")

    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower learning rate for fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Fine-tune for a few epochs (e.g., 3-5)
    for epoch in range(1, 4):
        loss, acc = train_adv_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f}")

    # Save the robust model
    torch.save(model.state_dict(), "models/robust_model.pth")
    print("Robust model saved.")