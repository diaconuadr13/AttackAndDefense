import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from attacks.cw import cw_l2_attack
import random

def train_adv_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} Adv Train", unit="batch", leave=False)
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # Dynamic attack strategy: Start small (8) and ramp up to (32)
        # This balances training speed early on with robustness later
        base_attack_num = 8
        max_attack_num = 32
        current_attack_num = min(max_attack_num, base_attack_num + (epoch - 1) * 4)
        
        num_to_attack = min(current_attack_num, data.size(0))
        
        # Randomly sample indices from the batch
        attack_indices = random.sample(range(data.size(0)), num_to_attack)
        
        adv_samples = []
        adv_targets = []

        model.eval() 
        for i in attack_indices:
            clean_sample = data[i].unsqueeze(0)
            target_label = target[i].unsqueeze(0)
            
            try:
                # Randomly choose between DeepFool and C&W
                if random.random() < 0.5:
                    adv_sample, _, _ = deepfool(model, clean_sample, device, max_iter=10)
                else:
                    adv_sample = cw_l2_attack(model, clean_sample, target_label, device, 
                                            c=1, max_iter=50, binary_search_steps=1)
                
                adv_samples.append(adv_sample)
                adv_targets.append(target_label)
            except Exception:
                pass
        
        model.train()
        
        # Combine Data
        if adv_samples:
            adv_batch = torch.cat(adv_samples, dim=0)
            adv_tgts = torch.cat(adv_targets, dim=0)
            
            combined_data = torch.cat([data, adv_batch], dim=0)
            combined_target = torch.cat([target, adv_tgts], dim=0)
        else:
            combined_data = data
            combined_target = target

        
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

    train_set = SubsetSC("training")
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = SimpleAudioCNN(n_classes=35).to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth"))
    print("Loaded baseline model. Starting Adversarial Fine-tuning...")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        loss, acc = train_adv_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f}")

        # Save the robust model
        torch.save(model.state_dict(), f"models/robust_model_3_epoch_{epoch}.pth")
        print("Robust model saved.")