# test_attacks.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import SubsetSC, collate_fn, get_labels
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from attacks.cw import cw_l2_attack
from tqdm import tqdm

def plot_attack(original, perturbed, label, adv_label, title):
    # Helper to plot MFCCs
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Original
    ax[0].imshow(original.squeeze().cpu().numpy(), origin='lower', aspect='auto')
    ax[0].set_title(f"Original: {label}")
    
    # Plot Adversarial
    ax[1].imshow(perturbed.squeeze().cpu().numpy(), origin='lower', aspect='auto')
    ax[1].set_title(f"Adversarial: {adv_label}")
    
    # Plot Noise
    noise = (perturbed - original).squeeze().cpu().numpy()
    ax[2].imshow(noise, origin='lower', aspect='auto')
    ax[2].set_title("Perturbation (Noise)")
    
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Load Data and Labels
    test_set = SubsetSC("testing")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    labels_list = get_labels(test_set)

    # Load Trained Model
    model = SimpleAudioCNN(n_classes=35).to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth"))
    model.eval()

    # Run Attack Loop
    print("Running attacks on 5 samples...")
    for i, (data, target) in enumerate(test_loader):
        if i >= 5: break # Stop after 5 examples
        
        data, target = data.to(device), target.to(device)
        original_label_name = labels_list[target.item()]
        
        # Check if model is correct initially
        init_pred = model(data).argmax(dim=1)
        if init_pred != target:
            continue 

        print(f"\n--- Sample {i} ({original_label_name}) ---")

        # --- DeepFool Attack ---
        print("Generating DeepFool attack...")
        adv_df, iters, pred_df = deepfool(model, data, device)
        pred_name_df = labels_list[pred_df]
        print(f"DeepFool: {original_label_name} -> {pred_name_df} (in {iters} iters)")
        plot_attack(data, adv_df, original_label_name, pred_name_df, "DeepFool Attack")

        # --- C&W Attack ---
        print("Generating C&W attack...")
        adv_cw = cw_l2_attack(model, data, target, device, c=100, learning_rate=0.01, max_iter=1000)
        pred_cw = model(adv_cw).argmax(dim=1).item()
        pred_name_cw = labels_list[pred_cw]
        print(f"C&W: {original_label_name} -> {pred_name_cw}")
        plot_attack(data, adv_cw, original_label_name, pred_name_cw, "Carlini & Wagner Attack")