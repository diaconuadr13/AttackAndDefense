import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn, get_labels
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from attacks.cw import cw_l2_attack

def evaluate_attacks(num_samples=100): # Set to None to run on all
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load Data
    test_set = SubsetSC("testing")
    # Increase batch size for faster iteration (attacks still need 1 by 1 loop or modified batch logic)
    # For simplicity and safety with our attack implementations, we keep batch_size=1
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    labels_list = get_labels(test_set)

    # 2. Load Model
    model = SimpleAudioCNN(n_classes=35).to(device)
    try:
        model.load_state_dict(torch.load("models/baseline_model.pth"))
    except FileNotFoundError:
        print("Baseline model not found! Run train.py first.")
        return
    model.eval()

    # Metrics
    total_tested = 0
    deepfool_success = 0
    cw_success = 0
    
    # We only count samples that were originally classified correctly
    correctly_classified_original = 0

    print(f"\n--- Running Attack Evaluation on {num_samples if num_samples else 'ALL'} samples ---")
    
    pbar = tqdm(test_loader, total=num_samples if num_samples else len(test_loader))

    for i, (data, target) in enumerate(pbar):
        if num_samples and i >= num_samples:
            break
            
        data, target = data.to(device), target.to(device)
        target_idx = target.item()
        
        # Check initial prediction
        init_pred = model(data).argmax(dim=1).item()
        
        if init_pred != target_idx:
            continue # Skip already misclassified examples
            
        correctly_classified_original += 1
        
        # --- DeepFool ---
        # DeepFool returns the perturbed image and the prediction
        _, _, pred_df = deepfool(model, data, device, max_iter=50) # Standard max_iter
        if pred_df != target_idx:
            deepfool_success += 1
            
        # --- C&W ---
        # C&W returns the perturbed image
        # Increase max_iter for better success rate (e.g., 100 or 1000)
        # For speed in this evaluation script, we use 100, but 1000 is better for final results.
        adv_cw = cw_l2_attack(model, data, device, c=10, learning_rate=0.01, max_iter=100) 
        pred_cw = model(adv_cw).argmax(dim=1).item()
        if pred_cw != target_idx:
            cw_success += 1
            
        # Update progress bar description
        df_rate = deepfool_success / correctly_classified_original * 100
        cw_rate = cw_success / correctly_classified_original * 100
        pbar.set_description(f"DF Success: {df_rate:.1f}% | CW Success: {cw_rate:.1f}%")

    print("\n" + "="*30)
    print(f"Total Evaluated (Correctly Classified Originally): {correctly_classified_original}")
    print(f"DeepFool Success Rate: {deepfool_success / correctly_classified_original * 100:.2f}%")
    print(f"C&W Success Rate:      {cw_success / correctly_classified_original * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    # Run on 200 samples for a good estimate. Set to None for full dataset (takes longer).
    evaluate_attacks(num_samples=200)