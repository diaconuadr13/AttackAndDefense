import torch
import copy

def deepfool(model, x, device, num_classes=35, overshoot=0.02, max_iter=50):
    """
    DeepFool Attack Implementation.
    x: Input tensor (1, 1, H, W)
    """
    model.eval()
    
    image = x.clone().detach().to(device)
    
    # Get the prediction for the original image
    with torch.no_grad():
        output = model(image)
    pred = output.argmax(dim=1)
    
    if pred != output.argmax():
        return image, 0, pred

    # Sort classes by probability
    I = output.argsort(descending=True)
    
    # Initialize perturbation variables
    w = torch.zeros(image.shape).to(device)
    r_tot = torch.zeros(image.shape).to(device)

    loop_i = 0
    x_adv = image.clone().detach()
    
    original_label = pred.item()
    current_label = original_label

    while current_label == original_label and loop_i < max_iter:
        x_adv.requires_grad = True
        
        # Forward pass
        output = model(x_adv)
        
        # Get gradient of the original class
        if x_adv.grad is not None:
            x_adv.grad.zero_()
            
        output[0, original_label].backward(retain_graph=True)
        grad_orig = x_adv.grad.data.clone()
        
        pert = float('inf')
        w_best = torch.zeros(image.shape).to(device)

        # Determine the closest decision boundary
        for k in range(1, min(num_classes, 10)):
            target_idx = I[0, k].item()
            
            x_adv.grad.zero_()
            
            output[0, target_idx].backward(retain_graph=True)
            grad_curr = x_adv.grad.data.clone()

            w_k = grad_curr - grad_orig
            f_k = (output[0, target_idx] - output[0, original_label]).data

            # Metric to minimize
            pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-8)

            if pert_k < pert:
                pert = pert_k
                w_best = w_k

        # Update total perturbation
        r_i = (pert + 1e-4) * w_best / (torch.norm(w_best) + 1e-8)
        r_tot = r_tot + r_i

        # Apply perturbation to the original image
        x_adv = image + (1 + overshoot) * r_tot
        
        x_adv = x_adv.detach() 
        
        # Check new prediction
        with torch.no_grad():
            current_label = model(x_adv).argmax(dim=1).item()
            
        loop_i += 1

    return x_adv, loop_i, current_label