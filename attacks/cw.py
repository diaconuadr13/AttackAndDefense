import torch
import torch.optim as optim

def cw_l2_attack(model, x, labels, device, target_class=None, c=1, kappa=0, max_iter=1000, learning_rate=0.01):
    """
    Carlini & Wagner L2 Attack (Untargeted).
    Optimizes: min ||delta||^2 + c * f(x + delta)
    """
    model.eval()
    
    # x shape: (1, 1, H, W) or (B, 1, H, W)
    # labels shape: (1,) or (B,)
    
    delta = torch.zeros_like(x, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=learning_rate)

    best_l2 = float('inf')
    best_adv = x.clone()
    
    # Track if we ever found a successful adversarial example
    found_success = False

    for step in range(max_iter):
        x_adv = x + delta
        output = model(x_adv)
        
        # 1. L2 distance loss
        l2_loss = torch.sum(delta ** 2)

        # 2. f-function (Attack Loss)
        # Get the logit corresponding to the REAL class
        real_logit = torch.gather(output, 1, labels.unsqueeze(1)).squeeze(1)
        
        # Find the max logit of the OTHER classes
        tmp_output = output.clone()
        tmp_output.scatter_(1, labels.unsqueeze(1), -float('inf'))
        max_other_logit, _ = tmp_output.max(dim=1)

        # Untargeted Logic: We want max_other > real_logit
        f_loss = torch.clamp(real_logit - max_other_logit + kappa, min=0).sum()

        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check if this step produced a valid adversarial example
        # We check the actual prediction to be sure
        pred = output.argmax(dim=1)
        is_adv = (pred != labels)
        
        if is_adv.item():
            found_success = True
            if l2_loss.item() < best_l2:
                best_l2 = l2_loss.item()
                best_adv = x_adv.detach().clone()

    # Return the best successful adversarial example found
    # If we never fooled the model, return the last attempt (or original x if you prefer)
    return best_adv if found_success else (x + delta).detach()