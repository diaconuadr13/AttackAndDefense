import torch
import torch.optim as optim

def cw_l2_attack(model, x, labels, device, target_class=None, c=1, kappa=0, max_iter=100, learning_rate=0.01, binary_search_steps=5):
    """
    Carlini & Wagner L2 Attack (Untargeted) with Binary Search.
    Optimizes: min ||delta||^2 + c * f(x + delta)
    """
    model.eval()
    
    # Binary search bounds
    c_lower = 0
    c_upper = 1e10
    c_current = c
    
    best_l2 = float('inf')
    best_adv = x.clone()
    found_success = False

    for b_step in range(binary_search_steps):
        delta = torch.zeros_like(x, requires_grad=True, device=device)
        optimizer = optim.Adam([delta], lr=learning_rate)
        
        # Track if this c_current yields any success
        current_c_success = False
        
        for step in range(max_iter):
            x_adv = x + delta
            output = model(x_adv)
            
            # L2 distance loss
            l2_loss = torch.sum(delta ** 2)

            # f-function (Attack Loss)
            real_logit = torch.gather(output, 1, labels.unsqueeze(1)).squeeze(1)
            tmp_output = output.clone()
            tmp_output.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_other_logit, _ = tmp_output.max(dim=1)

            f_loss = torch.clamp(real_logit - max_other_logit + kappa, min=0).sum()

            loss = l2_loss + c_current * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if this step produced a valid adversarial example
            pred = output.argmax(dim=1)
            if pred.item() != labels.item():
                current_c_success = True
                if l2_loss.item() < best_l2:
                    best_l2 = l2_loss.item()
                    best_adv = x_adv.detach().clone()
                    found_success = True
        
        # Binary Search Update
        if current_c_success:
            # Success! Try smaller c to reduce L2
            c_upper = c_current
            c_current = (c_lower + c_current) / 2
        else:
            # Failure! Need larger c to prioritize attack success
            c_lower = c_current
            if c_upper < 1e9:
                c_current = (c_current + c_upper) / 2
            else:
                c_current *= 10

    return best_adv if found_success else x