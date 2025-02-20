import torch
import torch.nn as nn

def calculate_lambda(sigma, k=1.0):
    """
    Calculate the upper bound for Lipschitz constant based on sigma.
    """
    lambda_val = k / (torch.exp(sigma**2 / 2) + 3 * torch.sqrt((torch.exp(sigma**2) - 1) * torch.exp(sigma**2)))
    return lambda_val

def lipschitz_regularization_loss(model, sigma, beta=0.01):
    """
    Compute the Lipschitz regularization loss.
    """
    lambda_val = calculate_lambda(sigma)  # sigma 값을 받아서 lambda_val 계산
    regularization_loss = 0
    
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight  # Get layer weights

            # Reshape Conv2D filters into 2D matrix
            weights_reshaped = weights.view(weights.shape[0], -1)

            # Compute W^T * W
            W_T_W = torch.matmul(weights_reshaped.t(), weights_reshaped)
            
            # Compute identity matrix scaled by lambda^2
            identity_scaled = torch.eye(W_T_W.shape[0], device=weights.device) * (lambda_val ** 2)
            
            # Compute Frobenius norm ||W^T * W - λ^2 * I||^2
            regularization_term = torch.norm(W_T_W - identity_scaled, p='fro') ** 2
            
            # Add regularization term to total loss
            regularization_loss += beta * regularization_term
    
    return regularization_loss

def custom_loss(model, sigma, beta):
    """
    Compute Lipschitz regularization loss to be added to the primary loss.
    
    This function is designed to be used in train.py as:
        loss = criterion(output, target)
        if hasattr(args, 'inject_variation') and args.inject_variation:
            loss += custom_loss(model, sigma)
    """
    return lipschitz_regularization_loss(model, sigma, beta)
