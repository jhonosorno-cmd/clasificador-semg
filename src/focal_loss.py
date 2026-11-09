"""
Focal Loss para manejar desbalance extremo de clases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Penaliza más los ejemplos difíciles de clasificar (bajo p_t).
    Reduce el peso de ejemplos fáciles (alto p_t).
    
    Args:
        alpha: Pesos por clase (Tensor de tamaño num_classes)
        gamma: Factor de enfoque (default: 2.0)
        reduction: 'mean' o 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (Batch, num_classes) - logits
            targets: (Batch,) - class indices
        """
        # Calcular probabilidades
        probs = F.softmax(inputs, dim=1)
        
        # Obtener probabilidad de la clase correcta
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # (Batch,)
        
        # Calcular log(pt)
        log_pt = torch.log(pt + 1e-8)
        
        # Focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Focal loss
        loss = -focal_weight * log_pt
        
        # Aplicar alpha (class weights)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Reducción
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # Test Focal Loss
    print("=== TEST FOCAL LOSS ===\n")
    
    # Crear datos de prueba
    batch_size = 8
    num_classes = 7
    
    # Logits aleatorios
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])  # Desbalanceado
    
    # Class weights (inverse frequency)
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0
    alpha = class_counts.sum() / (num_classes * class_counts)
    
    print(f"Targets: {targets.tolist()}")
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Alpha (class weights): {alpha.tolist()}\n")
    
    # Comparar CrossEntropyLoss vs FocalLoss
    ce_loss = nn.CrossEntropyLoss(weight=alpha)
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    
    ce_value = ce_loss(logits, targets)
    focal_value = focal_loss(logits, targets)
    
    print(f"CrossEntropyLoss: {ce_value.item():.4f}")
    print(f"FocalLoss (gamma=2): {focal_value.item():.4f}")
    print(f"\nFocal Loss penaliza más los errores en clases minoritarias")
