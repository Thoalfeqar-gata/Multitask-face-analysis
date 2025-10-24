import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Helper function 
def l2_norm(input, axis=1):
    """L2 Normalizes a tensor along the specified dimension."""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm + 1e-7) # Add epsilon for safety
    return output



class AdaFaceLoss(nn.Module):
    """
    Applies adaptive margin based on feature norms to pre-calculated logits,
    scales the logits, and computes CrossEntropyLoss.

    Args:
        h (float): Hyperparameter for margin scale adjustment range (e.g., 0.333).
        s (float): Scale factor (e.g., 64.0).
        m (float): Base additive margin (e.g., 0.4).
        t_alpha (float): Smoothing factor for EMA of norm statistics (e.g., or 0.99).
        eps (float): Epsilon for numerical stability.
    """
    def __init__(self,
                 h=0.333,
                 s=64.0,
                 m=0.4,
                 t_alpha=1.0,
                 eps=1e-6
                 ):
        super(AdaFaceLoss, self).__init__()
        # Store hyperparameters
        self.m = m
        self.eps = eps
        self.h = h
        self.s = s

        # EMA prep for batch mean/std of norms
        self.t_alpha = t_alpha
        # These are buffers, not learnable parameters
        self.register_buffer('t', torch.zeros(1))
        # Initialize with reasonable estimates (these adapt during training)
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

        # Final loss function (integrated)
        self.criterion = nn.CrossEntropyLoss()

        print('\nAdaFaceLoss Initialized')
        print(f' - h: {self.h}, s: {self.s}, m: {self.m}, t_alpha: {self.t_alpha}')

    def forward(self, logits: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the AdaFace loss.

        Args:
            logits (torch.Tensor): Pre-calculated cosine similarities (cos(theta)). Shape [B, C].
            norms (torch.Tensor): L2 norms of the *unnormalized* embeddings (||x||). Shape [B, 1].
            labels (torch.Tensor): Ground truth labels. Shape [B].

        Returns:
            torch.Tensor: The final loss value.
        """
        # Clamp logits for stability
        cosine = logits.clamp(-1 + self.eps, 1 - self.eps)

        # --- AdaFace Margin Calculation Logic ---
        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        # Update batch mean/std using EMA during training
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        # Calculate adaptive margin scaler
        margin_scaler = (safe_norms - self.batch_mean.item()) / (self.batch_std.item() + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # --- Apply Margin to Target Logits ---
        target_labels = labels.reshape(-1, 1).long() # Ensure labels are long type

        # 1. Apply g_angular (adaptive angular margin)
        m_arc_mask = torch.zeros_like(cosine)
        m_arc_mask.scatter_(1, target_labels, 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc_mask * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine_m_angular = theta_m.cos()

        # 2. Apply g_additive (adaptive additive margin)
        m_cos_mask = torch.zeros_like(cosine_m_angular)
        m_cos_mask.scatter_(1, target_labels, 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos_mask * g_add
        cosine_m_final = cosine_m_angular - m_cos

        # 3. Scale the final modified logits
        scaled_output_logits = cosine_m_final * self.s

        # 4. Calculate Cross Entropy Loss
        loss = self.criterion(scaled_output_logits, labels)
        return loss


class ArcFaceLoss(torch.nn.Module):
    """
    ArcFace Loss .
    Applies additive angular margin to pre-calculated logits, scales them,
    and computes CrossEntropyLoss.

    Args:
        s (float): Scale factor (e.g., 64.0).
        margin (float): Additive angular margin in radians (e.g., 0.5).
        easy_margin (bool): Whether to use easy margin modification.
        eps (float): Epsilon for numerical stability.
    """
    def __init__(self, s=64.0, margin=0.5, easy_margin=False, eps=1e-7):
        super(ArcFaceLoss, self).__init__()
        self.scale = s
        self.margin = margin
        self.easy_margin = easy_margin
        self.eps = eps

        # Precompute values for margin application
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for applying the penalty: cos(pi - m)
        self.theta_threshold = math.cos(math.pi - margin)
        # Penalty term: sin(pi - m) * m
        self.penalty_term = math.sin(math.pi - margin) * margin

        # Final loss function (integrated)
        self.criterion = nn.CrossEntropyLoss()

        print('\nArcFaceLoss  Initialized')
        print(f' - s: {self.scale}, margin: {self.margin} rad')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the ArcFace loss.

        Args:
            logits (torch.Tensor): Pre-calculated cosine similarities (cos(theta)). Shape [B, C].
            labels (torch.Tensor): Ground truth labels. Shape [B].

        Returns:
            torch.Tensor: The final loss value.
        """
        # Find valid samples (optional, depends on if you use -1 labels)
        # index = torch.where(labels != -1)[0]
        # if index.size(0) == 0: return logits.sum() * 0 # Handle empty batch case

        target_labels = labels.view(-1, 1).long() # Ensure labels are long type

        # Create index mask efficiently
        index_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target_labels, 1)

        # Select target logits
        target_logit = logits[index_mask] # cos(theta_yi)

        # Calculate sin(theta_yi) using sqrt(1 - cos^2(theta_yi))
        # Add epsilon for numerical stability, clamp to avoid sqrt of negative
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2) + self.eps)

        # Calculate cos(theta_yi + m) using angle addition formula
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

        # Apply penalty if theta_yi + m >= pi (i.e., target_logit <= theta_threshold)
        if self.easy_margin:
            # Simpler version: only apply margin if target_logit > 0
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            # Standard ArcFace penalty
            final_target_logit = torch.where(
                target_logit > self.theta_threshold, cos_theta_m, target_logit - self.penalty_term
            )

        # Create output tensor, replace target logit with the modified one
        output_logits = logits.clone()
        output_logits[index_mask] = final_target_logit

        # Scale all logits
        output_logits *= self.scale

        # Calculate Cross Entropy Loss
        loss = self.criterion(output_logits, labels)
        return loss


class CosFaceLoss(torch.nn.Module):
    """
    CosFace Loss / LMCL .
    Applies additive cosine margin to pre-calculated logits, scales them,
    and computes CrossEntropyLoss.

    Args:
        s (float): Scale factor (e.g., 64.0).
        m (float): Additive cosine margin (e.g., 0.40).
    """
    def __init__(self, s=64.0, m=0.40):
        super(CosFaceLoss, self).__init__()
        self.s = s
        self.m = m

        # Final loss function (integrated)
        self.criterion = nn.CrossEntropyLoss()

        print('\nCosFaceLoss  Initialized')
        print(f' - s: {self.s}, m: {self.m}')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the CosFace loss.

        Args:
            logits (torch.Tensor): Pre-calculated cosine similarities (cos(theta)). Shape [B, C].
            labels (torch.Tensor): Ground truth labels. Shape [B].

        Returns:
            torch.Tensor: The final loss value.
        """
        target_labels = labels.view(-1, 1).long() # Ensure labels are long type

        # Create index mask efficiently
        index_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target_labels, 1)

        # Select target logits
        target_logit = logits[index_mask] # cos(theta_yi)

        # Apply additive cosine margin: cos(theta_yi) - m
        final_target_logit = target_logit - self.m

        # Create output tensor, replace target logit with the modified one
        output_logits = logits.clone()
        output_logits[index_mask] = final_target_logit

        # Scale all logits
        output_logits *= self.s

        # Calculate Cross Entropy Loss
        loss = self.criterion(output_logits, labels)
        return loss


class AnnealedSphereFaceLoss(nn.Module):
    """
    SphereFace Loss (A-Softmax) with Lambda Annealing .
    Applies multiplicative angular margin (phi_theta) with annealing to pre-calculated logits,
    scales by feature norm, and computes CrossEntropyLoss.

    Args:
        m (int): Integer margin multiplier (e.g., 4).
        lambda_min (float): Minimum value for lambda during annealing (e.g., 5.0).
        lambda_max (float): Initial value for lambda (e.g., 1500.0).
        lambda_anneal_speed (float): Controls annealing speed (e.g., 0.1).
        use_chebyshev (bool): Use Chebyshev polynomials for cos(m*theta). Faster, needs m <= 5.
        eps (float): Epsilon for numerical stability in acos.
    """
    def __init__(self, m=4, lambda_min=5.0, lambda_max=1500.0, lambda_anneal_speed=0.1, use_chebyshev=True, eps=1e-7):
        super().__init__()
        assert isinstance(m, int) and m >= 1
        self.m = m
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_anneal_speed = lambda_anneal_speed
        self.use_chebyshev = use_chebyshev
        self.eps = eps

        # Iteration counter and lambda for annealing
        self.register_buffer('it', torch.zeros(1, dtype=torch.long))
        self.register_buffer('lamb', torch.tensor(lambda_max))

        # Chebyshev polynomials T_m(x), property: T_m(cos(theta)) = cos(m*theta)
        if use_chebyshev:
            assert m <= 5, "Chebyshev implementation here only supports m <= 5"
            self.mlambda = [
                lambda x: x**0, lambda x: x**1, lambda x: 2*x**2-1,
                lambda x: 4*x**3-3*x, lambda x: 8*x**4-8*x**2+1,
                lambda x: 16*x**5-20*x**3+5*x
            ]

        # Final loss function (integrated)
        self.criterion = nn.CrossEntropyLoss()

        print('\nAnnealedSphereFaceLoss Initialized')
        print(f' - m: {self.m}, lambda: {lambda_max}->{lambda_min}, anneal_speed: {lambda_anneal_speed}, chebyshev: {use_chebyshev}')


    def forward(self, logits: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the annealed SphereFace loss.

        Args:
            logits (torch.Tensor): Pre-calculated cosine similarities (cos(theta)). Shape [B, C].
            norms (torch.Tensor): L2 norms of the *unnormalized* embeddings (||x||). Shape [B, 1].
            labels (torch.Tensor): Ground truth labels. Shape [B].

        Returns:
            torch.Tensor: The final loss value.
        """
        # --- Lambda Annealing ---
        if self.training:
            self.it += 1
            current_lambda = max(self.lambda_min, self.lambda_max / (1.0 + self.lambda_anneal_speed * self.it.item()))
            if self.lamb.item() != current_lambda:
                 self.lamb.fill_(current_lambda)
        else:
            current_lambda = self.lambda_min # Use minimum lambda for eval

        # --- Calculate phi(theta) for the target class ---
        target_labels = labels.view(-1, 1).long() # Ensure labels are long type
        index_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target_labels, 1)
        target_logits = logits[index_mask] # cos(theta_yi)

        # Clamp for numerical stability before acos
        cos_theta = target_logits.clamp(-1 + self.eps, 1 - self.eps)
        theta = cos_theta.acos() # theta_yi

        # Calculate cos(m*theta_yi) using Chebyshev or Taylor approx
        if self.use_chebyshev:
            cos_m_theta = self.mlambda[self.m](cos_theta)
        else:
             cos_m_theta = torch.cos(self.m * theta)

        # Calculate k = floor(m*theta / pi)
        k = (self.m * theta / math.pi).floor()
        # Calculate phi(theta_yi) = (-1)^k * cos(m*theta_yi) - 2*k
        phi_theta_target = ((-1.0) ** k) * cos_m_theta - 2 * k

        # --- Combine Logits using Annealed Lambda ---
        # output_target = (lambda * cos(theta_yi) + phi(theta_yi)) / (1 + lambda)
        combined_target_logit = (current_lambda * target_logits + phi_theta_target) / (1 + current_lambda)

        # Create output tensor, replace target logit with the combined one
        output_logits_modified = logits.clone()
        output_logits_modified[index_mask] = combined_target_logit

        # --- Scale by Feature Norm ||x|| ---
        # Ensure norms are broadcastable (B, 1) vs (B, C)
        output_logits_scaled = output_logits_modified * norms

        # --- Calculate Cross Entropy Loss ---
        loss = self.criterion(output_logits_scaled, labels)

        return loss

# --- Example Type 2 Training Loop ---

# class FaceModelWithHead(nn.Module):
#     def __init__(self, backbone_module, embedding_size, num_classes):
#         super().__init__()
#         self.backbone = backbone_module # Outputs RAW embeddings (B, F)
#         self.head_weights = nn.Linear(embedding_size, num_classes, bias=False)
#
#     def forward(self, x):
#         raw_embeddings = self.backbone(x) # (B, F)
#         norms = torch.norm(raw_embeddings, p=2, dim=1, keepdim=True)
#         norm_embeddings = raw_embeddings / (norms + 1e-7)
#         norm_weights = F.normalize(self.head_weights.weight)
#         logits = F.linear(norm_embeddings, norm_weights) # cos(theta) (B, C)
#         return logits, norms # Return both logits and norms

# --- Setup ---
# model = FaceModelWithHead(...)
#
# # Choose Loss
# criterion = AdaFaceLoss().to('cuda')
# # criterion = ArcFaceLoss().to('cuda')
# # criterion = CosFaceLoss().to('cuda')
# # criterion = AnnealedSphereFaceLoss().to('cuda')
#
# optimizer = optim(...) # Only model.parameters()
# scheduler = ...

# --- Training Loop ---
# for epoch ...:
#   for images, labels ...:
#       optimizer.zero_grad()
#       logits, norms = model(images)
#       # AdaFace and AnnealedSphereFaceLoss require norms, others don't ignore it if passed
#       loss = criterion(logits, norms, labels)
#       loss.backward()
#       optimizer.step()
#       # ...