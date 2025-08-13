import torch
import numpy as np
import torch.nn as nn

class RocketTransform(torch.nn.Module):
    def __init__(self, input_length: int, num_kernels: int, device='cpu', generation_type='normal'):
        super().__init__()
        self.input_length = input_length
        self.num_kernels  = num_kernels
        self.device       = device

        candidate_lengths = torch.tensor([7, 9, 11], dtype=torch.int32, device=self.device)

        self.lengths      = candidate_lengths[torch.randint(0, len(candidate_lengths), (num_kernels,), device=self.device)]

        self.kernels      = torch.empty(self.lengths.sum(), dtype=torch.float32, device=self.device)
        self.biases       = torch.empty(num_kernels, dtype=torch.float32, device=self.device)
        self.dilations    = torch.empty(num_kernels, dtype=torch.int32, device=self.device) # use torch.empty as it allows to create 
        self.paddings     = torch.empty(num_kernels, dtype=torch.int32, device=self.device) # empty (to be allocated) tensors of any type 

        pos = 0
        for i, length in enumerate(self.lengths): 
            if generation_type == 'binary': 
                w = torch.randint(low=0, high=2, size=(length,), dtype=torch.float32, device=self.device)
                w = 2 * w - 1
            elif generation_type == 'ternary':
                w = torch.randint(low=0, high=3, size=(length,), dtype=torch.float32, device=self.device)
                w = w - 1
            else:
                w = torch.randn(length, dtype=torch.float32, device=self.device)
            w -= w.mean() # centralization as the article suggests 

            self.kernels[pos:pos+length] = w # we sample weights into single vector tensor to preserve it as length of each kernel may vary
            self.biases[i] = torch.empty(1, dtype=torch.float32, device=self.device).uniform_(-1, 1)

            self.dilation = 2 ** torch.empty(1, device=self.device).uniform_(
                0, torch.log2(torch.tensor((input_length - 1) / (length - 1), device=self.device)) # dilation is sampled as the article suggest
            )
             
            self.dilations[i] = int(self.dilation.item())
            self.paddings[i]  = ((length - 1) * self.dilations[i].item()) // 2 if torch.randint(0, 2, (1,), device=self.device).item() == 1 else 0
            pos += length

        self.classifier = torch.nn.Linear(2 * self.num_kernels, 1)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, X):
        """
        X: shape (batch_size, input_length)
        Returns features: shape (batch_size, num_kernels * 2)
        """
        batch_size = X.size(0)
        features = torch.empty((batch_size, self.num_kernels * 2), dtype=torch.float32, device=X.device)

        pos_w = 0
        for k in range(self.num_kernels):
            length = self.lengths[k].item()
            w      = self.kernels[pos_w:pos_w+length]
            b      = self.biases[k].item()
            d      = self.dilations[k].item()
            p      = self.paddings[k].item()

            # Unfold to get all sliding windows with dilation & padding
            X_unf = nn.functional.unfold(
                X.view(batch_size, 1, -1).unsqueeze(-1),
                kernel_size=(length, 1),
                dilation=(d, 1),
                padding=(p, 0)
            )  # shape: (batch_size, length, output_length)

            conv_out = (w.view(1, -1, 1) * X_unf).sum(1) + b  # (batch_size, output_length)
            ppv      = (conv_out > 0).float().mean(dim=1)     # (batch_size,)
            max_val  = conv_out.max(dim=1).values             # (batch_size,)

            features[:, k*2] = ppv
            features[:, k*2 + 1] = max_val

            pos_w += length
        
        return features

class ClassifierModel(torch.nn.Module):
    def __init__(self, input_length: int, num_kernels: int, device='cpu', generation_type='normal'):
        super().__init__()
        self.input_length = input_length
        self.num_kernels  = num_kernels
        self.device       = device

        self.classifier = torch.nn.Linear(2 * self.num_kernels, 1)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, X):
        """
        X: shape (batch_size, num_kernels * 2)
        Returns prediction: shape (batch_size, 1)
        """
        y_pred = self.classifier(X)

        return self.activation(y_pred)