import torch

class LoRALinear(torch.nn.Module):
    def __init__(self, in_dim , out_dim , rank , alpha , dropout):
        super().__init__()
        # These are the weights from the original pretrained model
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)

        # These are the new LoRA params. In general rank << in_dim, out_dim
        self.lora_a = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, out_dim, bias=False)

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha

        # Most implementations also include some dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        # The original params are frozen, and only LoRA params are trainable.
        self.linear.weight.requires_grad = False
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor):
        # This would be the output of the original model
        frozen_out = self.linear(x)

        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return frozen_out + (self.alpha / self.rank) * lora_out

