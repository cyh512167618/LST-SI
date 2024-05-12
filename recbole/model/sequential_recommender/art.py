import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentCapsuleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_capsules, capsule_dim, num_iterations):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_iterations = num_iterations
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.capsule_layer = nn.ModuleList([
            nn.Linear(hidden_dim, num_capsules * capsule_dim) for _ in range(num_capsules)
        ])
        self.output_layer = nn.Linear(num_capsules * capsule_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        h, _ = self.lstm(x)  # h shape: [batch_size, seq_len, hidden_dim]

        # Compute capsules for each time step
        capsule_list = []
        for t in range(h.size(1)):
            # Compute capsules for each capsule type
            capsule_t_list = []
            for c in range(self.num_capsules):
                # Compute capsule for capsule type c
                capsule_c = self.capsule_layer[c](h[:, t, :])  # capsule_c shape: [batch_size, capsule_dim]
                capsule_c = capsule_c.view(capsule_c.size(0), -1, self.capsule_dim)  # capsule_c shape: [batch_size, 1, capsule_dim]
                capsule_t_list.append(capsule_c)

            # Concatenate capsules for all capsule types at time step t
            capsule_t = torch.cat(capsule_t_list, dim=1)  # capsule_t shape: [batch_size, num_capsules, capsule_dim]
            capsule_list.append(capsule_t)

        # Concatenate capsules across all time steps
        capsules = torch.stack(capsule_list, dim=1)  # capsules shape: [batch_size, seq_len, num_capsules, capsule_dim]

        # Dynamic routing between capsules
        b = torch.zeros(capsules.size(0), capsules.size(2), self.num_iterations).to(capsules.device)
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=1)  # c shape: [batch_size, num_capsules, num_iterations]
            u = (c.unsqueeze(-1) * capsules).sum(dim=1)  # u shape: [batch_size, num_capsules, capsule_dim]
            v = self.squash(u)  # v shape: [batch_size, num_capsules, capsule_dim]
            if i < self.num_iterations - 1:
                b += (capsules * v.unsqueeze(1)).sum(dim=-1)

        # Output layer
        output = self.output_layer(v.view(v.size(0), -1))  # output shape: [batch_size, output_dim]
        return output

    def squash(self, x):
        norm_squared = (x ** 2).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(norm_squared)
        scale = norm_squared / (1 + norm_squared)
        return scale * x / norm
