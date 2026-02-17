import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, history_len, cuda="0"):

        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

        self.max_size = max_size
        self.buffer = []
        self.history_len = history_len

    def add(self, obs_seq, act_seq, opponent_seq):        
        experience = (obs_seq, act_seq, opponent_seq)
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            # If buffer size is smaller than batch_size, sample with replacement
            batch_idx = np.random.choice(np.arange(buffer_len), batch_size, replace=True)
        else:
            batch_idx = np.random.choice(buffer_len, batch_size, replace=False)
        
        obs_seq_batch = []
        act_seq_batch = []
        opponent_seq_batch = []

        # Construct the batch
        for idx in batch_idx:
            # Ensure the sequence length doesn't exceed history_len
            start_idx = max(0, idx - self.history_len + 1)
            
            # Create a sequence of states by slicing the buffer
            obs_seq_sequence = [self.buffer[i][0] for i in range(start_idx, idx + 1)]
            # Padding the sequence with the first state (if it's shorter than history_len)
            obs_seq_sequence = [obs_seq_sequence[0]] * (self.history_len - len(obs_seq_sequence)) + obs_seq_sequence

            # Create a sequence of states by slicing the buffer
            act_seq_sequence = [self.buffer[i][1] for i in range(start_idx, idx + 1)]
            # Padding the sequence with the first state (if it's shorter than history_len)
            act_seq_sequence = [act_seq_sequence[0]] * (self.history_len - len(act_seq_sequence)) + act_seq_sequence

            # Create a sequence of states by slicing the buffer
            opponent_seq_sequence = [self.buffer[i][2] for i in range(start_idx, idx + 1)]
            # Padding the sequence with the first state (if it's shorter than history_len)
            opponent_seq_sequence = [opponent_seq_sequence[0]] * (self.history_len - len(opponent_seq_sequence)) + opponent_seq_sequence


            obs_seq_batch.append(obs_seq_sequence)
            act_seq_batch.append(act_seq_sequence)
            opponent_seq_batch.append(opponent_seq_sequence)
            
        t_pursuer_obs_seq = torch.FloatTensor(np.array(obs_seq_batch)).to(self.device)
        t_pursuer_act_seq = torch.FloatTensor(np.array(act_seq_batch)).to(self.device)
        t_opponent_seq = torch.FloatTensor(np.array(opponent_seq_batch)).to(self.device)  
        
        return t_pursuer_obs_seq, t_pursuer_act_seq, t_opponent_seq
    
class RecurrentEncoder(nn.Module):
    """Encodes a sequence of (state, action) pairs into a latent embedding."""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64, num_layers=1):
        super(RecurrentEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h = self.gru(x)
        h = h[-1]  # final hidden state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class RecurrentDecoder(nn.Module):
    """Reconstructs the opponent trajectory from latent embedding z."""
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=4, num_layers=1):
        super(RecurrentDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len=10):
        # Expand latent z to a sequence
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # (1, batch, hidden_dim)
        # Dummy inputs: just zeros to drive the GRU for seq_len timesteps
        dummy_input = torch.zeros(z.size(0), seq_len, h0.size(-1), device=z.device)
        outputs, _ = self.gru(dummy_input, h0)
        recon = self.output_layer(outputs)  # (batch, seq_len, output_dim)
        return recon


class LIAMEncoderDecoder(nn.Module):
    """
    Local Information Agent Modelling (LIAM) module.
    Learns a latent embedding of opponent behavior using only
    the pursuer's local observation and action history.
    """
    def __init__(self, state_dim, action_dim, opponent_output_dim=4, capacity = 1e6, batch_size=256,
                 hidden_dim=256, latent_dim=64, history_len=10, lr=3e-4, device="0"):
        super(LIAMEncoderDecoder, self).__init__()
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")
        self.history_len = history_len
        self.batch_size = batch_size

        input_dim = state_dim + action_dim
        self.encoder = RecurrentEncoder(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        latent_dim=latent_dim).to(self.device)
        self.decoder = RecurrentDecoder(latent_dim=latent_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=opponent_output_dim).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(max_size=int(capacity), history_len = history_len, cuda=device)

    def encode(self, pursuer_obs_seq, pursuer_act_seq):
        """
        Encode local history into latent vector z.
        pursuer_obs_seq: (seq_len, state_dim)
        pursuer_act_seq: (seq_len, action_dim)
        """
        self.eval()
        if type(pursuer_obs_seq) is torch.Tensor:
            t_pursuer_obs_seq = pursuer_obs_seq.to(self.device)
            t_pursuer_act_seq = pursuer_act_seq.to(self.device)
        else:
            t_pursuer_obs_seq = torch.FloatTensor(pursuer_obs_seq).to(self.device)
            t_pursuer_act_seq = torch.FloatTensor(pursuer_act_seq).to(self.device)
            
        x = torch.cat([t_pursuer_obs_seq, t_pursuer_act_seq], dim=-1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z, _, _ = self.encoder(x)
        return z.squeeze(0)  # (latent_dim,)

    def forward(self, pursuer_obs_seq, pursuer_act_seq, opponent_seq):
        """
        Full forward pass for training.
        Returns reconstruction loss and KL divergence.
        """
        self.train()

        x = torch.cat([pursuer_obs_seq, pursuer_act_seq], dim=-1).to(self.device)
        
        opponent_seq = opponent_seq.to(self.device)

        
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z, seq_len=self.history_len)

        # Reconstruction + KL divergence loss (VAE-style)
        recon_loss = self.mse_loss(recon, opponent_seq)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + 1e-3 * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_step(self, ):
        # t_pursuer_obs_seq = torch.FloatTensor(pursuer_obs_seq).to(self.device)
        # t_pursuer_act_seq = torch.FloatTensor(pursuer_act_seq).to(self.device)
        # t_opponent_seq = torch.FloatTensor(opponent_seq).to(self.device)  
        
        t_pursuer_obs_seq, t_pursuer_act_seq, t_opponent_seq = self.replay_buffer.sample(self.batch_size)
        # print('t_pursuer_obs_seq.shape:', t_pursuer_obs_seq.shape)
        # print('t_pursuer_act_seq.shape:', t_pursuer_act_seq.shape)
        # print('t_opponent_seq.shape:', t_opponent_seq.shape)
        
        total_loss, recon_loss, kl_loss = self.forward(t_pursuer_obs_seq, t_pursuer_act_seq, t_opponent_seq)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def save(self, save_dir, name="liam"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(save_dir, f"{name}_model.pth"))
        print(f"LIAM model saved to {os.path.join(save_dir, f'{name}_model.pth')}")

    def load(self, filename, name="liam"):
        checkpoint_path = os.path.join(filename, f"{name}_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: LIAM checkpoint not found at {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"LIAM model loaded from {checkpoint_path}")
