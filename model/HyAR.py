import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import os

class DiscreteEmbedding(nn.Module):
    def __init__(self, num_actions, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, latent_dim, hidden_dim, action_bound):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.action_bound = action_bound  # New parameter for action bounding
        self.latent_dim = latent_dim

    def encode(self, x, e):
        z_params = self.encoder(torch.cat([x, e], dim=-1))
        mu, logvar = z_params.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, e):
        action = self.decoder(torch.cat([z, e], dim=-1))
        # Apply tanh to bound the action in [-action_bound, action_bound]
        return torch.tanh(action) * self.action_bound

    def forward(self, x, e):
        mu, logvar = self.encode(x, e)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, e)
        return x_hat, mu, logvar
    
# class ConditionalVAE(nn.Module):
#     def __init__(self, input_dim, embed_dim, latent_dim, hidden_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim + embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2 * latent_dim)  # mean and logvar
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim + embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
#         self.latent_dim = latent_dim

#     def encode(self, x, e):
#         z_params = self.encoder(torch.cat([x, e], dim=-1))
#         mu, logvar = z_params.chunk(2, dim=-1)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z, e):
#         return self.decoder(torch.cat([z, e], dim=-1))

#     def forward(self, x, e):
#         mu, logvar = self.encode(x, e)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z, e)
#         return x_hat, mu, logvar


class LatentActor(nn.Module):
    def __init__(self, state_dim, embed_dim, latent_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim + latent_dim)
        )
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

    def forward(self, state):
        out = self.fc(state)
        e_latent = out[:, :self.embed_dim]
        z_latent = out[:, self.embed_dim:]
        return e_latent, z_latent


class Critic(nn.Module):
    def __init__(self, state_dim, embed_dim, latent_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + embed_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, e, z):
        return self.q(torch.cat([state, e, z], dim=-1))


class ReplayBuffer:
    def __init__(self, max_size=1e6, cuda = '0'):
        self.buffer = []
        self.max_size = max_size
        self.device = torch.device(f'cuda:{cuda}')  # Default device, can be changed later

    def store(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    # def sample(self, batch_size):
    #     batch = random.sample(self.buffer, batch_size, replace=len(self.buffer) < batch_size)
    #     states, actions, x_conts, rewards, next_states, dones = zip(*batch)
    #     return (
    #         torch.tensor(states, dtype=torch.float32).to(self.device),
    #         torch.tensor(actions, dtype=torch.long).to(self.device),
    #         torch.tensor(x_conts, dtype=torch.float32).to(self.device),
    #         torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device),
    #         torch.tensor(next_states, dtype=torch.float32).to(self.device),
    #         torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
    #     )
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            # Sample with replacement using numpy
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        else:
            # Sample without replacement
            indices = random.sample(range(len(self.buffer)), batch_size)

        batch = [self.buffer[i] for i in indices]

        states, actions, x_conts, rewards, next_states, dones = zip(*batch)
        # print('type(states) = ', type(states))
        # print('type(actions) = ', type(actions))
        # print('type(x_conts) = ', type(x_conts))
        # print('type(rewards) = ', type(rewards))
        # print('type(next_states) = ', type(next_states))
        # print('type(dones) = ', type(dones))
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.long).to(self.device),
            torch.tensor(np.array(x_conts), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1).to(self.device),
        )

    


class HyARAgent:
    def __init__(self, state_dim, binary_action_dim, continuous_action_dim, embed_dim=8, latent_dim=4, hidden_dim=256,
                 lr=3e-4, gamma=0.99, cuda = '0'):
        self.device = torch.device(f'cuda:{cuda}')
        self.gamma = gamma

        # Update for binary and continuous actions
        self.binary_action_dim = binary_action_dim
        self.continuous_action_dim = continuous_action_dim

        # Embedding for discrete action (binary)
        self.embedding = DiscreteEmbedding(binary_action_dim, embed_dim).to(self.device)
        self.vae = ConditionalVAE(input_dim=continuous_action_dim, embed_dim=embed_dim,
                                  latent_dim=latent_dim, hidden_dim=hidden_dim,
                                  action_bound=1).to(self.device)
        self.actor = LatentActor(state_dim, embed_dim, latent_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, embed_dim, latent_dim, hidden_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.vae_opt = optim.Adam(list(self.vae.parameters()) + list(self.embedding.parameters()), lr=lr)

        self.buffer = ReplayBuffer(cuda = cuda)

    def remember(self, state, action_idx, x_cont, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        
        Parameters:
        - state (ndarray): The state observed at the current timestep.
        - action_idx (int): The discrete action index chosen by the agent (e.g., 0 or 1).
        - x_cont (ndarray): The continuous action chosen by the agent.
        - reward (float): The reward the agent received after taking the action.
        - next_state (ndarray): The next state observed after the action.
        - done (bool): A boolean flag indicating if the episode is done (True if done, False otherwise).
        """
        self.buffer.store((state, action_idx, x_cont, reward, next_state, done))

    def choose_action(self, state):
        # Convert state to tensor and move to device
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get latent action (discrete embedding + continuous latent)
        e_latent, z_latent = self.actor(state_tensor)
        # print(e_latent.shape, z_latent.shape)
        
        # Discrete action selection via embedding table
        e_table = self.embedding.embedding.weight
        # print(e_table.shape)
        # print(e_table)
        dists = torch.cdist(e_latent, e_table)  # Compute distances to embeddings
        discrete_idx = torch.argmin(dists, dim=-1)  # Select the closest action
        e = e_table[discrete_idx]  # Select the corresponding embedding

        # Decode the continuous action using the VAE
        continuous_action = self.vae.decode(z_latent, e).squeeze(0)

        # Return both discrete action (binary) and continuous action
        return discrete_idx.item(), continuous_action.detach().cpu().numpy()
    

    def update(self, batch_size=64):

        states, actions, x_conts, rewards, next_states, dones = self.buffer.sample(batch_size)
        # states, actions, x_conts, rewards, next_states, dones = [
        #     x.to(self.device) for x in (states, actions, x_conts, rewards, next_states, dones)
        # ]

        # --- VAE + Embedding ---
        e = self.embedding(actions)
        x_recon, mu, logvar = self.vae(x_conts, e)
        recon_loss = nn.MSELoss()(x_recon, x_conts)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        vae_loss = recon_loss + kl_loss

        self.vae_opt.zero_grad()
        vae_loss.backward(retain_graph=False)
        self.vae_opt.step()

        # --- Critic ---
        with torch.no_grad():
            next_e_latent, next_z_latent = self.actor(next_states)
            next_e_table = self.embedding.embedding.weight
            next_idxs = torch.argmin(torch.cdist(next_e_latent, next_e_table), dim=-1)
            next_e = next_e_table[next_idxs]
            q_next = self.critic(next_states, next_e, next_z_latent).squeeze(-1)
            q_target = rewards.squeeze(-1) + self.gamma * (1 - dones.squeeze(-1)) * q_next
            
    
        # z_targets, _ = self.vae.encode(x_conts, e)
        
        # --- after VAE update ---
        with torch.no_grad():
            e_detached = self.embedding(actions).detach()
            x_conts_detached = x_conts.detach()
            z_targets, _ = self.vae.encode(x_conts_detached, e_detached)
        q_vals = self.critic(states, e_detached, z_targets).squeeze(-1)
        
        # q_vals = self.critic(states, e, z_targets).squeeze(-1)
        critic_loss = nn.MSELoss()(q_vals, q_target.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor ---
        e_latent, z_latent = self.actor(states)
        q_pi = self.critic(states, e_latent, z_latent).squeeze(-1)
        actor_loss = -q_pi.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        

    def save_models(self, save_dir, name=''):
        
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'HyAR_actor_' + name))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'HyAR_critic_' + name))
        torch.save(self.vae.state_dict(), os.path.join(save_dir, 'HyAR_vae_' + name))
        torch.save(self.embedding.state_dict(), os.path.join(save_dir, 'HyAR_embedding_' + name))
        
        print('Models saved successfully')

    def load_models(self, save_dir, name=''):
        device = self.device  # e.g., cuda or cpu fallback
        
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'HyAR_actor_' + name), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(save_dir, 'HyAR_critic_' + name), map_location=device))
        self.vae.load_state_dict(torch.load(os.path.join(save_dir, 'HyAR_vae_' + name), map_location=device))
        self.embedding.load_state_dict(torch.load(os.path.join(save_dir, 'HyAR_embedding_' + name), map_location=device))
        
        print('Models loaded successfully')