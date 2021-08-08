import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_size, action_space,  base_kwargs = None, mode='disc'):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        base = MLPBase
        self.base = base(obs_size, **base_kwargs) # 11.22 placeholder

        if mode == 'disc': self.dist = nn.ModuleList([Categorical(self.base.output_size, action_space[i]) for i in range(len(action_space))])
        else: self.dist = nn.ModuleList([DiagGaussian(self.base.output_size, action_space[i]) for i in range(len(action_space))])
    

    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError
    """
    def act(self, inputs, rnn_hxs, masks, deterministic=False, return_prob=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist, action, action_log_probs = [None for i in range(len(self.dist))], [None for i in range(len(self.dist))], [None for i in range(len(self.dist))]
        pb = [None for i in range(len(self.dist))]
        for i in range(len(self.dist)):
            dist[i] = self.dist[i](actor_features)

            if deterministic:
                action[i] = dist[i].mode().float()
            else:
                action[i] = dist[i].sample().float()

            action_log_probs[i] = dist[i].log_probs(action[i])
            if (not isinstance(dist[i], torch.distributions.categorical.Categorical)) and return_prob:
                raise NotImplementedError
            pb[i] = dist.probs[i] if return_prob else torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True)
        return value, torch.cat(action, -1), pb, rnn_hxs
    """
    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist, action, action_log_probs = [None for i in range(len(self.dist))], [None for i in range(len(self.dist))], [None for i in range(len(self.dist))]
        for i in range(len(self.dist)):
            dist[i] = self.dist[i](actor_features)

            if deterministic:
                action[i] = dist[i].mode().float()
            else:
                action[i] = dist[i].sample().float()

            action_log_probs[i] = dist[i].log_probs(action[i])
        return value, torch.cat(action, -1), torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True), rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        a = action.split(1, -1)
        for x in a:x = x.long()
        action = a
        dist, action_log_probs, dist_entropy = [None for i in range(len(self.dist))], [None for i in range(len(self.dist))], [None for i in range(len(self.dist))]
        for i in range(len(self.dist)):
            dist[i] = self.dist[i](actor_features)
            action_log_probs[i] = dist[i].log_probs(action[i])
            dist_entropy[i] = dist[i].entropy().mean()
        return value, torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True), torch.mean(torch.Tensor(dist_entropy)), rnn_hxs
        
def std_init(t):

    for name, param in t.named_parameters():
        if 'bias' in name:
                nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

class MHAT2(nn.Module):
    def __init__(self, q_size, k_size, hidden_size, out_size, n_head = 4):
        super(MHAT2, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.k_size = k_size
        self.fc_q = nn.ModuleList([nn.Linear(q_size, hidden_size, bias = True) for i in range(n_head)])
        self.fc_v = nn.ModuleList([nn.Linear(k_size, hidden_size, bias = False) for i in range(n_head)])
        self.fc_k = nn.ModuleList([nn.Linear(k_size, hidden_size, bias = False) for i in range(n_head)])
        self.fc_o = nn.Linear(n_head * hidden_size, out_size, bias = False)
        self.ln = nn.LayerNorm(normalized_shape = (n_head * hidden_size, ))
        for i in range(self.n_head):
            std_init(self.fc_k[i])
            std_init(self.fc_q[i])
            std_init(self.fc_v[i])
        std_init(self.fc_o)
    
    def forward(self, me, other): # size of me and other are [T, N, q_size], [T, N, k, v_size]
        l_out = [] 
        for i in range(self.n_head):
            k = self.fc_k[i](other) # [T, N, k, h_size]
            v = self.fc_v[i](other)
            q = self.fc_q[i](me)
            a = torch.matmul(q.unsqueeze(-2), k.transpose(-1, -2)) / self.hidden_size ** 0.5 # [T, N, 1, h] [T, N, h, k] [T, N, 1, k]
            a = torch.softmax(a, dim = -1).squeeze() #[T, N, k]
            out = torch.sum(a.unsqueeze(-1) * v, -2) #[T, N, h]
            l_out.append(out)
        if self.n_head == 1:c = l_out
        else:c = torch.cat(l_out, -1)
        return self.fc_o(torch.relu(self.ln(c)))
        #return self.ln(self.fc_o(c))
        #return self.fc_o(c)

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, attention = 0):
        super(NNBase, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._attention = attention
        
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        if attention:
            self.mhat_b = MHAT2(6, 10, hidden_size // 2, hidden_size // 2, attention) # q k h out head
            self.mhat_f = MHAT2(6, 7, hidden_size // 2, hidden_size // 2, attention)
            self.mhat_o = MHAT2(6, 10, hidden_size // 2, hidden_size // 2, attention) # q k h out head
            self.em = nn.Sequential(init_(nn.Linear(6 + hidden_size * 3 // 2, hidden_size)), nn.ReLU())
        
    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_attention(self):
        return self._attention
    
    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            
            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, attention = 0):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size, attention)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs = None, masks = None):
        x = inputs
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
        
