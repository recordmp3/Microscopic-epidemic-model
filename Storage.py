import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_size, action_size, device = None):
        self.obs = torch.zeros(num_steps + 1, num_processes, obs_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, action_size)
        self.returns = torch.zeros(num_steps + 1, num_processes, action_size)
        self.actions = torch.zeros(num_steps, num_processes, action_size)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        # bad_masks are Masks that indicate whether it's a true terminal state or time limit end state
        self.num_steps = num_steps
        self.step = 0
        self.device = device
        if device is not None:
            self.to(device)
    
    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, actions, value_preds, rewards, masks, bad_masks): 
        # after init, in main the step 0 is fed with first obs, then the first insert will be put in step 1
        # bad_mask[i] means there is no transition between step i and step i + 1
        #print('insert', obs, actions, value_preds, rewards, masks, bad_masks)
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def copy_from_help(self, a, b, rdp1, n_ocu):

        a = a[:, rdp1, :]
        b = b[:, rdp1, :]
        a[:, :n_ocu, :] = b[:, :n_ocu, :]

    def copy_from(self, ro2, pos, ocu):
 
        rdp1 = torch.randperm(self.obs.size(1))
        n_ocu = int(self.obs.size(1) * ocu // 1)
        
        self.obs = self.obs[:, rdp1, :]
        b = ro2.obs[:, rdp1, :]
        self.obs[:, :n_ocu, :] = b[:, :n_ocu, :]
        
        self.rewards = self.rewards[:, rdp1, :]
        b = ro2.rewards[:, rdp1, :]
        self.rewards[:, :n_ocu, :] = b[:, :n_ocu, :]
        
        self.value_preds = self.value_preds[:, rdp1, :]
        b = ro2.value_preds[:, rdp1, :]
        self.value_preds[:, :n_ocu, :] = b[:, :n_ocu, :]

        self.returns = self.returns[:, rdp1, :]
        b = ro2.returns[:, rdp1, :]
        self.returns[:, :n_ocu, :] = b[:, :n_ocu, :]

        self.actions = self.actions[:, rdp1, :]
        b = ro2.actions[:, rdp1, :]
        self.actions[:, :n_ocu, :] = b[:, :n_ocu, :]

        self.masks = self.masks[:, rdp1, :]
        b = ro2.masks[:, rdp1, :]
        self.masks[:, :n_ocu, :] = b[:, :n_ocu, :]

        self.bad_masks = self.bad_masks[:, rdp1, :]
        b = ro2.bad_masks[:, rdp1, :]
        self.bad_masks[:, :n_ocu, :] = b[:, :n_ocu, :]

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
                self.value_preds[-1] = next_value
                gae = torch.zeros((self.value_preds.shape[-1], )).to(self.device)
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]

    
    def feed_forward_generator(self,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        #for x in sampler:print(x)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            #next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, self.actions.size(-1))[indices]
            #masks_batch = self.masks[:-1].view(-1, 1)[indices]

            yield obs_batch, actions_batch, return_batch
