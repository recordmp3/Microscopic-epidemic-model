class Args_ind():

    def __init__(self):

        self.gamma = 0.9#0.99
        self.lamda = 0.9#0.95
        self.n_epi_per_train = 1       # PPO info
        self.n_epi_lag = 1     #  deprecated.
        self.n_epi_ocu = 1 / 3 # how many trajectories are substituted each episode?
        self.clip_param = 0.1
        self.ppo_epoch = 5
        self.num_mini_batch = 100 # per episode
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 7e-4
        self.eps = 1e-5
        self.max_grad_norm = 1e4

        self.n_process = 1
        self.alpha = 1 / 3 # softness of q-learning

class Args_gov():# government agent optimizing utility (revenue vs epidemic control); almost deprecated, now serves as a supplement to args_ind.

    def __init__(self):

        self.alpha = 1 # "softness" of q-learning
        self.train_length = 4096
        self.alpha_train = 0.01
        self.learn_rate_decay = 1 # for 200-step epoch, *0.81 per epoch
        self.gamma = 0.99
        self.beta1 = 0.9
        self.lamda = 0.95
        self.n_epi_per_train = 1
        # PPO info
        self.clip_param = 0.05
        self.ppo_epoch = 5
        self.num_mini_batch = 1
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 0
        self.eps = 1e-5
        self.max_grad_norm = 2e3#1000#0.5
        self.use_gae = True
