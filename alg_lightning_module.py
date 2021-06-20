from CONSTANTS import *
from alg_net import ALGNet
from alg_logger import run


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    The following takes the trajectory with steps and calculates advantages for
    the actor and reference values for the critic training.

    :param trajectory: in size of every batch
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    last_gae = 0.0
    result_adv = []
    result_ref = []

    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae

        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.log_for_loss = []
        self.net = ALGNet(self.obs_size, self.n_actions)

        # NOT USING OPTIMIZERS AUTOMATICALLY
        self.automatic_optimization = False

        # self.agent = Agent()
        # self.total_reward = 0
        # self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # ------------------------------------- Unpack Batch -------------------------------------- #
        # rewards, log_probs, states, actions, values, Qval
        rewards = torch.cat(batch[0]).numpy()
        # log_probs = batch[1]
        states = torch.cat(batch[2])
        actions = torch.cat(batch[3])
        # values = batch[4]
        # Qval = batch[5]

        # --------------------------------------- Advantage --------------------------------------- #

        # ---------------------------------- Normalize Advantage ---------------------------------- #

        # ----------------------------------------- Epochs ---------------------------------------- #

        # ------------------------------------ Critic Update -------------------------------------- #

        # ------------------------------------- Actor Update -------------------------------------- #

        # compute Real values
        real_values = self.compute_Qvals(rewards)
        discounted_rewards = torch.tensor(real_values)
        # normalize discounted rewards
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # values, policy_dists = self.net(torch.unsqueeze(states, 0))
        values, policy_dists = self.net(states.numpy())

        # Accumulate the policy gradients
        action_prob_v = policy_dists[0, range(len(actions)), actions]
        log_action_prob_v = torch.log(action_prob_v)
        adv_v = real_values - values.detach().squeeze().numpy()
        adv_v = torch.tensor(adv_v)
        # actor_loss = -1 * log_action_prob_v * discounted_rewards
        actor_loss = -1 * log_action_prob_v * adv_v
        actor_loss = actor_loss.sum()
        # actor_loss.sum().backward(retain_graph=True)

        # Accumulate the value gradients
        values_to_mse = values.squeeze(-1).float()
        real_values_to_mse = torch.unsqueeze(torch.tensor(real_values), 0).float()
        critic_loss = F.mse_loss(values_to_mse, real_values_to_mse)

        # Entropy
        # entropy = action_prob_v.detach() * log_action_prob_v.detach()
        # entropy = - entropy.sum().item()
        # entropy_term = ENTROPY_BETA * entropy

        # ac_loss = actor_loss + critic_loss - entropy_term
        # ac_loss = critic_loss + entropy_term
        ac_loss = actor_loss + critic_loss

        # logging
        # tr = np.sum(rewards)
        # self.log('total_reward', tr)
        tl = ac_loss.item()
        # self.log('train loss', tl, on_step=True)
        self.log_for_loss.append(tl)

        if NEPTUNE:
            run['acc_loss'].log(ac_loss)
            run['acc_loss_log'].log(f'{ac_loss}')

        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(ac_loss, opt)
        # opt.step()
        return ac_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)

    @staticmethod
    def compute_Qvals(rewards):
        Qval = 0
        Qvals = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        return Qvals



# update actor critic
# values = torch.FloatTensor(values)
# Qvals = torch.FloatTensor(Qvals)
# log_probs = torch.stack(log_probs)

# advantage = Qvals - values
# actor_loss = (-log_probs * advantage).mean()
#
# values, _ = self.net(states.numpy())
# values = torch.FloatTensor(values.squeeze())
# advantage = Qvals - values
# # critic_loss = F.mse_loss(values, Qvals)
# critic_loss = 0.5 * advantage.pow(2).mean()


