from CONSTANTS import *
from alg_net import ALGNet


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.log_for_loss = []
        self.net = ALGNet(self.obs_size, self.n_actions)
        # self.automatic_optimization = False

        # self.agent = Agent()
        # self.total_reward = 0
        # self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # rewards, log_probs, states, actions, values, Qval
        rewards = torch.cat(batch[0]).numpy()
        # log_probs = batch[1]
        states = torch.cat(batch[2])
        actions = torch.cat(batch[3])
        # values = batch[4]
        # Qval = batch[5]

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


