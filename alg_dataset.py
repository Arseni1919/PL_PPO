from CONSTANTS import *


class ALGDataset(torch.utils.data.IterableDataset):
    def __init__(self, net, env, sample_size: int = 1):
        self.net = net
        self.env = env
        self.sample_size = sample_size

    def __iter__(self):  # -> Tuple:
        state = self.env.reset()
        log_probs = []
        rewards = []
        values = []
        states = []
        actions = []
        Qval = 0
        self.net.eval()
        for steps in range(MAX_LENGTH_OF_A_GAME):
            value, policy_dist = self.net(np.expand_dims(state, axis=0))
            # value = value.detach().squeeze().numpy()[0, 0]
            value = value.detach().squeeze().item()
            dist = policy_dist.detach().squeeze().numpy()

            action = np.random.choice(self.net.n_actions, p=dist)
            log_prob = torch.log(policy_dist.squeeze()[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = self.env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            self.net.entropy_term += entropy
            state = new_state

            if done or steps == MAX_LENGTH_OF_A_GAME - 1:
                Qval, _ = self.net.forward(np.expand_dims(new_state, axis=0))
                Qval = Qval.detach().numpy()
                break
        self.net.train()
        yield rewards, log_probs, states, actions, values, Qval

    def append(self, experience):
        self.buffer.append(experience)
