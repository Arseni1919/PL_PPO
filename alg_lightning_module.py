from CONSTANTS import *
from alg_net import ALGNet
from alg_logger import run


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

        self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
        # self.total_reward = 0
        # self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # --------------------------------------- Optimizer --------------------------------------- #
        opt = self.optimizers()
        # ------------------------------------- Unpack Batch -------------------------------------- #
        # rewards, log_probs, states, actions, values, Qval, dones
        rewards = torch.cat(batch[0]).numpy()
        log_probs = batch[1]
        states = torch.cat(batch[2])
        actions = torch.cat(batch[3])
        values = batch[4]
        # Qval = batch[5]
        dones = batch[6]
        lengths = batch[7]

        # --------------------------------------- Advantage --------------------------------------- #
        last_gae = 0.0
        result_adv = []
        result_ref = []

        for val, next_val, reward, done in zip(reversed(values[:-1]), reversed(values[1:]),
                                               reversed(rewards[:-1]), reversed(dones[:-1])):
            if done:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + GAMMA * next_val - val
                last_gae = delta + GAMMA * GAE_LAMBDA * last_gae

            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        result_ref.append(rewards[-1])
        adv_v = torch.FloatTensor(list(reversed(result_adv)))
        ref_v = torch.FloatTensor(list(reversed(result_ref)))

        # ---------------------------------- Normalize Advantage ---------------------------------- #
        log_action_prob_v = torch.FloatTensor(log_probs)
        adv_v = adv_v - torch.mean(adv_v)
        adv_v /= torch.std(adv_v)
        # ----------------------------------------- Epochs ---------------------------------------- #
        for epoch in range(PPO_EPOCHES):
            for indx, batch_ofs in enumerate(range(0, len(states), PPO_BATCH_SIZE)):
                # print(f'batch indx: {indx}')
                batch_l = batch_ofs + PPO_BATCH_SIZE
                states_v = states[batch_ofs:batch_l]
                actions_v = actions[batch_ofs:batch_l]
                batch_adv_v = adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = log_action_prob_v[batch_ofs:batch_l]
                # ---------------------------- Critic Update -------------------------------------- #
                # opt = self.optimizers()
                # opt.zero_grad()
                # self.manual_backward(ac_loss, opt)
                # opt.step()

                opt.zero_grad()
                values_v, policy_dists = self.net(states_v.numpy())
                batch_values_v = values_v.squeeze()
                if batch_values_v.size() != batch_ref_v.size():
                    print(f'bad -> input: {batch_values_v.size()}, target: {batch_ref_v.size()}')
                loss_value_v = F.mse_loss(batch_values_v, batch_ref_v)
                loss_value_v.backward()
                opt.step()
                # ----------------------------- Actor Update -------------------------------------- #
                opt.zero_grad()
                # mu_v = net_act(states_v)
                values_v, policy_dists = self.net(states_v.numpy())

                # logprob_pi_v = calc_logprob( mu_v, net_act.logstd, actions_v)
                action_prob_v = policy_dists[0, range(len(actions_v)), actions_v]
                logprob_pi_v = torch.log(action_prob_v)

                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                opt.step()
                # -------------------------------------------------------------------------------- #

                # logging
                loss = loss_value_v + loss_policy_v
                self.log_for_loss.append(loss.item())

                if NEPTUNE:
                    run['acc_loss'].log(loss)
                    run['acc_loss_log'].log(f'{loss}')

        self.plot({'rewards': rewards, 'values': values, 'ref_v': ref_v.numpy(),
                   'loss': self.log_for_loss, 'lengths': lengths, 'adv_v': adv_v.numpy()})

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)

    def plot(self, graph_dict):
        # plot live:
        if PLOT_LIVE:
            # plt.clf()
            # plt.plot(list(range(len(self.log_for_loss))), self.log_for_loss)
            # plt.plot(list(range(len(rewards))), rewards)

            ax = self.fig.get_axes()

            for indx, (k, v) in enumerate(graph_dict.items()):
                ax[indx].cla()
                ax[indx].plot(list(range(len(v))), v, c='r')  #, edgecolor='b')
                ax[indx].set_title(f'Plot: {k}')
                ax[indx].set_xlabel('iters')
                ax[indx].set_ylabel(f'{k}')

            plt.pause(0.05)
            # plt.pause(1.05)
            # plt.show()




    @staticmethod
    def compute_Qvals(rewards):
        Qval = 0
        Qvals = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        return Qvals


