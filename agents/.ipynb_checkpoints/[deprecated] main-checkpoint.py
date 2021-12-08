from collections import namedtuple
from itertools import count
from ppo import PPO
from env.MA_DemandResponse import MADemandResponseEnv as env
from .utils import datetime2List, superDict2List, normSuperDict

# if __name__ == '__main__':
#     Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
#     max_episode = 1000
#     agent = PPO()
#     render = False
#     for episode in range(max_episode):
#         state = env.reset()
#         if render:
#             env.render()
#         for t in count():
#             action, action_prob = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             trans = Transition(state, action, action_prob, reward, next_state)
#             if render:
#                 env.render()
#             agent.store_transition(trans)
#             state = next_state

#             if done :
#                 if len(agent.buffer) >= agent.batch_size:
#                     agent.update(episode)
#                 break
                
                
if __name__ == '__main__':
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
    max_episode = 1000
    max_iter = 1000
    agent = PPO()
    render = False
    for episode in range(max_episode):
        obs_dict = env.reset()
        if render:
            env.render()
        # for t in count():
        for t in range(max_iter):
            action_and_prob = {k:agent.select_action(np.array(normSuperDict(obs_dict, k))) for k in obs_dict.keys()}
            action = {k:action_and_prob[k][0] for k in obs_dict.keys()}
            action_prob = {k:action_and_prob[k][1] for k in obs_dict.keys()}
            next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for k in obs_dict.keys():
                agent.store_transition(Transition(state[k], action[k], action_prob[k], rewards_dict[k], next_obs_dict[k]))
                if render:
                    env.render()
            obs_dict = next_obs_dict

    #             if done: ## Question
    #                 if len(agent.buffer) >= agent.batch_size:
    #                     agent.update(episode)
    #                 break
        if len(agent.buffer) >= agent.batch_size:
            agent.update(episode)