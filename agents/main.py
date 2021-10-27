from collections import namedtuple
from itertools import count
from ppo import PPO
from env.MA_DemandResponse import MADemandResponseEnv as env


if __name__ == '__main__':
    Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
    max_episode = 1000
    agent = PPO()
    render = False
    for episode in range(max_episode):
        state = env.reset()
        if render: env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state

            if done :
                if len(agent.buffer) >= agent.batch_size: agent.update(episode)
                break