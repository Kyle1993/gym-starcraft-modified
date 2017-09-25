from gym_starcraft.envs.easy_battle_env import EasyBattleEnv
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--ip', help="server ip")
parser.add_argument('--port', help="server port", type=int, default=11111)
parser.add_argument('--frame-skip', type=int, default=2)
parser.add_argument('--max-step', type=int, default=10000)
parser.add_argument('--max-episode', type=int, default=1000)

args = parser.parse_args()


env = EasyBattleEnv(args.ip,args.port,frame_skip=args.frame_skip)

for e in range(args.max_episode):
    state = env.reset()
    for step in range(args.max_step):
        action = np.random.uniform(-1.,1.,[5,3])
        next_state,reward,done,info = env.step(action)
        print(reward,done,info)
        state = next_state
        if done:
            break

