from causal_env_v0 import CausalEnv_v0

if __name__=='__main__':

    env_config = {"reward_structure": 'quiz', "quiz_disabled_steps": -1}

    env = CausalEnv_v0(env_config)

    for i in range(10):
        print("environment count: {}".format(i+1))

        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_, r, done, info = env.step(a)
            print("State: {}\tAction: {}\tNext State: {}\tReward: {}\tDone: {}\tInfo: {}".format(s, a, s_, r, done, info))
            s = s_