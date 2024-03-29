import yaml, easydict, argparse, cdt
from causal_env_v0 import CausalEnv_v0

from hypotheses import *
from policy import CausalPolicy
from utils import evaluate_agent, Logger

METHODS = {
    "pc": cdt.causality.graph.PC,
    "cgnn": cdt.causality.graph.CGNN,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Causal Overhypotheses")
    parser.add_argument("-e", "--env_config", required=True, help="path to env config")
    parser.add_argument("-x", "--exp_config", required=True, help="path to exp config")
    parser.add_argument(
        "-m", "--method", required=True, help="method to use for CD based on CI tests"
    )
    args = parser.parse_args()

    # Load env config
    with open(args.env_config, "r") as f:
        env_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(env_config["hypotheses"], globals(), ldict)
        env_config["hypotheses"] = ldict["hypotheses"]

    env = CausalEnv_v0(env_config)
    action_shape = len(env.action_space)
    obs_shape = env.observation_space._shape[0]

    # Load exp config
    with open(args.exp_config, "r") as f:
        exp_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(exp_config.env.hypotheses, globals(), ldict)
        exp_config.env.hypotheses = ldict["hypotheses"]

    # Initialize policy with overhypotheses clusters
    c_agent = CausalPolicy(
        obs_shape=obs_shape,
        action_shape=action_shape,
        graph=None,
        model=METHODS[args.method](),
    )
    exp_config.env.n_blickets = env_config.n_blickets

    log = Logger(log_dir=f"./logs/{args.method}_random/")

    # Evaluate Agent
    eps_rew, log = evaluate_agent(
        policy=c_agent, env_cfg=exp_config.env, n_eval=exp_config.n_episodes, log=log
    )

    log.save(file_name=f"{args.method}.pkl")
