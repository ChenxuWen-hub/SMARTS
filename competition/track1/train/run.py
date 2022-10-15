import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import warnings
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Dict

import gym
import stable_baselines3 as sb3lib
import torch as th
from ruamel.yaml import YAML
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from train import env as multi_scenario_env
import network
import policy

from stable_baselines3 import PPO
from stable_baselines3 import DDPG

print("\nTorch cuda is available: ", th.cuda.is_available(), "\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    """
    main函数主要就是定义需要载入的文件、输出保存地址、搭建env、运行run函数。
    输入：args，在主函数定义
    输出：无
    里面没有需要改的东西，都是功能性的。
    """

    # Load config file.
    config_file = yaml.load(
        (Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config = config_file["smarts"]
    config["mode"] = args.mode

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).absolute().parents[0] / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("\nLogdir:", logdir, "\n")

    # Setup model.
    if config["mode"] == "evaluate":
        # Begin evaluation.
        config["model"] = args.model  # 当命令中添加mode为evaluate时，需要添加model为train保存的路径下的model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    wrappers = multi_scenario_env.wrappers(config=config)  # 在这里导入打包好的action, observation和reward!!!
    for scen in config["scenarios"]:  # 遍历config里的场景作为train和eval的场景，这俩的场景是相同的
        envs_train[f"{scen}"] = multi_scenario_env.make(
            config=config, scenario=scen, wrappers=wrappers
        )
        envs_eval[f"{scen}"] = multi_scenario_env.make(
            config=config, scenario=scen, wrappers=wrappers
        )

    # Run training or evaluation.
    run(envs_train=envs_train, envs_eval=envs_eval, config=config)

    # Close all environments
    for env in envs_train.values():
        env.close()
    for env in envs_eval.values():
        env.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
):
    """
    run函数是整套程序的核心部分，定义了模型(model = xxx)，并在循环里训练模型(model.learn)；
    评估evaluate部分要导入训练之后的模型，并用evaluate_policy评估，返回mean_reward和std_reward
    输入：训练和评估场景，config文件
    输出：无
    需要改：
        model = xxx model里的接口参数需要调整
        model.learn(xxx) learn里的接口参数需要对应调整
    """


    if config["mode"] == "train":
        print("\nStart training.\n")
        scenarios_iter = cycle(config["scenarios"])  # 在config给的几个场景中无限循环取

        ###############定义模型##############
        model = getattr(sb3lib, config["alg"])(
            env=envs_train[next(scenarios_iter)], # 这里的env里就有我们打包好的action等信息了
            verbose=1,
            tensorboard_log=config["logdir"] / "tensorboard",
            **network.combined_extractor(config),
        )

        # model = policy.PPO(
        #     env=envs_train[next(scenarios_iter)],  # 环境
        #     verbose=1,  # 当verbose=1时，带进度条的输出日志信息
        #     tensorboard_log=config["logdir"] / "tensorboard",  # 输出log日志文件的地址
        #     **network.combined_extractor(config),  # 传入network里的几个参数：
        #     # policy —— 选择MlpPolicy, CnnPolicy和MultiInputPolicy中的一个
        #     # policy_kwargs —— 附加参数
        #     # target_kl —— 模型更新之间kl散度的限制阈值
        # )


        for index in range(config["epochs"]):
            scen = next(scenarios_iter)
            env_train = envs_train[scen]
            # 在这里env_train里就包含了观测空间和动作空间(gym.Env)
            # env_train.observation_space
            # env_train.action_space
            env_eval = envs_eval[scen]
            print(f"\nTraining on {scen}.\n")
            checkpoint_callback = CheckpointCallback(  # 保存模型的格式和保存频率
                save_freq=config["checkpoint_freq"],
                save_path=config["logdir"] / "checkpoint",
                name_prefix=f"{config['alg']}_{index}",
            )
            model.set_env(env_train)
            model.learn(
                total_timesteps=config["train_steps"],
                callback=[checkpoint_callback],
            )
            ##############每一轮的reward怎么收集？？？？？？？？？？？

        # Save trained model.
        save_dir = config["logdir"] / "train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir / ("model_" + time))  # model.save函数在base_class.py第787行
        print("\nSaved trained model.\n")

    if config["mode"] == "evaluate":
        print("\nEvaluate policy.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        for env_name, env_eval in envs_eval.items():
            print(f"\nEvaluating env {env_name}.")
            mean_reward, std_reward = evaluate_policy( # 输入train保存的模型，在同样的场景下评估，输出得分的均值和标准差
                model, env_eval, n_eval_episodes=config["eval_eps"], deterministic=True
            )
            print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
        print("\nFinished evaluating.\n")


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
