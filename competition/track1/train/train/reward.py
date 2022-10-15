from typing import Any, Dict

import gym
import numpy as np
import matplotlib.pyplot as plt


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_done in done.items():
            if agent_id != "__all__" and agent_done == True:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_road"]
                    | obs[agent_id]["events"]["off_route"]
                    | obs[agent_id]["events"]["on_shoulder"]
                    | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, done, info

    def _reward(
        self, obs: Dict[str, Dict[str, Any]], env_reward: Dict[str, np.float64]
    ) -> Dict[str, np.float64]:
        reward = {agent_id: np.float64(0) for agent_id in env_reward.keys()}
        reward_change = []

        for agent_id, agent_reward in env_reward.items():
            # Penalty for colliding
            if obs[agent_id]["events"]["collisions"]:
                penalty = 10
                reward[agent_id] -= np.float64(10)
                reward_change.append(reward[agent_id])
                print(f"{agent_id}: Collided. Penalty of collisions = {penalty}. Agent reward: {agent_reward}")
                break

            # Penalty for driving off road
            if obs[agent_id]["events"]["off_road"]:
                penalty = 10
                reward[agent_id] -= np.float64(10)
                reward_change.append(reward[agent_id])
                print(f"{agent_id}: Went off road. Penalty of off_road = {penalty}. Agent reward: {agent_reward}")
                break

            # Penalty for driving off route
            if obs[agent_id]["events"]["off_route"]:
                penalty = 10
                reward[agent_id] -= np.float64(10)
                reward_change.append(reward[agent_id])
                print(f"{agent_id}: Went off route. Penalty of off_route = {penalty}. Agent reward: {agent_reward}")
                break

            # Penalty for driving on road shoulder
            if obs[agent_id]["events"]["on_shoulder"]:
                penalty = 2
                reward[agent_id] -= np.float64(2)
                reward_change.append(reward[agent_id])
                print(f"{agent_id}: Went on shoulder. Penalty of on_shoulder = {penalty}. Agent reward: {agent_reward}")
                break

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                penalty = 10
                reward[agent_id] -= np.float64(10)
                reward_change.append(reward[agent_id])
                print(f"{agent_id}: Went wrong way. Penalty of wrong_way = {penalty}. Agent reward: {agent_reward}")
                break

            # Reward for reaching goal
            if obs[agent_id]["events"]["reached_goal"]:
                reward[agent_id] += np.float64(30)
                print(f"{agent_id}: Reached_goal! Reward of reach_goal = {30}. Agent reward: {agent_reward}")

            # Reward for distance travelled
            reward[agent_id] += np.float64(agent_reward)  ########agent_reward是路程奖励，只有当到达目标后，才给
            reward_change.append(reward[agent_id])

        plt.plot(np.arange(len(reward_change)), reward_change)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward change of an agent')
        plt.ion()
        plt.show()

        return reward
