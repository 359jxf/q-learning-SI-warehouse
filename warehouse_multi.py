from interface_maze import Maze_display, MAZE_W, MAZE_H, UNIT
import numpy as np
import os
import pandas as pd
from SharedMap import SharedMap
import matplotlib.pyplot as plt


class NewQLearningTable:
    def __init__(self, actions, q_table_path=None):
        self.actions = actions
        if q_table_path and os.path.exists(q_table_path):
            self.q_table = pd.read_pickle(q_table_path)
            print(f"Loaded Q-table from {q_table_path}")
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 为新状态添加一行，初始化为0
            self.q_table.loc[state] = [0.0] * len(self.actions)
    
    def learn(self, s, a, r, s_, alpha=0.1, gamma=0.9):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        
        if s_ != 'terminal':
            q_target = r + gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        
        self.q_table.loc[s, a] += alpha * (q_target - q_predict)

class InteractiveWarehouse:
    def __init__(self):
        self.env = Maze_display()
        self.RL1 = NewQLearningTable(actions=list(range(self.env.n_actions)), q_table_path=None)
        self.RL2 = NewQLearningTable(actions=list(range(self.env.n_actions)), q_table_path=None)
        self.RL3 = NewQLearningTable(actions=list(range(self.env.n_actions)), q_table_path=None)
        self.fixed_targets = [(5, 5), (7, 10), (15, 5)]
        self.max_episodes = 1000
        self.epsilon_start = 0.8
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = int(self.max_episodes * 0.8)  # 衰减到最小值所需轮数
        self.sharemap = SharedMap(width=MAZE_W, height=MAZE_H, unit=UNIT)
        self.robot_paths = {'robot1': None, 'robot2': None, 'robot3': None}
        self.executing_robots = set()
        self.total_steps = 0
        self.robot_steps = {'robot1': 0, 'robot2': 0, 'robot3': 0}
        self.completed_targets_this_episode = set()
        self.robot_target = {'robot1': None, 'robot2': None, 'robot3': None}  # 记录每个机器人当前目标
        self.lambda_weight = 0.5
        self.train_all_episodes()

    def add_fixed_targets(self):
        for x, y in self.fixed_targets:
            self.env.add_target(x, y)

    def calculate_reward(self, step_count, done, prev_obs=None, curr_obs=None, target_id=None):
        """
        - 轮数最少
        - 到达目标
        - 避免碰撞
        """
        reward = 0
        reward_info = {
            'step_penalty': 0,
            'arrive_reward': 0,
            'hit_penalty': 0,
        }
        
        reward_info['step_penalty'] = -1
        reward += reward_info['step_penalty']
        # 到达目标
        if done == 'arrive':
            reward_info['arrive_reward'] = 40
            reward += 40
        if done == 'hit':
            reward_info['hit_penalty'] = -10
            reward += reward_info['hit_penalty']
        return reward, reward_info

    def get_action(self, RL, observation, available_targets, robot_name):
        best_target = None
        best_value = -np.inf
        if not available_targets:
            return 4, None
        # 目标分配
        if np.random.rand() < self.epsilon:
            best_target = np.random.choice(available_targets)
        else:
            for target_id in available_targets:
                target_pos = self.env.target_positions[target_id]
                obs_tuple = tuple(observation)
                completed_bitmap = ''.join(['1' if tid in self.completed_targets_this_episode else '0' for tid in self.env.target_positions])
                state_str = f"{obs_tuple}_{target_pos}_{completed_bitmap}"
                if state_str in RL.q_table.index:
                    value = RL.q_table.loc[state_str, :].max()
                else:
                    value = 0
                if value > best_value:
                    best_value = value
                    best_target = target_id
        self.robot_target[robot_name] = best_target
        # 路径规划：每一帧都实时A*规划一步
        if best_target is not None:
            start_pos = (observation[0], observation[1])
            target_pos = self.env.target_positions[best_target]
            start_grid = self.sharemap.pixel_to_grid(start_pos)
            target_grid = self.sharemap.pixel_to_grid(target_pos)
            if not self.sharemap.is_accessible(target_grid):
                return 4, best_target
            if not self.sharemap.is_accessible(start_grid):
                return 4, best_target
            actions = self.sharemap.find_path(start_pos, target_pos)
            next_action = actions[0] if actions else 4
            return next_action, best_target
        return 4, best_target

    def train_all_episodes(self):
        rewards_per_episode = []
        N = 100  # 检查收敛的窗口大小
        min_episodes = 200  # 至少训练这么多轮才检查收敛
        converge_var_threshold = 1.0  # 收敛判据：reward方差小于此值
        for episode in range(self.max_episodes):
            print(f"\n==== Episode {episode + 1} ====")
            reward_log = {
                'step_penalty': 0,
                'arrive_reward': 0,
                'hit_penalty': 0,
            }

            self.sharemap = SharedMap(width=MAZE_W, height=MAZE_H, unit=UNIT)
            self.env.clear_all_targets()
            self.add_fixed_targets()
            obs1, obs2, obs3 = self.env.resetRobot()
            freeze1 = freeze2 = freeze3 = False
            self.robot_paths = {'robot1': None, 'robot2': None, 'robot3': None}
            self.executing_robots.clear()
            self.total_steps = 0
            self.robot_steps = {'robot1': 0, 'robot2': 0, 'robot3': 0}
            self.completed_targets_this_episode = set()
            self.robot_target = {'robot1': None, 'robot2': None, 'robot3': None}
            step_count = 0
            max_steps = 1000
            episode_reward = 0
            prev_obs1, prev_obs2, prev_obs3 = obs1, obs2, obs3
            if episode < self.epsilon_decay_episodes:
                self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (episode / self.epsilon_decay_episodes)
            else:
                self.epsilon = self.epsilon_min
            while len(self.env.available_targets) > 0 and step_count < max_steps:
                self.env.render()
                # time.sleep(0.1) 
                step_count += 1

                a1, t1 = self.get_action(self.RL1, obs1, self.env.available_targets, 'robot1') if not freeze1 else (4, self.robot_target['robot1'])
                a2, t2 = self.get_action(self.RL2, obs2, self.env.available_targets, 'robot2') if not freeze2 else (4, self.robot_target['robot2'])
                a3, t3 = self.get_action(self.RL3, obs3, self.env.available_targets, 'robot3') if not freeze3 else (4, self.robot_target['robot3'])
                # 同步执行
                (obs1_, r1, d1), (obs2_, r2, d2), (obs3_, r3, d3) = self.env.step_multi([a1, a2, a3])
                
                # 机器人1
                if not freeze1:
                    self.robot_steps['robot1'] += 1
                    self.total_steps += 1
                # 机器人2
                if not freeze2:
                    self.robot_steps['robot2'] += 1
                    self.total_steps += 1
                # 机器人3
                if not freeze3:
                    self.robot_steps['robot3'] += 1
                    self.total_steps += 1
                completed_bitmap = ''.join(['1' if tid in self.completed_targets_this_episode else '0' for tid in self.env.target_positions])
                obs1_tuple = tuple(obs1)
                obs1_tuple_ = tuple(obs1_)
                obs2_tuple = tuple(obs2)
                obs2_tuple_ = tuple(obs2_)
                obs3_tuple = tuple(obs3)
                obs3_tuple_ = tuple(obs3_)
                reward1, info1 = self.calculate_reward(step_count, d1, obs1, obs1_, t1)
                reward2, info2 = self.calculate_reward(step_count, d2, obs2, obs2_, t2)
                reward3, info3 = self.calculate_reward(step_count, d3, obs3, obs3_, t3)
                # print(f"reward1: {reward1}, reward2: {reward2}, reward3: {reward3}")
                episode_reward += reward1 + reward2 + reward3
                # 累加各类 reward
                for key in reward_log:
                    reward_log[key] += info1[key] + info2[key] + info3[key]
                    
                # Q表更新（只更新各自目标，且目标必须还在target_positions）
                reward1_mixed = self.lambda_weight * reward1 + (1 - self.lambda_weight) * episode_reward
                reward2_mixed = self.lambda_weight * reward2 + (1 - self.lambda_weight) * episode_reward
                reward3_mixed = self.lambda_weight * reward3 + (1 - self.lambda_weight) * episode_reward

                if t1 is not None and t1 in self.env.target_positions:
                    target_pos1 = self.env.target_positions[t1]
                    state1_str = f"{obs1_tuple}_{target_pos1}_{completed_bitmap}"
                    state1_next_str = f"{obs1_tuple_}_{target_pos1}_{completed_bitmap}" if obs1_ != 'terminal' else 'terminal'
                    self.RL1.check_state_exist(state1_str)
                    if state1_next_str != 'terminal':
                        self.RL1.check_state_exist(state1_next_str)
                    self.RL1.learn(state1_str, a1, reward1_mixed, state1_next_str, 0.03, 0.9)
                if t2 is not None and t2 in self.env.target_positions:
                    target_pos2 = self.env.target_positions[t2]
                    state2_str = f"{obs2_tuple}_{target_pos2}_{completed_bitmap}"
                    state2_next_str = f"{obs2_tuple_}_{target_pos2}_{completed_bitmap}" if obs2_ != 'terminal' else 'terminal'
                    self.RL2.check_state_exist(state2_str)
                    if state2_next_str != 'terminal':
                        self.RL2.check_state_exist(state2_next_str)
                    self.RL2.learn(state2_str, a2, reward2_mixed, state2_next_str, 0.03, 0.9)
                if t3 is not None and t3 in self.env.target_positions:
                    target_pos3 = self.env.target_positions[t3]
                    state3_str = f"{obs3_tuple}_{target_pos3}_{completed_bitmap}"
                    state3_next_str = f"{obs3_tuple_}_{target_pos3}_{completed_bitmap}" if obs3_ != 'terminal' else 'terminal'
                    self.RL3.check_state_exist(state3_str)
                    if state3_next_str != 'terminal':
                        self.RL3.check_state_exist(state3_next_str)
                    self.RL3.learn(state3_str, a3, reward3_mixed, state3_next_str, 0.03, 0.9)
                prev_obs1, prev_obs2, prev_obs3 = obs1, obs2, obs3
                obs1, obs2, obs3 = obs1_, obs2_, obs3_
                # 检查是否到达目标
                for robot_name, obs, d in [('robot1', obs1, d1), ('robot2', obs2, d2), ('robot3', obs3, d3)]:
                    if d == 'arrive':
                        for target_id in list(self.env.available_targets):
                            target_pos = self.env.target_positions[target_id]
                            if abs(obs[0] - target_pos[0]) < UNIT and abs(obs[1] - target_pos[1]) < UNIT:
                                self.env.remove_target(target_id)
                                print(f"{robot_name} 到达目标 {target_id}")
                                # 清空所有机器人对该目标的分配
                                for robot in self.robot_target:
                                    if self.robot_target[robot] == target_id:
                                        self.robot_target[robot] = None
                                break
                if d1 == 'hit':
                    freeze1 = True
                    self.robot_paths['robot1'] = None
                    self.executing_robots.discard('robot1')
                if d2 == 'hit':
                    freeze2 = True
                    self.robot_paths['robot2'] = None
                    self.executing_robots.discard('robot2')
                if d3 == 'hit':
                    freeze3 = True
                    self.robot_paths['robot3'] = None
                    self.executing_robots.discard('robot3')
                if freeze1 and freeze2 and freeze3:
                    print("所有机器人撞墙退出")
                    break
                if step_count % 100 == 0:
                    print(f"Step {step_count}, 剩余目标: {len(self.env.available_targets)}, 总步数: {self.total_steps}, 当前累计reward: {episode_reward}")
            print(f"Episode {episode + 1} 完成，共进行 {step_count} 轮, 机器人总步数: {self.total_steps}, 剩余目标: {len(self.env.available_targets)}, 累计reward: {episode_reward}, 探索概率：{self.epsilon}")
            for k, v in reward_log.items():
                print(f"  {k:20s}: {v:.2f}")

            rewards_per_episode.append(episode_reward)

            # 收敛检测
            if episode + 1 >= min_episodes and (episode + 1) % N == 0:
                recent_rewards = rewards_per_episode[-N:]
                mean_reward = sum(recent_rewards) / N
                var_reward = sum((r - mean_reward) ** 2 for r in recent_rewards) / N
                print(f"[Converge Check] 最近{N}轮reward均值: {mean_reward:.2f}, 方差: {var_reward:.4f}")
                if var_reward < converge_var_threshold:
                    print(f"Reward已收敛，提前停止训练 (episode={episode+1})")
                    break

        print("\n训练完成，保存Q表...")
        os.makedirs('trained_models', exist_ok=True)
        self.RL1.q_table.to_pickle("trained_models/new_q_table1.pkl")
        self.RL2.q_table.to_pickle("trained_models/new_q_table2.pkl")
        self.RL3.q_table.to_pickle("trained_models/new_q_table3.pkl")
        print("新Q表已保存至 trained_models/")

        # 绘制reward曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_per_episode, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Curve per Episode')
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_curve.png')
        print('Reward curve saved as reward_curve.png')

if __name__ == "__main__":
    warehouse = InteractiveWarehouse()
    warehouse.env.mainloop()
