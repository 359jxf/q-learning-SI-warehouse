import numpy as np
import time
import sys
from SharedMap import SharedMap

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20   # pixels
MAZE_H = 21  # grid height
MAZE_W = 21 # grid width

#shelf coordinates
X_Block_pic = [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
X_Block = [element * UNIT for element in X_Block_pic]
Y_Block_pic = [5,6,8,9,11,12,14,15,17,18]
Y_Block = [element * UNIT for element in Y_Block_pic]

origin1 = np.array([70, 50])
origin2 = np.array([210,50])
origin3 = np.array([350,50])

class Maze_display(tk.Tk, object):
    def __init__(self):
        super(Maze_display, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r','w'] #up, down, left, right, wait
        self.n_actions = len(self.action_space)
        self.title('Warehouse')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        
        # 动态目标管理
        self.targets = {}  # {target_id: canvas_item_id}
        self.target_positions = {}  # {target_id: position}
        self.available_targets = []  # 可用目标列表
        self.completed_targets = []  # 已完成目标
        self.robot_targets = {'robot1': None, 'robot2': None, 'robot3': None}  # 机器人当前目标
        self.target_counter = 0  # 目标计数器
        
        # 初始化共享地图
        self.shared_map = SharedMap(MAZE_W, MAZE_H, UNIT)
        
        # 机器人路径规划
        self.robot_paths = {'robot1': [], 'robot2': [], 'robot3': []}
        self.path_index = {'robot1': 0, 'robot2': 0, 'robot3': 0}
        
        self._build_maze()

   
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='oldlace',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
        # create operation desks
        for i in range (1,16,7):
            self.canvas.create_rectangle(i*UNIT, 0, (i+5)*UNIT, 2*UNIT, fill = 'sandybrown')
        
        # create shelves
        for k in range (1,17,5):
            for i in range (5,18,3):
                self.canvas.create_rectangle(k*UNIT, i*UNIT, (k+4)*UNIT,(i+2)*UNIT,fill='bisque4')

        # 初始化时不创建目标，等待用户添加
        
        # define starting points       
        self.org1 = self.canvas.create_rectangle(
            origin1[0] - 10, origin1[1] - 10,
            origin1[0] + 10, origin1[1] + 10)  
        self.org2 = self.canvas.create_rectangle(
            origin2[0] - 10, origin2[1] - 10,
            origin2[0] + 10, origin2[1] + 10) 
        self.org3 = self.canvas.create_rectangle(
            origin3[0] - 10, origin3[1] - 10,
            origin3[0] + 10, origin3[1] + 10)         

        #create robot1
        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 10, origin1[1] - 10,
            origin1[0] + 10, origin1[1] + 10,
            fill='SkyBlue1')
        #create robot2
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 10, origin2[1] - 10,
            origin2[0] + 10, origin2[1] + 10,
            fill='SteelBlue2')
        #create robot3
        self.rect3 = self.canvas.create_rectangle(
            origin3[0] - 10, origin3[1] - 10,
            origin3[0] + 10, origin3[1] + 10,
            fill='RoyalBlue1' )
        # pack all
        self.canvas.pack()

    def resetRobot(self):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)
        self.canvas.delete(self.rect3)
        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 10, origin1[1] - 10,
            origin1[0] + 10, origin1[1] + 10,
            fill='SkyBlue1')
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 10, origin2[1] - 10,
            origin2[0] + 10, origin2[1] + 10,
            fill='SteelBlue2')   
        self.rect3 = self.canvas.create_rectangle(
            origin3[0] - 10, origin3[1] - 10,
            origin3[0] + 10, origin3[1] + 10,
            fill='RoyalBlue1' )
        
        # 不重置目标，保持用户设置的目标
        self.completed_targets = []
        self.robot_targets = {'robot1': None, 'robot2': None, 'robot3': None}
        
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2), self.canvas.coords(self.rect3)
    
    def step_multi(self, actions):
        """同步执行三个机器人的动作"""
        obs1_, r1, d1 = self.step1(actions[0])
        obs2_, r2, d2 = self.step2(actions[1])
        obs3_, r3, d3 = self.step3(actions[2])
        return (obs1_, r1, d1), (obs2_, r2, d2), (obs3_, r3, d3)

    def returnStep1(self, action): 
        s = self.canvas.coords(self.rect1)
        base_action = moveAgent(s, action)
        self.canvas.move(self.rect1, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect1)  # next state
      
        # reward function 
        
        if s_ == self.canvas.coords(self.org1):
            reward = 50
            done = 'arrive'
            s_ = 'terminal'   
        elif any(s_ == self.canvas.coords(self.targets[tid]) for tid in self.targets if tid in self.targets):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_ == self.canvas.coords(self.org2) or s_ == self.canvas.coords(self.org3):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_[0] == 0 or s_[1] < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        else:
            reward = 0
            done = 'nothing'
              
        return s_, reward, done

    def returnStep2(self, action): 
        s = self.canvas.coords(self.rect2)
        base_action = moveAgent(s, action)
        self.canvas.move(self.rect2, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect2)  # next state
      
        # reward function 
        
        if s_ == self.canvas.coords(self.org2):
            reward = 50
            done = 'arrive'
            s_ = 'terminal'   
        elif any(s_ == self.canvas.coords(self.targets[tid]) for tid in self.targets if tid in self.targets):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_ == self.canvas.coords(self.org1) or s_ == self.canvas.coords(self.org3):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_[0] == 0 or s_[1] < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        else:
            reward = 0
            done = 'nothing'
              
        return s_, reward, done
    
    def returnStep3(self, action): 
        s = self.canvas.coords(self.rect3)
        base_action = moveAgent(s, action)
        self.canvas.move(self.rect3, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect3)  # next state
      
        # reward function 
        
        if s_ == self.canvas.coords(self.org3):
            reward = 50
            done = 'arrive'
            s_ = 'terminal'   
        elif any(s_ == self.canvas.coords(self.targets[tid]) for tid in self.targets if tid in self.targets):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_ == self.canvas.coords(self.org1) or s_ == self.canvas.coords(self.org2):
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif s_[0] == 0 or s_[1] < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        else:
            reward = 0
            done = 'nothing'
              
        return s_, reward, done    
    
    def step1(self, action, obstacle=None):
        s = self.canvas.coords(self.rect1)
        base_action = moveAgent(s, action)
        next_coords = [s[0] + base_action[0], s[1] + base_action[1], s[2] + base_action[0], s[3] + base_action[1]]
        # 检查是否有其他机器人
        other1 = self.canvas.coords(self.rect2)
        other2 = self.canvas.coords(self.rect3)
        if next_coords == other1 or next_coords == other2:
            return s, -50, 'hit'
        self.canvas.move(self.rect1, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect1)  # next state
      
        # reward function
        done = 'nothing'
        reward = 0
        
        # 检查是否到达任何可用目标
        arrived_target = None
        for target_id in self.available_targets:
            if target_id in self.targets:
                target_coords = self.canvas.coords(self.targets[target_id])
                if s_ == target_coords:
                    arrived_target = target_id
                    break
        
        if arrived_target:
            reward = 50
            done = 'arrive'
            self.completed_targets.append(arrived_target)
            self.available_targets.remove(arrived_target)
            
            # 从画布上删除目标
            if arrived_target in self.targets:
                self.canvas.delete(self.targets[arrived_target])
                del self.targets[arrived_target]
                del self.target_positions[arrived_target]
                # 从共享地图中删除
                if arrived_target in self.shared_map.targets:
                    del self.shared_map.targets[arrived_target]
            
            if self.robot_targets['robot1'] == arrived_target:
                self.robot_targets['robot1'] = None
                # 清除路径缓存
                self.robot_paths['robot1'] = []
                self.path_index['robot1'] = 0
            
            # 合作奖励：如果这是最后一个目标，额外奖励
            if len(self.available_targets) == 0:
                reward += 20
            # 不返回terminal，让机器人可以继续移动
        elif s_[0] == 0 or s_[1] < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        
        if obstacle != None:
            if s_ == obstacle:
                reward = -50
                done = 'hit'
                s_ = 'terminal'
        
        return s_, reward, done
    
    def step2(self, action, obstacle=None):
        s = self.canvas.coords(self.rect2)
        base_action = moveAgent(s, action)
        next_coords = [s[0] + base_action[0], s[1] + base_action[1], s[2] + base_action[0], s[3] + base_action[1]]
        # 检查是否有其他机器人
        other1 = self.canvas.coords(self.rect1)
        other2 = self.canvas.coords(self.rect3)
        if next_coords == other1 or next_coords == other2:
            return s, -50, 'hit'
        self.canvas.move(self.rect2, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect2)  # next state
      
        # reward function
        done = 'nothing'
        reward = 0
        
        # 检查是否到达任何可用目标
        arrived_target = None
        for target_id in self.available_targets:
            if target_id in self.targets:
                target_coords = self.canvas.coords(self.targets[target_id])
                if s_ == target_coords:
                    arrived_target = target_id
                    break
        
        if arrived_target:
            reward = 50
            done = 'arrive'
            self.completed_targets.append(arrived_target)
            self.available_targets.remove(arrived_target)
            
            # 从画布上删除目标
            if arrived_target in self.targets:
                self.canvas.delete(self.targets[arrived_target])
                del self.targets[arrived_target]
                del self.target_positions[arrived_target]
                # 从共享地图中删除
                if arrived_target in self.shared_map.targets:
                    del self.shared_map.targets[arrived_target]
            
            if self.robot_targets['robot2'] == arrived_target:
                self.robot_targets['robot2'] = None
                # 清除路径缓存
                self.robot_paths['robot2'] = []
                self.path_index['robot2'] = 0
            
            # 合作奖励：如果这是最后一个目标，额外奖励
            if len(self.available_targets) == 0:
                reward += 20
            # 不返回terminal，让机器人可以继续移动
        elif s_[0] == 0 or int(s_[1]) < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
                
        if obstacle != None:
            if s_ == obstacle:
                reward = -50
                done = 'hit'
                s_ = 'terminal'

        return s_, reward, done      

    def step3(self, action, obstacle=None):
        s = self.canvas.coords(self.rect3)
        base_action = moveAgent(s, action)
        next_coords = [s[0] + base_action[0], s[1] + base_action[1], s[2] + base_action[0], s[3] + base_action[1]]
        # 检查是否有其他机器人
        other1 = self.canvas.coords(self.rect1)
        other2 = self.canvas.coords(self.rect2)
        if next_coords == other1 or next_coords == other2:
            return s, -50, 'hit'
        self.canvas.move(self.rect3, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect3)  # next state
      
        # reward function
        done = 'nothing'
        reward = 0
        
        # 检查是否到达任何可用目标
        arrived_target = None
        for target_id in self.available_targets:
            if target_id in self.targets:
                target_coords = self.canvas.coords(self.targets[target_id])
                if s_ == target_coords:
                    arrived_target = target_id
                    break
        
        if arrived_target:
            reward = 50
            done = 'arrive'
            self.completed_targets.append(arrived_target)
            self.available_targets.remove(arrived_target)
            
            # 从画布上删除目标
            if arrived_target in self.targets:
                self.canvas.delete(self.targets[arrived_target])
                del self.targets[arrived_target]
                del self.target_positions[arrived_target]
                # 从共享地图中删除
                if arrived_target in self.shared_map.targets:
                    del self.shared_map.targets[arrived_target]
            
            if self.robot_targets['robot3'] == arrived_target:
                self.robot_targets['robot3'] = None
                # 清除路径缓存
                self.robot_paths['robot3'] = []
                self.path_index['robot3'] = 0
            
            # 合作奖励：如果这是最后一个目标，额外奖励
            if len(self.available_targets) == 0:
                reward += 20
            # 不返回terminal，让机器人可以继续移动
        elif s_[0] == 0 or s_[1] < 40 or s_[2] >= MAZE_H * UNIT or s_[3] >= MAZE_W * UNIT:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
        elif int(s_[0]) in X_Block and int(s_[1]) in Y_Block:
            reward = -50
            done = 'hit'
            s_ = 'terminal'
       
        if obstacle != None:
            if s_ == obstacle:
                reward = -50
                done = 'hit'
                s_ = 'terminal'
          
        return s_, reward, done

    def add_target(self, x, y, color='gold'):
        """在指定位置添加目标"""
        # 检查位置是否有效
        if not self.shared_map.is_accessible((x, y)):
            return False
        
        # 创建目标ID
        target_id = f"target_{self.target_counter}"
        self.target_counter += 1
        
        # 在画布上创建目标
        canvas_id = self.canvas.create_rectangle(
            x*UNIT, y*UNIT, (x+1)*UNIT, (y+1)*UNIT,
            fill=color
        )
        
        # 记录目标信息
        self.targets[target_id] = canvas_id
        self.target_positions[target_id] = [x*UNIT+10, y*UNIT+10]
        self.available_targets.append(target_id)
        
        # 更新到共享地图
        self.shared_map.update_target(target_id, self.target_positions[target_id])
        
        return target_id
    
    def remove_target(self, target_id):
        """移除指定目标"""
        if target_id in self.targets:
            # 从画布移除
            self.canvas.delete(self.targets[target_id])
            
            # 清理数据
            del self.targets[target_id]
            del self.target_positions[target_id]
            
            if target_id in self.available_targets:
                self.available_targets.remove(target_id)
            
            # 清理机器人目标分配
            for robot in self.robot_targets:
                if self.robot_targets[robot] == target_id:
                    self.robot_targets[robot] = None
                    self.robot_paths[robot] = []
                    self.path_index[robot] = 0
            
            # 从共享地图移除
            if target_id in self.shared_map.targets:
                del self.shared_map.targets[target_id]
    
    def clear_all_targets(self):
        """清除所有目标"""
        target_ids = list(self.targets.keys())
        for target_id in target_ids:
            self.remove_target(target_id)
    
    def get_manhattan_distance(self, pos1, pos2):
        """计算两个位置之间的曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_map_based_action(self, robot_name, robot_pos):
        """基于共享地图获取最优动作"""
        # 更新机器人位置到地图
        self.shared_map.update_dynamic_obstacle(robot_name, robot_pos)
        
        # 如果当前没有分配目标，返回None
        if self.robot_targets[robot_name] is None:
            return None
        
        # 检查是否有缓存的路径
        if robot_name in self.robot_paths and self.robot_paths[robot_name]:
            path = self.robot_paths[robot_name]
            index = self.path_index[robot_name]
            
            if index < len(path):
                action = path[index]
                self.path_index[robot_name] += 1
                return action
        
        # 计算新路径（排除自己作为障碍物）
        target_name = self.robot_targets[robot_name]
        if target_name in self.shared_map.targets:
            target_pixel = self.shared_map.grid_to_pixel(self.shared_map.targets[target_name])
            path = self.shared_map.find_path(robot_pos, target_pixel, 
                                           ignore_dynamic=False, 
                                           exclude_self=robot_name)
            
            if path:
                self.robot_paths[robot_name] = path
                self.path_index[robot_name] = 1
                return path[0] if path else 4  # 返回第一个动作
        
        return None  # 无法找到路径
    
    def assign_nearest_target(self, robot_name, robot_pos):
        """为机器人分配最近的可用目标，避免冲突"""
        if not self.available_targets:
            return None
        
        # 获取其他机器人已经分配的目标
        assigned_targets = set()
        for r_name, target in self.robot_targets.items():
            if r_name != robot_name and target is not None:
                assigned_targets.add(target)
        
        # 找出未被分配的目标
        unassigned_targets = [t for t in self.available_targets if t not in assigned_targets]
        
        # 如果没有未分配的目标，根据机器人数量决定
        if not unassigned_targets:
            # 如果目标少于机器人，有些机器人不分配目标
            if len(self.available_targets) < 3:
                return None
            # 否则选择最近的目标（可能与其他机器人冲突）
            unassigned_targets = self.available_targets
        
        min_distance = float('inf')
        nearest_target = None
        
        for target_id in unassigned_targets:
            if target_id in self.targets:
                target_coords = self.canvas.coords(self.targets[target_id])
                distance = self.get_manhattan_distance(robot_pos, target_coords)
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target_id
        
        self.robot_targets[robot_name] = nearest_target
        
        # 清除该机器人的路径缓存
        self.robot_paths[robot_name] = []
        self.path_index[robot_name] = 0
        
        return nearest_target

    def render(self):
        time.sleep(0.01)
        self.update()

    
def moveAgent(s, action):
    base_action = np.array([0, 0])
    if action == 0:   # up
        if s[1] > UNIT:
            base_action[1] -= UNIT
    elif action == 1:   # down
        if s[1] < (MAZE_H - 1) * UNIT:
            base_action[1] += UNIT
    elif action == 2:   # right
        if s[0] < (MAZE_W - 1) * UNIT:
            base_action[0] += UNIT
    elif action == 3:   # left
        if s[0] > UNIT:
            base_action[0] -= UNIT
    elif action == 4:   # wait
        base_action = np.array([0, 0])
    return base_action

def update():
    for t in range(10):
        s1, s2 = env.resetRobot()
        while True:
            env.render()
                      
            s1,r1, done1 = env.step1(2)
            s2,r2,done2 = env.step2(1)
            
            if done1 == 'hit' and done2 == 'hit':
                break
            elif done1 == 'hit' and done2 == 'nothing':
                break
            elif done1 == 'arrive' and done2 == 'arrive':
                break
            elif s1 == s2:
                break

if __name__ == '__main__':
    env = Maze_display()
    env.after(2000, update)
    env.mainloop()