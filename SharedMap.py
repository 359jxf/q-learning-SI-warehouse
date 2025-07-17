import numpy as np
from collections import deque
import heapq

class SharedMap:
    def __init__(self, width, height, unit=20):
        """
        初始化共享地图
        width, height: 地图的网格数
        unit: 每个网格的像素大小
        """
        self.width = width
        self.height = height
        self.unit = unit
        
        # 地图状态：0=未知，1=可通行，2=障碍物，3=操作台
        self.map_grid = np.zeros((height, width), dtype=int)
        
        # 初始化已知的静态障碍物（货架和操作台）
        self._init_static_obstacles()
        
        # 动态障碍物（其他机器人和人类）
        self.dynamic_obstacles = {}
        
        # 目标位置
        self.targets = {}
        
        # 路径缓存
        self.path_cache = {}
        
    def _init_static_obstacles(self):
        """初始化静态障碍物（货架和操作台）"""
        # 操作台位置
        for i in range(1, 16, 7):
            for j in range(2):
                self.map_grid[j][i:i+6] = 3
        
        # 货架位置
        shelf_x = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
        shelf_y = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
        
        for x in shelf_x:
            for y in shelf_y:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.map_grid[y][x] = 2
        
        # 将其他区域标记为可通行
        for i in range(self.height):
            for j in range(self.width):
                if self.map_grid[i][j] == 0:
                    self.map_grid[i][j] = 1
    
    def pixel_to_grid(self, pixel_coords):
        """将像素坐标转换为网格坐标"""
        x = int(pixel_coords[0] / self.unit)
        y = int(pixel_coords[1] / self.unit)
        return (x, y)
    
    def grid_to_pixel(self, grid_coords):
        """将网格坐标转换为像素坐标"""
        x = grid_coords[0] * self.unit + self.unit // 2
        y = grid_coords[1] * self.unit + self.unit // 2
        return [x - 10, y - 10, x + 10, y + 10]
    
    def update_dynamic_obstacle(self, obstacle_id, position):
        """更新动态障碍物位置"""
        if position is None:
            # 移除障碍物
            if obstacle_id in self.dynamic_obstacles:
                del self.dynamic_obstacles[obstacle_id]
        else:
            grid_pos = self.pixel_to_grid(position)
            self.dynamic_obstacles[obstacle_id] = grid_pos
    
    def update_target(self, target_id, position):
        """更新目标位置"""
        grid_pos = self.pixel_to_grid(position)
        self.targets[target_id] = grid_pos
    
    def is_accessible(self, grid_pos, ignore_dynamic=False, exclude_self=None):
        """检查网格位置是否可通行"""
        x, y = grid_pos
        
        # 检查边界
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # 检查静态障碍物
        if self.map_grid[y][x] == 2 or (self.map_grid[y][x] == 3 and y >= 2):
            return False
        
        # 检查动态障碍物
        if not ignore_dynamic:
            for obs_id, obs_pos in self.dynamic_obstacles.items():
                if obs_id != exclude_self and obs_pos == grid_pos:
                    return False
        
        return True
    
    def get_neighbors(self, pos, ignore_dynamic=False, exclude_self=None):
        """获取可通行的邻居节点"""
        x, y = pos
        neighbors = []
        
        # 四个方向：上下左右
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if self.is_accessible(new_pos, ignore_dynamic, exclude_self):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos1, pos2):
        """启发式函数：曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_path(self, start_pixel, goal_pixel, ignore_dynamic=False, exclude_self=None):
        """
        使用A*算法寻找最优路径
        返回路径的动作序列：0=上，1=下，2=右，3=左，4=等待
        exclude_self: 要排除的机器人ID（自己）
        """
        start = self.pixel_to_grid(start_pixel)
        goal = self.pixel_to_grid(goal_pixel)
        
        # 检查缓存
        cache_key = (start, goal, ignore_dynamic)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # A*算法
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    prev = came_from[current]
                    # 计算动作
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    if dy == -1:
                        action = 0  # 上
                    elif dy == 1:
                        action = 1  # 下
                    elif dx == 1:
                        action = 2  # 右
                    elif dx == -1:
                        action = 3  # 左
                    else:
                        action = 4  # 等待
                    path.append(action)
                    current = prev
                
                path.reverse()
                self.path_cache[cache_key] = path
                return path
            
            for neighbor in self.get_neighbors(current, ignore_dynamic, exclude_self):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 没有找到路径
        return None

    