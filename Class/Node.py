import random
# 定义一个表示计算节点的类
class Node:
    def __init__(self, id, compution_abilty, ram_capacity):
        self.id = id
        self.compution_abilty = compution_abilty
        self.ram_capacity = ram_capacity
        #self.gpu_capacity = gpu_capacity
        self.ram_available = ram_capacity
        #self.gpu_available = gpu_capacity
        self.tasks = []  # 当前节点的任务列表

    def assign_task(self, task):
        self.tasks.append(task)
        self.ram_available -= task.ram_capacity
        #self.gpu_available -= task.gpu_capacity
    def release_task(self,task):
        self.tasks.remove(task)
        self.ram_available += task.ram_capacity

    def display_value(self):
        print(self.cpu_capacity)