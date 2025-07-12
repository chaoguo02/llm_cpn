import random
# 定义一个表示计算节点的类
# 节点类

"""
self.type = {"CloudNode","EdgeNode","FogNode"}
"""
class Node:
    def __init__(self, id, cpu_process,gpu_process, bandwidth,delay,cpu_capacity, ram_capacity, gpu_capacity,node_type):
        self.id = id
        self.node_type = node_type
        self.cpu_process = cpu_process #cpu单核计算能力
        self.gpu_process = gpu_process #gpu单核计算能力
        self.cpu_capacity = cpu_capacity #cpu核数
        self.ram_capacity = ram_capacity #ram容量
        self.gpu_capacity = gpu_capacity #gpu核数

        self.cpu_available = cpu_capacity
        self.ram_available = ram_capacity
        self.gpu_available = gpu_capacity
        self.bandwidth = bandwidth #带宽
        self.delay = delay
        self.tasks = []  # 当前节点的任务列表
       # self.waiting_queue = []  # 等待队列



    def assign_task(self, task):
        self.tasks.append(task)
        self.ram_available -= task.ram_require
        self.cpu_available -= task.cpu_require
        self.gpu_available -= task.gpu_require
    def release_task(self,task):
        self.tasks.remove(task)
        self.ram_available += task.ram_require
        self.cpu_available += task.cpu_require
        self.gpu_available += task.gpu_require

    def display_value(self):
        print(self.cpu_capacity)
"""
    def find_compatible_task(self):
        在等待队列中找到一个适合当前节点资源的任务
        for i, task in enumerate(self.waiting_queue):
            if (self.cpu_available >= task.cpu_require and
                    self.ram_available >= task.ram_require and
                    self.gpu_available >= task.gpu_require):
                compatible_task = self.waiting_queue.pop(i)  # 取出第一个符合条件的任务
                print(f">>>>>>>>>>节点 {self.id} 从等待队列中取出任务 {compatible_task.id} 执行")
                #print("节点信息：", self.id, self.cpu_available, self.ram_available, self.gpu_available)
                #print(">>>>>>>>>>取出任务信息", compatible_task.id, compatible_task.cpu_require, compatible_task.ram_require,
                      #compatible_task.gpu_require)
                return compatible_task
            else:
                return None

    def assign_task(self, task):
        if self.cpu_available >= task.cpu_require and self.ram_available >= task.ram_require and self.gpu_capacity >= task.gpu_require:
            self.tasks.append(task)
            self.cpu_available -= task.cpu_require
            self.ram_available -= task.ram_require
            self.gpu_available -= task.gpu_require
            task.node = self
            task.uploadtime = task.uploadSize/self.bandwidth
            print(f">>>>>>>>>>任务 {task.id} 被分配到节点 {self.id}")

    def release_task(self, task,nodes,exeTaskList,current_time):
        # 释放资源
        self.tasks.remove(task)
        self.cpu_available += task.cpu_require
        self.ram_available += task.ram_require
        self.gpu_available += task.gpu_require
        task.outputtime = task.outputFileSize / self.bandwidth
        print(f">>>>>>>>>>任务 {task.id} 在节点 {self.id} 完成，释放资源")

        # 检查等待队列是否有任务可以执行
        if self.waiting_queue:
            next_task = self.find_compatible_task()
            if next_task is not None:
                # 分配任务
                self.assign_task(next_task)
                exeTaskList.append(next_task)
                start_event = ev.Event(current_time, 'start', next_task, self)  # 假设 current_time 为 0
                ev.add_event(start_event)
            else:
                # 调度策略baseline：无法找到兼容任务，迁移第一个任务
                first_task = self.waiting_queue.pop(0) #已经弹出来任务了，在后面不需要迁移
                print(f">>>>>>>>>>节点 {self.id} 调度任务 {first_task.id} 到其他节点")
                migrated = self.migrate_task(first_task, nodes,exeTaskList,current_time)
                if not migrated:
                    # 如果迁移失败，重新放回等待队列
                    self.waiting_queue.insert(0, first_task)

    def migrate_task(self, task, nodes,exeTaskList,current_time):
        print("************有迁移***********")
        将任务迁移到其他节点的等待队列
        调度后也要判断是否要执行任务
        for node in nodes:
            if node != self:  # 跳过自身节点
                if node.cpu_available >= task.cpu_require and node.ram_available >= task.ram_require and node.gpu_available >= task.gpu_require:
                    node.add_to_waiting_queue(task) #添加该任务
                    if node.waiting_queue:
                        next_task = node.find_compatible_task()
                        if next_task is not None:
                            # 分配任务
                            node.assign_task(next_task)
                            next_task.transtime = next_task.runtime/ (node.delay + self.delay)
                            exeTaskList.append(next_task)
                            start_event = ev.Event(current_time, 'start', next_task, node)  # 假设 current_time 为 0
                            ev.add_event(start_event)
                    #node.assign_task(task)
                    print(f">>>>>>>>>>任务 {task.id} 从节点 {self.id} 被调度到节点 {node.id} 的等待队列")

                    return True
        print(f">>>>>>>>>>任务 {task.id} 无法迁移，所有节点资源不足")
        return False

    def add_to_waiting_queue(self, task):
        将任务按照 FIFO 策略加入节点的等待队列
        self.waiting_queue.append(task)
        print(f">>>>>>>>>>任务 {task.id} 加入节点 {self.id} 的等待队列")

    def move_to_waiting_queue(self, task):
        将任务按照 FIFO 策略加入节点的等待队列
        self.waiting_queue.remove(task)
        print(f">>>>>>>>>>任务 {task.id} 被移出节点 {self.id} 的等待队列")
"""