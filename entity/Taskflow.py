import random
import networkx as nx
import matplotlib.pyplot as plt

from core_function.data_loader import AlibabaClusterReader, createTask
from core_function.update_time import computing_Task

"""
下标大的节点是下标小的节点的后继
只有下标小的节点完成后

"""
# 任务类
class TaskFlow:
    def __init__(self,id,all_arrive_time,genre,seed): #genre用于表示是上面种类的任务流,seed随机树种子
        # 随机生成节点数量
        self.rng=random.Random(seed)
        self.id=id
        self.num_tasks = self.rng.randint(1, 5)
        self.graph = nx.DiGraph()  # 有向图，保存任务的依赖关系
        self.all_arrive_time=all_arrive_time
        self.finish_time=0 #inf表示任务流失败
        #Filename = "Dataset/Alibaba/AlibabaCluster1.csv"
        Filename = "./Dataset/Alibaba/AlibabaCluster1.csv"
        if genre==0:
            dataSet = AlibabaClusterReader(Filename, self.num_tasks,0,70,seed) #在前百分之70中随机找num_tasks个数据
        elif genre==1:
            dataSet = AlibabaClusterReader(Filename, self.num_tasks,70,100,seed)#在后百分之30中随机找num_tasks个数据
        # 此处tasks是一个列表类型的数据
        self.tasks = createTask(dataSet,self.id,self.all_arrive_time)
        # 随机生成任务节点
        for task_id in range(self.num_tasks):
            self.graph.add_node(task_id, task=self.tasks[task_id])

        # self.tasks = createTask(dataSet, self.id, self.all_arrive_time)
        #
        # # 为每个 task 添加唯一 global_id，并加入图中
        # for task_id in range(self.num_tasks):
        #     task = self.tasks[task_id]
        #     task.global_id = f"TF{self.id}_T{task_id}"  # 全局唯一标识
        #     task.id = task_id  # 确保 id 字段是 taskflow 内的局部编号
        #     self.graph.add_node(task_id, task=task)

        # 随机生成依赖关系，确保图是无环的，并且每个节点至少有一个依赖关系
        self.generate_random_dependencies()
        self.generate_task_upward_rank()

    def generate_random_dependencies(self):
        # 获取节点列表
        nodes = list(self.graph.nodes)
        # 确保每个节点至少有一个依赖关系
        for node in nodes:
            # 获取当前节点能依赖的节点：只能依赖下标更大的节点
            possible_dependencies = [n for n in nodes if n > node]

            if possible_dependencies:
                m=self.rng.randint(1,2)
                # 随机选择n个下标更大的节点作为依赖
                for x in range(m):
                    dep = random.choice(possible_dependencies)
                    self.tasks[node].descendant.append(dep)
                    self.tasks[dep].predecessor.append(node)
                    self.graph.add_edge(node,dep)  # 添加依赖关系

    def find_descendant_avg_time(self,taskflows,task,nodes):#后继任务的平均完成时间（区分串行并行，单个任务的平均完成时间为任务跟每一个节点完成时间的平均值）
        all_queue=[]
        all_task_computing_time = []
        avg_computing_time=[]
        if len(task.descendant)==0:
            return 0
        all_queue.append(task.descendant)
        for i in task.descendant:
            sum1=0
            for j in nodes:
                sum1+=(computing_Task(taskflows[task.taskflow_id].tasks[i],j))
            time=sum1/len(nodes)
            avg_computing_time.append(time)
        all_task_computing_time.append(avg_computing_time)
        floor = 0
        while True:
            a=[]
            for x in all_queue[floor]:
                a.extend(taskflows[task.taskflow_id].tasks[x].descendant)
            if len(a)==0:
                break
            a = list(dict.fromkeys(a))
            for i in a:
                avg_computing_time = []
                sum1 = 0
                for j in nodes:
                    sum1 += (computing_Task(taskflows[task.taskflow_id].tasks[i], j))
                time = sum1 / len(nodes)
                avg_computing_time.append(time)
            all_task_computing_time.append(avg_computing_time)
            all_queue.append(a)
            floor+=1
        taskflow_descendant_time=sum(max(row) for row in all_task_computing_time)
        return taskflow_descendant_time

    def find_predecessor_is_zero(self):
        tasks1=[]
        for x in self.tasks:
            if len(x.predecessor)==0 and x.finish is False:
                tasks1.append(x)
        return tasks1
    def find_descendant_is_zero(self):
        tasks1=[]
        for x in self.tasks:
            if len(x.descendant)==0 and x.finish is False:
                tasks1.append(x)
        return tasks1
    def find_unfinish_task_number(self): #找到还未完成任务的数量
        k=0
        for task in self.tasks:
            if task.finish is False:
                k+=1
        return k
    def print_taskflow(self):
        for node, data in self.graph.nodes(data=True):
            task = data['task']
            print(
                f"Task {task.id} - Runtime: {task.runtime}, CPU: {task.cpu_require}, RAM: {task.ram_require} GB, GPU: {task.gpu_require}%")

        print("\nTask Dependencies (edges):")
        for u, v in self.graph.edges():
            print(f"Task {u} -> Task {v}")

    def draw_taskflow(self):
        # 使用 pygraphviz 和 graphviz dot 布局
        try:
            A = nx.nx_agraph.to_agraph(self.graph)  # 将 NetworkX 图转换为 Graphviz 图
            A.layout(prog='dot')  # 使用 Graphviz dot 布局进行优化
            A.draw('taskflow_graph.png')  # 将图形保存为图片文件

            # 展示生成的图形
            img = plt.imread('taskflow_graph.png')
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')  # 关闭坐标轴显示
            plt.show()
        except ImportError:
            print("pygraphviz is not installed or Graphviz is not available. Please install pygraphviz.")

    def generate_task_upward_rank(self):
        tasks=self.find_descendant_is_zero()
        for x in tasks:
            x.upward_rank = x.runtime / (x.cpu_require * x.gpu_require)
        while len(tasks)!=0:
            for x in tasks:
                for y in x.predecessor:
                    self.tasks[y].upward_rank=max(self.tasks[y].upward_rank,x.upward_rank+self.tasks[y].runtime/(self.tasks[y].cpu_require*self.tasks[y].gpu_require))
                    tasks.append(self.tasks[y])
                tasks.remove(x)







if __name__ == '__main__':
    taskflow = TaskFlow(1,20,0,80)
    for x in taskflow.tasks:
        print(x.upward_rank)
    taskflow.draw_taskflow()




