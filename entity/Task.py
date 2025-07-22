import random
# 任务类
class Task:
    def __init__(self, id,taskflow_id,runtime,cpu_require, ram_require, gpu_require,arrivetime):
        self.id = id
        self.global_id = f"TF{taskflow_id}-T{id}"
        self.uploadSize = 300 #文件大小
        self.outputFileSize = 300
        #self.uploadtime = 0  # 上传时间
        #self.outputtime = 0  # 下载时间
        #self.transtime = 0  # 传输时间
        self.informationshare= None #给后继节点的信息分享量
        self.finish=False #0表示该任务未完成，初始设为0
        self.predecessor =[]# 前继的下标信息
        self.descendant= []  # 后继的下标信息
        self.taskflow_id=taskflow_id
        self.runtime = runtime #与任务总量有关
        #self.exetime = 0 #任务的执行时间
        self.arrivetime = arrivetime #到达时间
        #self.starttime = None #（到达时间与开始时间的差值为等待时间） 任务的开始时间就是最终的present_time
        self.endtime = None
        self.get_bandwidth=None
        self.present_time=None  #任务在整个工作流中的当前时间（不断更新的# ）
        self.cpu_require = cpu_require #要求核数
        self.ram_require = ram_require #GB
        self.gpu_require = gpu_require #百分比

        #self.cpu_allocated = cpu_require  # 要求核数
        #self.ram_allocated = ram_require  # GB
        #self.gpu_allocated = gpu_require  # 百分比
        self.node = None


        self.upward_rank=0 #每个任务的计算量为runtime/cpu_require*gpu_require

    # def __repr__(self):
    #     return (f"<Global id={self.global_id},  arrive={self.arrivetime:.2f}, "
    #             f"cpu={self.cpu_require}, ram={self.ram_require}, gpu={self.gpu_require}, "
    #             f"end={'{:.2f}'.format(self.endtime) if self.endtime else 'None'}, "
    #             f"on_node={self.node.id if self.node else 'None'}>")

    def __repr__(self):
        return (
            f"<Task {self.global_id} |\n"
            f"Arrive: {self.arrivetime:} | Present: {self.present_time:} | "
            f"Runtime: {self.runtime} | End: {self.endtime if self.endtime else 'None'} \n"
            f"CPU: {self.cpu_require} | RAM: {self.ram_require} | GPU: {self.gpu_require}\n"
            f"On Node: {self.node.id if self.node else 'None'} | Flow: TF{self.taskflow_id}\n"
            f"Upload: {self.uploadSize} | Output: {self.outputFileSize} | Finish: {self.finish}\n"
            f">"
        )

    def want_get_bandwidth(self,node):#seed随机种子,设置为node的下标乘上taskflow_id
        seed=self.taskflow_id * node.id
        rng=random.Random(seed)
        if node.node_type == "CloudNode":
            self.get_bandwidth = rng.randint(40,60) #MHZ
        elif node.node_type == "FogNode":
            self.get_bandwidth = rng.randint(80,100)
        elif node.node_type == "EdgeNode":
            self.get_bandwidth = rng.randint(120,150)

