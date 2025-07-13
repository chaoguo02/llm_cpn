from entity.Node import Node

def createNode(node_num):
    proportions = [0.6, 0.3, 0.1]
    node_types = ["EdgeNode", "FogNode", "CloudNode"]

    # 各类型节点属性列表（按顺序：Edge, Fog, Cloud）
    gpu_process     = [10000,   100000,   1000000]
    cpu_process      = [10000,   100000,   500000]
    ram_capacities  = [100,     250,      512]
    prices          = [0.0000029, 0.0000229, 0.000115]

    bandwidths      = [100,     100,      100]
    delays          = [100,     100,      100]
    cpu_capacities  = [100,     100,      100]
    gpu_capacities  = [100,     100,      100]

    node_counts = [int(p * node_num) for p in proportions]

    nodes = []
    node_id = 0

    for i in range(len(node_types)):
        for _ in range(node_counts[i]):
            node = Node(
                id=node_id,
                gpu_process=gpu_process[i],
                cpu_process=cpu_process[i],
                bandwidth=bandwidths[i],
                delay=delays[i],
                cpu_capacity=cpu_capacities[i],
                ram_capacity=ram_capacities[i],
                gpu_capacity=gpu_capacities[i],
                node_type=node_types[i],
                price=prices[i]
            )
            nodes.append(node)
            node_id += 1

    return nodes

nodes = createNode(100)

for node in nodes:
    print(f"ID: {node.id}, Type: {node.node_type}, GPU: {node.gpu_process}, CPU: {node.cpu_process}, "
          f"RAM: {node.ram_capacity}, Price: {node.price}, Bandwidth: {node.bandwidth}, Delay: {node.delay}")
