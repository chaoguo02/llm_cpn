def get_fifo_node_score(node):
    return -node.begin_idle_time

def get_fifo_task_score(task):
    return -task.arrivetime