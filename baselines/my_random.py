import random

def decode1_random(individual, task, nodes, taskflows, pset):
    return random.choice(nodes)

def decode2_random(individual, node, taskflows, nodes, pset):
    return random.choice(node.waiting_queue)
