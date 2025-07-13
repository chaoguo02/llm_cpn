import operator

from core_function.operators import initIndividual, mutUniformListOfTrees, cxOnePointListOfTrees
from dirty_work.dirty_works import protect_div
from deap import gp, creator, base, tools
from core_function.evaluate import work_processing

def init_pset_toolbox(tournament_size):
    pset = []
    for idx in range(2):
        if idx == 0:
            pset1 = gp.PrimitiveSet("MAIN", 5)
            pset1.addPrimitive(operator.add, 2)
            pset1.addPrimitive(operator.sub, 2)
            pset1.addPrimitive(operator.mul, 2)
            pset1.addPrimitive(protect_div, 2)
            pset1.addPrimitive(operator.neg, 1)
            pset1.addPrimitive(max, 2)
            pset1.addPrimitive(min, 2)
            pset1.renameArguments(ARG0='TCT')  # 任务的计算时间 tct
            pset1.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间 twt
            pset1.renameArguments(ARG2='TNC')  # 任务的后继数量 tnc
            pset1.renameArguments(ARG3='TUT')  # 任务的上传时间  tut
            pset1.renameArguments(ARG4='FIFO1')
            pset.append(pset1)

        elif idx == 1:
            pset2 = gp.PrimitiveSet("MAIN", 6)
            pset2.addPrimitive(operator.add, 2)
            pset2.addPrimitive(operator.sub, 2)
            pset2.addPrimitive(operator.mul, 2)
            pset2.addPrimitive(protect_div, 2)
            pset2.addPrimitive(operator.neg, 1)
            pset2.addPrimitive(max, 2)
            pset2.addPrimitive(min, 2)
            pset2.renameArguments(ARG0='TCT')  # 任务的计算时间
            pset2.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间
            pset2.renameArguments(ARG2='TNC')  # 任务的后继数量
            pset2.renameArguments(ARG3='TPS')  # 任务的给后继传递消息的时间 tps
            pset2.renameArguments(ARG4='ACT')  # 后继节点的平均完成时间 act
            pset2.renameArguments(ARG5='FIFO2')
            pset.append(pset2)

    # 创建一个适应度类和个体类，个体由多棵树组成
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, min_=2, max_=4)
    toolbox.register("individual", initIndividual, creator.Individual, toolbox.expr, pset=pset, size=2)  # 假设我们创建2个树
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", work_processing)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, expr=toolbox.expr, pset=pset)
    toolbox.register("compile", gp.compile)
    return pset,toolbox