import operator
import random
from deap import gp

def initIndividual(container, func, pset, size):
    return container(gp.PrimitiveTree(func(pset[i])) for i in range(size))

def cxOnePointListOfTrees(ind1, ind2):
    print("type:", type(ind1))
    for idx, (tree1, tree2) in enumerate(zip(ind1, ind2)):
        print(f"\n===== 交叉处理第 {idx} 棵子树 =====")
        print("🔵 原始 tree1 (str):", str(tree1))
        print("🔵 原始 tree2 (str):", str(tree2))

        HEIGHT_LIMIT = 8
        dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
        tree1, tree2 = dec(gp.cxOnePoint)(tree1, tree2)
        print("🟢 交叉后 tree1_new (str):", str(tree1))
        print("🟢 交叉后 tree2_new (str):", str(tree2))
        ind1[idx], ind2[idx] = tree1, tree2

    return ind1, ind2

def mutUniformListOfTrees(individual, expr, pset):
    for idx, tree in enumerate(individual):
        HEIGHT_LIMIT = 8
        dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
        tree, = dec(gp.mutUniform)(tree, expr=expr,pset=pset[idx])
        individual[idx] = tree

    return (individual,)

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    evlpb=cxpb/(cxpb+mutpb)
    if random.random() < evlpb:
        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])

            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    else:
        for i in range(len(offspring)):
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    # 使用 sorted 按适应度升序排序
    sorted_population = sorted(populationCopy, key=lambda ind: ind.fitness.values)
    return sorted_population