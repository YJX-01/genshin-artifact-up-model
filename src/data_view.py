import json
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from classifier import *

__subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
          'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']
__positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']


def evaluate_artifact(a: Art) -> float:
    if not a:
        return 0
    v_a = a.to_array()
    v_w = np.array([0.2, 0.6, 0, 0, 0, 0, 0, 0, 1, 1])
    return np.matmul(v_a, v_w)


def get_max_combinations(artifact_finish: List[Tuple], target_stat: List[str]):
    def find_max_set1(stat_list: list, top_dict: dict) -> str:
        max_stat = ('', 0)
        for stat in stat_list:
            n1 = evaluate_artifact(top_dict[stat][1])
            n0 = evaluate_artifact(top_dict[stat][0])
            if n1-n0 > max_stat[1]:
                max_stat = (stat, n1-n0)
        return max_stat
    result_artifact = dict.fromkeys(__positions)
    top_for_each_set = dict.fromkeys(__positions)
    for pos in __positions:
        top_for_each_set[pos] = [None, None]
    for tup in artifact_finish:
        art = tup[1]
        if art.sets == 0 and evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos][0]):
            top_for_each_set[art.pos][0] = art
        elif art.sets == 1 and evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos][1]):
            top_for_each_set[art.pos][1] = art
    stat_list = []
    for s in __positions:
        if top_for_each_set[s][0]:
            result_artifact[s] = top_for_each_set[s][0]
        else:
            stat_list.append(s)
    max_stat = find_max_set1(stat_list, top_for_each_set) \
        if stat_list else find_max_set1(__positions, top_for_each_set)
    if max_stat[0]:
        result_artifact[max_stat[0]] = top_for_each_set[max_stat[0]][1]
    return result_artifact


def view_stack_plot_for_one(sim: ArtClassifier):
    if not sim.artifact_finish:
        return

    x: List[int] = []
    y: Dict[str, List[int]] = {}
    for k in (sim.target_stat+['OTHER']):
        y[k] = []

    for index in range(len(sim.artifact_finish)):
        x.append(sim.artifact_finish[index][0])
        result_artifact = get_max_combinations(
            sim.artifact_finish[:index+1], sim.target_stat)
        stack_data = dict.fromkeys(sim.target_stat+['OTHER'], 0)
        for a in result_artifact.values():
            if not a:
                continue
            for k, v in a.__dict__.items():
                if k in sim.target_stat:
                    stack_data[k] += v
                elif k in __subs:
                    stack_data['OTHER'] += v
        for k, v in stack_data.items():
            y[k].append(v)

    fig, ax = plt.subplots()
    ax.stackplot(x, y.values(), labels=y.keys(), alpha=0.7)
    ax.set_yticks(np.linspace(0, 46, 24))
    ax.yaxis.grid(True)
    ax.set_title('stack plot')
    ax.set_xlabel('times')
    ax.set_ylabel('stat num')
    ax.legend(loc='upper left')
    plt.show()


def view_step_plot_for_one(sim: ArtClassifier):
    if not sim.artifact_finish:
        return

    x: List[int] = []
    y: List[float] = []
    colors: List[float] = []
    area: List[int] = []

    for index in range(len(sim.artifact_finish)):
        result_artifact = get_max_combinations(
            sim.artifact_finish[:index+1], sim.target_stat)
        value = sum([evaluate_artifact(a) for a in result_artifact.values()])
        complete_flag = all(result_artifact.values())
        stat_num = sum([v for k, v in sim.artifact_finish[index][1].__dict__.items()
                        if k in sim.target_stat])

        if complete_flag:
            colors.append('g')
        else:
            colors.append('r')
        x.append(sim.artifact_finish[index][0])
        y.append(value)
        area.append(stat_num*6)

    plt.step(x, y, where='post', label='growth curve', c='k')
    plt.scatter(x, y, c=colors, s=area, alpha=0.8, label='artifact upgraded')
    plt.grid(axis='y')
    plt.title('step plot')
    plt.xlabel('times')
    plt.ylabel('stat value')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    sim = ArtClassifier()
    sim.set_main_stat(dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                               ['HP', 'ATK', 'ER', 'ELECTRO_DMG', 'CR'])))
    sim.set_stat_criterion(['CD', 'CR', 'ATK_P'])
    sim.set_stop_criterion(stopping)
    sim.w_data_input('./data/optimal_w.json')

    sim.set_output(False)

    sim.start_simulation(5000)
    view_step_plot_for_one(sim)
    view_stack_plot_for_one(sim)
