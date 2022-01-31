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


def get_max_combinations(artifact_finish: List[Tuple]):
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
        result_artifact = get_max_combinations(sim.artifact_finish[:index+1])
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
        result_artifact = get_max_combinations(sim.artifact_finish[:index+1])
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


def view_stack_plot():
    return


def view_step_plot():
    return


def view_scatter_plot(recorder: List[Tuple[List, List]], x: str = 'finish', y: str = 'abandon'):
    '''
    plot the relationship between x and y using scatter plot\n
    \trecoder: List[Tuple[List, List]] = a list that contain [(artifact_finish, artifact_abandon)]\n
    \tx = ['finish'|'half'|'init'|'abandon']\n
    \ty = ['finish'|'half'|'init'|'abandon']\n
    finish: the artifact that finish upgrade\n
    half: the artifact that is abandoned halfway\n
    init: the artifact that is initially abandoned\n
    abandon: the artifact that is abandoned
    '''
    if x == 'finish':
        x_data = [len(d[0]) for d in recorder]
    elif x == 'half':
        x_data = [sum(1 for h in d[1] if 0 < h[1] < 5) for d in recorder]
    elif x == 'init':
        x_data = [sum(1 for h in d[1] if 0 == h[1]) for d in recorder]
    elif x == 'abandon':
        x_data = [len(d[1]) for d in recorder]
    else:
        return

    if y == 'finish':
        y_data = [len(d[0]) for d in recorder]
    elif y == 'half':
        y_data = [sum(1 for h in d[1] if 0 < h[1] < 5) for d in recorder]
    elif y == 'init':
        y_data = [sum(1 for h in d[1] if 0 == h[1]) for d in recorder]
    elif y == 'abandon':
        y_data = [len(d[1]) for d in recorder]
    else:
        return

    max_value = []
    for d in recorder:
        max_comb = get_max_combinations(d[0])
        max_v = sum([evaluate_artifact(a) for a in max_comb.values()])
        max_value.append(max_v)
    area = (np.array(max_value) - 20)**2
    colors = np.random.rand(len(recorder))

    plt.scatter(x_data, y_data, s=area, c=colors, alpha=0.8)
    plt.grid(True)
    plt.title('scatter plot')
    plt.xlabel(x+' number')
    plt.ylabel(y+' number')
    plt.show()


def view_hist_plot(recorder: List[Tuple[List, List]], x: str = 'value'):
    '''
    plot the histogram of the data, x-axis is x\n
    \trecoder: List[Tuple[List, List]] = a list that contain [(artifact_finish, artifact_abandon)]\n
    \tx = ['finish'|'half'|'init'|'abandon']\n
    finish: the artifact that finish upgrade\n
    half: the artifact that is abandoned halfway\n
    init: the artifact that is initially abandoned\n
    abandon: the artifact that is abandoned\n
    value: the value of max-value artifact combination using evaluate_artifact()
    '''
    max_value = []
    for d in recorder:
        max_comb = get_max_combinations(d[0])
        max_v = sum([evaluate_artifact(a) for a in max_comb.values()])
        max_value.append(max_v)

    max_int = int(max(max_value))+1
    min_int = int(min(max_value))
    if len(max_value) >= 10000:
        bins = np.arange(min_int, max_int+1, 0.25)
    elif len(max_value) >= 1000:
        bins = np.arange(min_int, max_int+1, 0.5)
    else:
        bins = np.arange(min_int, max_int+1, 1)

    fig, ax = plt.subplots()
    n, b, pathes = ax.hist(max_value, bins, facecolor='tab:blue')
    ax.set_xlim(min_int, max_int)
    ax.set_xticks(np.arange(min_int, max_int+1, 1))
    ax.set_xticks(bins, minor=True)
    ax.set_xlabel(x)
    if n.max() < 25:
        ax.set_yticks(np.arange(0, n.max()+1, 1))
    ax.set_ylabel('number')
    ax.set_title('histogram for {}'.format(x))
    ax.xaxis.grid(color='dimgrey', alpha=0.9, which='both')
    ax.yaxis.grid(color='dimgrey', alpha=0.5)
    plt.show()


def view_scatter_hist():
    return


def view_box_plot(recorder: List[Tuple[List, List]]):
    '''draw boxplot from the data of recorder'''
    finish_vector = [len(d[0]) for d in recorder]
    ab_vector = [len(d[1]) for d in recorder]
    half_ab_vector = [sum(1 for h in d[1] if 0 < h[1] < 5) for d in recorder]
    # init_ab_vector = [sum(1 for h in d[1] if 0 == h[1]) for d in recorder]
    max_value = []
    for d in recorder:
        max_comb = get_max_combinations(d[0])
        max_v = sum([evaluate_artifact(a) for a in max_comb.values()])
        max_value.append(max_v)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(9, 4))
    bp0 = ax0.boxplot(finish_vector,
                      vert=True,
                      patch_artist=True,
                      labels=['finish'])
    bp1 = ax1.boxplot(ab_vector,
                      vert=True,
                      patch_artist=True,
                      labels=['abandon'])
    bp2 = ax2.boxplot(half_ab_vector,
                      vert=True,
                      patch_artist=True,
                      labels=['half abandon'])
    bp3 = ax3.boxplot(max_value,
                      vert=True,
                      patch_artist=True,
                      labels=['value'])
    colors = ['pink', 'aquamarine', 'lightblue', 'lightgreen']
    for bp, c in zip((bp0, bp1, bp2, bp3), colors):
        for patch in bp['boxes']:
            patch.set_facecolor(c)
    plt.show()


def no_stop(*args):
    return False


if __name__ == '__main__':
    sim = ArtClassifier()
    sim.set_main_stat(dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                               ['HP', 'ATK', 'ER', 'ELECTRO_DMG', 'CR'])))
    sim.set_stat_criterion(['CD', 'CR', 'ATK_P'])
    sim.set_stop_criterion(stopping)
    # sim.set_stop_criterion(no_stop)
    sim.w_data_input('./data/optimal_w.json')

    sim.set_output(False)

    sim.start_simulation(5000)
    view_step_plot_for_one(sim)
    view_stack_plot_for_one(sim)

    recorder = []
    sim.clear_result()
    for i in range(1000):
        sim.start_simulation(3000)
        recorder.append((sim.artifact_finish.copy(),
                        sim.artifact_abandon.copy()))
        sim.clear_result()

    view_box_plot(recorder)
    view_hist_plot(recorder)
    view_scatter_plot(recorder)
