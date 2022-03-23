from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from artsim import ArtSimulation
from art import Art

__subs = ['ATK_PER', 'ATK_CONST',
          'DEF_PER', 'DEF_CONST',
          'HP_PER', 'HP_CONST',
          'EM', 'ER',
          'CRIT_RATE', 'CRIT_DMG']
__positions = ['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET']


def evaluate_artifact(a: Art, value_vector=[0.6, 0.2, 0, 0, 0, 0, 0, 0.4, 1, 1]) -> float:
    return sum(map(lambda x, y: x*y, value_vector, a.list)) if a else 0


def get_max_combinations(artifact_finish: List[Tuple]) -> Dict[str, Art]:
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
        art: Art = tup[1]
        if art.set_type.value == 0 and \
                evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos_type.name][0]):
            top_for_each_set[art.pos_type.name][0] = art
        elif art.set_type.value == 1 and \
                evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos_type.name][1]):
            top_for_each_set[art.pos_type.name][1] = art
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


def get_data_from_recorder(recorder, data_name: str):
    '''
    data_name choice:\n
    \tfinish: the artifact that finish upgrade\n
    \thalf: the artifact that is abandoned halfway\n
    \tinit: the artifact that is initially abandoned\n
    \tabandon: the artifact that is abandoned\n
    \tvalue: the value of max-value artifact combination using evaluate_artifact()\n
    \tcomplete: the time that complete rolling a 4-piece set\n
    \tlast: the time that upgrade the last artifact\n
    '''
    if data_name == 'finish':
        return [len(d[0]) for d in recorder]
    elif data_name == 'half':
        return [sum(1 for h in d[1] if 0 < h[1] < 5) for d in recorder]
    elif data_name == 'init':
        return [sum(1 for h in d[1] if 0 == h[1]) for d in recorder]
    elif data_name == 'abandon':
        return [len(d[1]) for d in recorder]
    elif data_name == 'value':
        max_value = []
        for d in recorder:
            max_comb = get_max_combinations(d[0])
            max_v = sum([evaluate_artifact(a) for a in max_comb.values()])
            max_value.append(max_v)
        return max_value
    elif data_name == 'complete':
        complete_set = []
        for d in recorder:
            for index in range(len(d[0])):
                result_artifact = get_max_combinations(d[0][:index+1])
                if all(result_artifact.values()):
                    complete_set.append(d[0][index][0])
                    break
                if index == len(d[0])-1:
                    complete_set.append(d[0][-1][0])
                    break
        return complete_set
    elif data_name == 'last':
        return [d[0][-1][0] for d in recorder]
    else:
        return


def view_stack_plot_for_one(sim: ArtSimulation):
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
            for k, v in a.sub_stat.items():
                if k in sim.target_stat:
                    stack_data[k] += v
                else:
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


def view_step_plot_for_one(sim: ArtSimulation):
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
        stat_num = sum([v for k, v in sim.artifact_finish[index][1].sub_stat.items()
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


def view_value_growth(recorder: List[Tuple[List, List]], length: int = 1000):
    '''
    length should be no greater than simulation max times
    '''
    x: List[int] = np.arange(1, length+1, 1)
    y: List[float] = np.zeros(length)

    for d in recorder:
        tmp_y = [0]
        for index in range(len(d[0])):
            get_time = d[0][index][0]
            for j in range(get_time-len(tmp_y)):
                tmp_y.append(tmp_y[-1])
            result_artifact = get_max_combinations(d[0][:index+1])
            value = sum([evaluate_artifact(a)
                        for a in result_artifact.values()])
            tmp_y[-1] = value
        for _ in range(length-len(tmp_y)):
            tmp_y.append(tmp_y[-1])
        for i in range(length):
            y[i] += tmp_y[i]

    y /= len(recorder)

    plt.plot(x, y)
    plt.grid(axis='both')
    plt.title('value growth curve')
    plt.xlabel('times')
    plt.ylabel('stat value')
    plt.show()


def view_stack_plot(recorder: List[Tuple[List, List]], length: int, target_stat: List[str]):
    x: List[int] = np.arange(1, length+1, 1)
    y: Dict[str, List[int]] = {}
    for k in (target_stat+['OTHER']):
        y[k] = np.zeros(length)

    for d in recorder:
        tmp_y = [dict.fromkeys(target_stat+['OTHER'], 0)]
        for index in range(len(d[0])):
            get_time = d[0][index][0]
            for j in range(get_time-len(tmp_y)):
                tmp_y.append(tmp_y[-1])
            result_artifact = get_max_combinations(d[0][:index+1])
            stack_data = dict.fromkeys(target_stat+['OTHER'], 0)
            for a in result_artifact.values():
                if not a:
                    continue
                for k, v in a.sub_stat.items():
                    if k in target_stat:
                        stack_data[k] += v
                    else:
                        stack_data['OTHER'] += v
            tmp_y[-1] = stack_data
        for _ in range(length-len(tmp_y)):
            tmp_y.append(tmp_y[-1])
        for i in range(length):
            for k in target_stat+['OTHER']:
                y[k][i] += tmp_y[i][k]

    for v in y.values():
        v /= len(recorder)

    fig, ax = plt.subplots()
    ax.stackplot(x, y.values(), labels=y.keys(), alpha=0.7)
    ax.set_yticks(np.linspace(0, 46, 24))
    ax.set_xlim(0, length)
    ax.yaxis.grid(True)
    ax.set_title('stack plot')
    ax.set_xlabel('times')
    ax.set_ylabel('stat num')
    ax.legend(loc='upper left')
    plt.show()


def view_scatter_plot(recorder: List[Tuple[List, List]], x: str = 'finish', y: str = 'abandon'):
    '''
    plot the relationship between x and y using scatter plot\n
    \t\trecoder: List[Tuple[List, List]] = a list that contain [(artifact_finish, artifact_abandon)]\n
    \t\tx = ['finish'|'half'|'init'|'abandon'|'value'|'complete'|'last']\n
    \t\ty = ['finish'|'half'|'init'|'abandon'|'value'|'complete'|'last']\n
    \tfinish: the artifact that finish upgrade\n
    \thalf: the artifact that is abandoned halfway\n
    \tinit: the artifact that is initially abandoned\n
    \tabandon: the artifact that is abandoned\n
    \tvalue: the value of max-value artifact combination using evaluate_artifact()\n
    \tcomplete: the time that complete rolling a 4-piece set\n
    \tlast: the time that upgrade the last artifact\n
    '''
    x_data = get_data_from_recorder(recorder, x)
    y_data = get_data_from_recorder(recorder, y)
    max_value = get_data_from_recorder(recorder, 'value')

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
    \t\trecoder: List[Tuple[List, List]] = a list that contain [(artifact_finish, artifact_abandon)]\n
    \t\tx = ['finish'|'half'|'init'|'abandon'|'value'|'complete'|'last']\n
    \tfinish: the artifact that finish upgrade\n
    \thalf: the artifact that is abandoned halfway\n
    \tinit: the artifact that is initially abandoned\n
    \tabandon: the artifact that is abandoned\n
    \tvalue: the value of max-value artifact combination using evaluate_artifact()\n
    \tcomplete: the time that complete rolling a 4-piece set\n
    \tlast: the time that upgrade the last artifact\n
    '''
    if x == 'value':
        max_value = get_data_from_recorder(recorder, x)
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
        if n.max() < 25:
            ax.set_yticks(np.arange(0, n.max()+1, 1))
    else:
        y = get_data_from_recorder(recorder, x)
        fig, ax = plt.subplots()
        ax.hist(y, 20, facecolor='tab:blue')
    ax.set_xlabel(x)
    ax.set_ylabel('number')
    ax.set_title('histogram for {}'.format(x))
    ax.xaxis.grid(color='dimgrey', alpha=0.9, which='both')
    ax.yaxis.grid(color='dimgrey', alpha=0.5)
    plt.show()


def view_box_plot(recorder: List[Tuple[List, List]]):
    '''draw boxplot from the data of recorder'''
    finish_vector = [len(d[0]) for d in recorder]
    ab_vector = [len(d[1]) for d in recorder]
    half_ab_vector = [sum(1 for h in d[1] if 0 < h[1] < 5) for d in recorder]
    max_value = []
    complete_set = []
    last_vector = []
    for d in recorder:
        max_comb = get_max_combinations(d[0])
        max_v = sum([evaluate_artifact(a) for a in max_comb.values()])
        max_value.append(max_v)
        for index in range(len(d[0])):
            result_artifact = get_max_combinations(d[0][:index+1])
            if all(result_artifact.values()):
                complete_set.append(d[0][index][0])
                break
            if index == len(d[0])-1:
                complete_set.append(d[0][-1][0])
                break
        last_vector.append(d[0][-1][0])

    fig, ((ax0, ax1, ax2),
          (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    bp0 = ax0.boxplot(finish_vector,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['finish'])
    bp1 = ax1.boxplot(ab_vector,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['abandon'])
    bp2 = ax2.boxplot(half_ab_vector,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['half abandon'])
    bp4 = ax4.boxplot(max_value,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['value'])
    bp5 = ax5.boxplot(complete_set,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['complete time'])
    bp6 = ax6.boxplot(last_vector,
                      sym='',
                      vert=True,
                      patch_artist=True,
                      labels=['end time'])
    colors = ['lightblue', 'aquamarine', 'steelblue',
              'pink', 'lightgreen', 'gold']
    for ax in (ax0, ax1, ax2, ax4, ax5, ax6):
        ax.yaxis.grid(True)
    for bp, c in zip((bp0, bp1, bp2, bp4, bp5, bp6), colors):
        for patch in bp['boxes']:
            patch.set_facecolor(c)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def stop(array):
    value_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    flag = np.matmul(array, value_vector) >= 24
    value_vector2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    flag2 = np.matmul(array, value_vector2) >= 18
    return flag and flag2


def no_stop(*args):
    return False


if __name__ == '__main__':
    sim = ArtSimulation()
    sim.main_stat = dict(zip(['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET'],
                             ['HP_CONST', 'ATK_CONST', 'ER', 'ELECTRO_DMG', 'CRIT_DMG']))
    sim.threshold = dict(zip(['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET'],
                             [5, 5, 3.5, 3.5, 3]))
    sim.target_stat = ['CRIT_DMG', 'CRIT_RATE', 'ATK_PER']
    sim.p.percent = 0.3
    # sim.stop_criterion = stop
    sim.stop_criterion = no_stop
    sim.output_mode = True
    sim.initialize()

    sim.start_simulation(2000)
    view_step_plot_for_one(sim)
    view_stack_plot_for_one(sim)
    sim.output_mode = False

    recorder = []
    sim.clear_result()
    for i in range(1000):
        sim.start_simulation(2000)
        recorder.append((sim.artifact_finish.copy(),
                        sim.artifact_abandon.copy()))
        sim.clear_result()

    view_box_plot(recorder)
    view_hist_plot(recorder)
    view_scatter_plot(recorder)
    view_value_growth(recorder, 3000)
    view_stack_plot(recorder, 3000, sim.target_stat)
