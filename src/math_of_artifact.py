import json
from itertools import product
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
          'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

__positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']

__weight: Dict[str, int] = {}
with open('./doc/distribution.json', 'r') as f:
    __weight = json.load(f)['sub']


def WRS():
    '''Weighted Random Sampling(WRS)'''
    main_stat = ['ATK', 'HP', 'ATK_P', 'HP_P', 'DEF_P',
                 'ER', 'EM', 'CR', 'CD', 'HEAL_BONUS', 'ELEM_DMG']

    # for 3 times
    occur_for_stat3 = dict.fromkeys(main_stat)
    for stat in main_stat:
        weight = __weight.copy()
        weight.pop(stat, None)
        occur_for_stat = dict.fromkeys(weight, 0)
        for key1 in weight:
            weight1 = weight.copy()
            weight1.pop(key1, None)
            p1 = weight[key1]/sum(list(weight.values()))
            occur_for_stat[key1] += p1/3
            for key2 in weight1:
                weight2 = weight1.copy()
                weight2.pop(key2, None)
                p2 = weight1[key2]/sum(list(weight1.values()))
                occur_for_stat[key2] += p1*p2/3
                for key3 in weight2:
                    p3 = weight2[key3]/sum(list(weight2.values()))
                    occur_for_stat[key3] += p1*p2*p3/3
        occur_for_stat3[stat] = occur_for_stat

    # for 4 times
    occur_for_stat4 = dict.fromkeys(main_stat)
    for stat in main_stat:
        weight = __weight.copy()
        weight.pop(stat, None)
        occur_for_stat = dict.fromkeys(weight, 0)
        for key1 in weight:
            weight1 = weight.copy()
            weight1.pop(key1, None)
            p1 = weight[key1]/sum(list(weight.values()))
            occur_for_stat[key1] += p1/4
            for key2 in weight1:
                weight2 = weight1.copy()
                weight2.pop(key2, None)
                p2 = weight1[key2]/sum(list(weight1.values()))
                occur_for_stat[key2] += p1*p2/4
                for key3 in weight2:
                    weight3 = weight2.copy()
                    weight3.pop(key3, None)
                    p3 = weight2[key3]/sum(list(weight2.values()))
                    occur_for_stat[key3] += p1*p2*p3/4
                    for key4 in weight3:
                        p4 = weight3[key4]/sum(list(weight3.values()))
                        occur_for_stat[key4] += p1*p2*p3*p4/4
        occur_for_stat4[stat] = occur_for_stat

    # for init (4stat:3stat=1:4)
    occur_for_stat_init = dict.fromkeys(main_stat)
    for m in occur_for_stat_init:
        tmp_dis = dict.fromkeys(occur_for_stat3[m])
        for k in tmp_dis:
            tmp_dis[k] = occur_for_stat3[m][k]*3*0.8\
                + occur_for_stat4[m][k]*4*0.2
        occur_for_stat_init[m] = tmp_dis

    # for finish (4stat 100%)
    occur_for_stat_finish = dict.fromkeys(main_stat)
    for m in occur_for_stat_finish:
        tmp_dis = dict.fromkeys(occur_for_stat3[m])
        for k in tmp_dis:
            tmp_dis[k] = occur_for_stat4[m][k]*4
        occur_for_stat_finish[m] = tmp_dis

    table_for_WRS(occur_for_stat3,
                  'stat occurrence in stats of initial 3-stat-artifact')
    table_for_WRS(occur_for_stat4,
                  'stat occurrence in stats of initial 4-stat-artifact')
    table_for_WRS(occur_for_stat_init,
                  'stat occurrence in initial artifact')
    table_for_WRS(occur_for_stat_finish,
                  'stat occurrence in upgraded artifact')
    return


def table_for_WRS(occur: Dict[str, Dict[str, float]], title: str):
    '''Draw table for occurrence'''
    main_stat = ['ATK', 'HP', 'ATK_P', 'DEF_P', 'HP_P',
                 'ER', 'EM', 'CR', 'CD', 'HEAL_BONUS', 'ELEM_DMG']
    sub_stat = ['ATK', 'DEF', 'HP', 'ATK_P',  'DEF_P', 'HP_P',
                'ER', 'EM', 'CR', 'CD']
    cell_text: List[List[str]] = [
        ['' for __ in range(len(main_stat))] for _ in range(len(sub_stat))
    ]
    for m, d in occur.items():
        for s, o in d.items():
            cell_text[sub_stat.index(s)][main_stat.index(m)] = \
                '{:.2f}%'.format(o*100)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.set_axis_off()
    ax.table(cellText=cell_text,
             rowLabels=sub_stat,
             colLabels=main_stat,
             cellLoc='center',
             loc='center')
    ax.text(0.45, 0.84, 'main stats', va='center', fontsize=8)
    plt.title(title)
    plt.show()
    return


def show_weight():
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    col1 = __positions
    row1 = ['ATK', 'DEF', 'HP', 'ATK_P',  'DEF_P', 'HP_P',
            'ER', 'EM', 'CR', 'CD',
            "HEAL_BONUS", "PRYO_DMG", "ELECTRO_DMG", "CRYO_DMG", "HYDRO_DMG",
            "ANEMO_DMG", "GEO_DMG", "PHYSICAL_DMG"]
    cell1 = [
        ['' for __ in range(len(col1))] for _ in range(len(row1))
    ]
    with open('./doc/distribution.json', 'r') as f:
        main_dis = json.load(f)['main']
    for p, d in main_dis.items():
        for s, o in d.items():
            cell1[row1.index(s)][col1.index(p)] = str(o)
    ax1.set_axis_off()
    ax1.set_title('weight of main stat')
    ax1.table(cellText=cell1,
              rowLabels=row1,
              colLabels=col1,
              cellLoc='center',
              colWidths=[0.1]*5,
              loc='center',
              fontsize='small')

    cell2 = [[str(__weight[k])] for k in __subs]
    ax2.set_axis_off()
    ax2.set_title('weight of sub stat')
    ax2.table(cellText=cell2,
              rowLabels=__subs,
              colLabels=['weight'],
              cellLoc='center',
              colWidths=[0.6],
              loc='center',
              fontsize='small')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1)
    plt.show()
    return


def MD():
    '''Multinomial Distribution'''
    md_dis: Dict[int, Dict[int, int]] = dict.fromkeys(range(3, 10))
    for stat_num in range(3, 10):
        md_dis[stat_num] = dict.fromkeys(range(7*stat_num, 10*stat_num+1), 0)
        for u in product([7, 8, 9, 10], repeat=stat_num):
            md_dis[stat_num][sum(u)] += 1

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.invert_yaxis()
    ax.set_xlim(20, 61)
    ax.set_yticks([3, 4, 5, 6])
    ax.set_xticks(np.arange(20, 61, 1))
    ax.tick_params(labelsize=5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='tab:gray', alpha=0.8)
    ax.set_title('Multinomial Distribution stat_num = 3, 4, 5, 6')
    ax.set_ylabel('stat num')
    ax.set_xlabel('upgrade value')

    color_map = plt.colormaps['coolwarm']
    for stat_num in range(3, 7):
        x = np.array(list(md_dis[stat_num].keys()))
        possibility = np.array(list(md_dis[stat_num].values()))
        n_sum = possibility.sum()
        possibility = np.true_divide(possibility, n_sum)
        max_pos = possibility.max()
        min_pos = possibility.min()
        cl = np.add(0.15, np.multiply(0.7, np.true_divide(
            possibility-min_pos, max_pos-min_pos)))
        label = []
        for p in possibility:
            if p > 0.001:
                label.append('{:.1%}'.format(p))
            else:
                label.append('{:.0e}'.format(p))
        rects = ax.barh(stat_num, np.ones(len(x)), left=x - 0.5,
                        color=color_map(cl), height=1)
        ax.bar_label(rects, label_type='center',
                     color='k', labels=label, fontsize=2.5)

    fig2, ax2 = plt.subplots(figsize=(9, 4), dpi=300)
    ax2.invert_yaxis()
    ax2.set_xlim(48, 91)
    ax2.set_yticks([7, 8, 9])
    ax2.set_xticks(np.arange(48, 91, 1))
    ax2.tick_params(labelsize=5)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(color='tab:gray', alpha=0.8)
    ax2.set_title('Multinomial Distribution stat_num = 7, 8, 9')
    ax2.set_ylabel('stat num')
    ax2.set_xlabel('upgrade value')

    for stat_num in range(7, 10):
        x = np.array(list(md_dis[stat_num].keys()))
        possibility = np.array(list(md_dis[stat_num].values()))
        n_sum = possibility.sum()
        possibility = np.true_divide(possibility, n_sum)
        max_pos = possibility.max()
        min_pos = possibility.min()
        cl = np.add(0.15, np.multiply(0.7, np.true_divide(
            possibility-min_pos, max_pos-min_pos)))
        label = []
        for p in possibility:
            if p > 0.001:
                label.append('{:.1%}'.format(p))
            else:
                label.append('{:.0e}'.format(p))
        rects = ax2.barh(stat_num, np.ones(len(x)), left=x - 0.5,
                         color=color_map(cl), height=1)
        ax2.bar_label(rects, label_type='center',
                      color='k', labels=label, fontsize=2.5)

    plt.show()
    return


if __name__ == '__main__':
    show_weight()
    WRS()
    MD()
