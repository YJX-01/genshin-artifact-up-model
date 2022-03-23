import json
import random
from typing import Callable, Sequence, Mapping, Any, Dict, Tuple
import numpy as np
from art import Art
from artpossibility import ArtPossibility


class ArtSimulation(object):
    '''
    you should set <main stat>, <target stat>, <threshold> and <stop criterion>\n
    then start simulation\n
    '''
    __subs = ['ATK_PER', 'ATK_CONST',
              'DEF_PER', 'DEF_CONST',
              'HP_PER', 'HP_CONST',
              'EM', 'ER',
              'CRIT_RATE', 'CRIT_DMG']

    __positions = ['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET']

    __distribution_main = {
        "FLOWER": {"HP_CONST": 1000},
        "PLUME": {"ATK_CONST": 1000},
        "SANDS": {
            "HP_PER": 1334,
            "ATK_PER": 1333,
            "DEF_PER": 1333,
            "ER": 500,
            "EM": 500
        },
        "GOBLET": {
            "HP_PER": 850,
            "ATK_PER": 850,
            "DEF_PER": 800,
            "PYRO_DMG": 200,
            "ELECTRO_DMG": 200,
            "CRYO_DMG": 200,
            "HYDRO_DMG": 200,
            "ANEMO_DMG": 200,
            "GEO_DMG": 200,
            "PHYSICAL_DMG": 200,
            "EM": 100
        },
        "CIRCLET": {
            "HP_PER": 1100,
            "ATK_PER": 1100,
            "DEF_PER": 1100,
            "CRIT_RATE": 500,
            "CRIT_DMG": 500,
            "HEAL_BONUS": 500,
            "EM": 200
        }
    }

    __random_repo = np.random.random(1_000_000)
    __random_pointer = 0

    @classmethod
    def get_r(cls) -> float:
        cls.__random_pointer += 1
        if cls.__random_pointer >= 1_000_000:
            cls.__random_repo = np.random.random(1_000_000)
            cls.__random_pointer = 0
        return cls.__random_repo[cls.__random_pointer]

    def __init__(self) -> None:
        self.artifact_finish: Sequence[Tuple[int, Art]] = []
        self.artifact_abandon: Sequence[Tuple[int, int, Art]] = []

        self.set_type: int = 0
        self.main_stat: Dict[str, str] = {}
        self.target_stat: Sequence[str] = []
        self.stop_criterion: Callable[[Any], bool] = None
        self.output_mode: bool = False

        self.p = ArtPossibility()

        self.result_artifact: Dict[str, Art] = {}
        self.top_for_each_set: Dict[str, Tuple[Art, Art]] = {}

    @property
    def threshold(self):
        return self.p.threshold

    @threshold.setter
    def threshold(self, t):
        self.p.threshold = t

    def clear_result(self):
        self.artifact_finish.clear()
        self.artifact_abandon.clear()
        self.result_artifact.clear()
        self.top_for_each_set.clear()

    @staticmethod
    def evaluate_artifact(a: Art, value_vector: Sequence[float] = [0.6, 0.2, 0, 0, 0, 0, 0, 0.4, 1, 1]) -> float:
        return sum(map(lambda x, y: x*y, value_vector, a.list)) if a else 0

    def comb_update(self, art: Art):
        if not self.result_artifact or not self.top_for_each_set or not art:
            self.comb_build()
            return
        if art.set_type.value == 0 and \
                self.evaluate_artifact(art) > self.evaluate_artifact(self.top_for_each_set[art.pos_type.name][0]):
            self.top_for_each_set[art.pos_type.name][0] = art
        elif art.set_type.value == 1 and \
                self.evaluate_artifact(art) > self.evaluate_artifact(self.top_for_each_set[art.pos_type.name][1]):
            self.top_for_each_set[art.pos_type.name][1] = art

    def comb_build(self):
        self.result_artifact = dict.fromkeys(self.__positions)
        self.top_for_each_set = dict.fromkeys(self.__positions)
        for pos in self.__positions:
            self.top_for_each_set[pos] = [None, None]
        for t, art in self.artifact_finish:
            if art.set_type.value == 0 and \
                    self.evaluate_artifact(art) > self.evaluate_artifact(self.top_for_each_set[art.pos_type.name][0]):
                self.top_for_each_set[art.pos_type.name][0] = art
            elif art.set_type.value == 1 and \
                    self.evaluate_artifact(art) > self.evaluate_artifact(self.top_for_each_set[art.pos_type.name][1]):
                self.top_for_each_set[art.pos_type.name][1] = art

    def get_max_combinations(self) -> Dict[str, Art]:
        '''
        choose the combination that has max target stat\n
        '''
        # the default target set is 0, and there is a uncertain postion for set1
        # to simplify, set1 only occur in 'sands' / 'goblet' / 'circlet'
        # the sort method is simply to count target stat which is naive and need discussion
        def find_max_set1(stat_list: list, top_dict: dict) -> str:
            max_stat = ('', 0)
            for stat in stat_list:
                n1 = self.evaluate_artifact(top_dict[stat][1])
                n0 = self.evaluate_artifact(top_dict[stat][0])
                if n1-n0 > max_stat[1]:
                    max_stat = (stat, n1-n0)
            return max_stat

        stat_list = []
        for s in self.__positions:
            if self.top_for_each_set[s][0]:
                self.result_artifact[s] = self.top_for_each_set[s][0]
            else:
                stat_list.append(s)

        max_stat = find_max_set1(stat_list, self.top_for_each_set) \
            if stat_list else find_max_set1(self.__positions, self.top_for_each_set)

        if max_stat[0]:
            self.result_artifact[max_stat[0]] = self.top_for_each_set[max_stat[0]][1]

        return self.result_artifact

    def get_max_combinations_array(self):
        '''
        choose the combination that has max target stat\n
        \treturn: np.array
        '''
        self.get_max_combinations()
        result_array = np.zeros(10)
        for v in self.result_artifact.values():
            if v:
                result_array += v.array
            else:
                return np.zeros(10)
        return result_array

    def print_max_combinations(self):
        '''print the combination that has max target stat if output mode is on'''
        self.get_max_combinations()
        for v in self.result_artifact.values():
            print(v)

    def initialize(self):
        # check the preparation
        if not self.stop_criterion \
                or not self.main_stat \
                or not self.target_stat \
                or not self.p.threshold:
            print('something not set')
            return
        self.p.main_stat = self.main_stat
        self.p.evaluate = self.evaluate_artifact
        self.p.initialize()
        return

    def sample_generation(self) -> Art:
        # generate position
        pos = int(self.get_r()*5)+1

        # generate sets and main stat
        main_dis: Dict[str, int] = \
            self.__distribution_main[self.__positions[pos-1]]
        main = random.choices(list(main_dis.keys()),
                              list(main_dis.values()))[0]
        sets = int(self.get_r()*2)+self.set_type

        sample = Art()
        sample.generate(main, sets, pos)
        return sample

    def start_simulation(self, max_times: int = 1000):
        # check the preparation
        if not self.stop_criterion \
                or not self.main_stat \
                or not self.target_stat \
                or not self.p.threshold:
            print('something not set')
            return

        if self.output_mode:
            print('\n--- simulation start ---\n')

        # simulate for max_times
        for i in range(max_times):
            # generate an artifact
            sample: Art = self.sample_generation()
            if self.output_mode:
                print('in <{:<4}> simultion: '.format(i), end='')
                print(sample, end='')

            upgrade_flag: bool = True

            # if main stat doesn't match, abandon.
            # if set doesn't match and not in 'sands', 'goblet', 'circlet', abandon
            if sample.main_stat.name != self.main_stat[sample.pos_type.name]:
                upgrade_flag = False
            if sample.set_type.value == 1 and sample.pos_type.value not in [3, 4, 5]:
                upgrade_flag = False

            upgrade_flag = self.p.judge(sample) if upgrade_flag else False

            while(sample.upgrade_time < 5):
                if not upgrade_flag:
                    self.artifact_abandon.append((i+1, sample.upgrade_time))
                    break
                sample.upgrade()
                if sample.upgrade_time == 5:
                    break
                upgrade_flag = self.p.judge(sample)

            if self.output_mode:
                print(f'   upgrade stop at {sample.upgrade_time} times')

            if sample.upgrade_time == 5:
                self.artifact_finish.append((i+1, sample))
                self.comb_update(sample)
                if self.output_mode:
                    print('finish upgrade:', sample)
                    self.print_max_combinations()
                # judge the stopping criterion
                if len(self.artifact_finish) >= 5 and self.stop_criterion(self.get_max_combinations_array()):
                    break
        # print output
        if self.output_mode:
            print('\n', '-'*50)
            print('artifact finish upgrade:', len(self.artifact_finish))
            print('artifact abandon halfway:',
                  sum([1 for a in self.artifact_abandon if 0 < a[1] < 5]))
            print('artifact abandon initially:',
                  sum([1 for a in self.artifact_abandon if 0 == a[1]]))
            print('\n', '-'*50)
            self.print_max_combinations()
            print(self.get_max_combinations_array())
        return
