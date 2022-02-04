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
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    __positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']

    with open('./doc/distribution.json', 'r') as f:
        js = json.load(f)
        __distribution_main: Dict[str, Dict[str, int]] = js['main']

    def __init__(self) -> None:
        self.artifact_finish: Sequence[Tuple[int, Art]] = []
        self.artifact_abandon: Sequence[Tuple[int, int, Art]] = []

        self.main_stat: Dict[str, str] = {}
        self.target_stat: Sequence[str] = []
        self.stop_criterion: Callable[[Any], bool] = None
        self.output_mode: bool = False

        self.p = ArtPossibility()
    
    @property
    def threshold(self):
        return self.p.threshold
    
    @threshold.setter
    def threshold(self, t):
        self.p.threshold = t

    def clear_result(self):
        self.artifact_finish.clear()
        self.artifact_abandon.clear()
        self.p.tolerance = dict.fromkeys(self.__positions, 0)

    @staticmethod
    def evaluate_artifact(a: Art, value_vector: Sequence[float] = [0.2, 0.6, 0, 0, 0, 0, 0.4, 0, 1, 1]) -> float:
        return sum(map(lambda x, y: x*y, value_vector, a.list)) if a else 0

    def get_max_combinations(self) -> Dict[str, Any]:
        '''
        choose the combination that has max target stat\n
        \treturn: np.array
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

        result_artifact = dict.fromkeys(self.__positions)
        top_for_each_set = dict.fromkeys(self.__positions)
        for pos in self.__positions:
            top_for_each_set[pos] = [None, None]
        for t, art in self.artifact_finish:
            if art.sets == 0 and self.evaluate_artifact(art) > self.evaluate_artifact(top_for_each_set[art.pos][0]):
                top_for_each_set[art.pos][0] = art
            elif art.sets == 1 and self.evaluate_artifact(art) > self.evaluate_artifact(top_for_each_set[art.pos][1]):
                top_for_each_set[art.pos][1] = art

        stat_list = []
        for s in self.__positions:
            if top_for_each_set[s][0]:
                result_artifact[s] = top_for_each_set[s][0]
            else:
                stat_list.append(s)

        max_stat = find_max_set1(stat_list, top_for_each_set) \
            if stat_list else find_max_set1(self.__positions, top_for_each_set)

        if max_stat[0]:
            result_artifact[max_stat[0]] = top_for_each_set[max_stat[0]][1]

        return result_artifact

    def get_max_combinations_array(self):
        '''
        choose the combination that has max target stat\n
        \treturn: np.array
        '''
        result_artifact = self.get_max_combinations()
        result_array = np.zeros(10)
        for v in result_artifact.values():
            if v:
                result_array += v.array
            else:
                return np.zeros(10)
        return result_array

    def print_max_combinations(self):
        '''print the combination that has max target stat if output mode is on'''
        if not self.output_mode:
            return
        result_artifact = self.get_max_combinations()
        for v in result_artifact.values():
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
        position = random.choice(self.__positions)

        # generate sets and main stat
        main_dis: Dict[str, int] = self.__distribution_main[position]
        main = random.choices(list(main_dis.keys()),
                              list(main_dis.values()))[0]
        sets = random.randint(0, 1)

        sample = Art()
        sample.generate(main, sets, position)
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
                print(f'in <{i}> simultion get artifact below:', end='')
                print(sample, end='')

            upgrade_flag: bool = True

            # if main stat doesn't match, abandon.
            # if set doesn't match and not in 'sands', 'goblet', 'circlet', abandon
            if sample.main != self.main_stat[sample.pos]:
                upgrade_flag = False
            if sample.sets == 1 and sample.pos not in ['sands', 'goblet', 'circlet']:
                upgrade_flag = False

            upgrade_flag = self.p.judge(sample) if upgrade_flag else False
            
            while(sample.upgrade_time < 5):
                if not upgrade_flag:
                    self.artifact_abandon.append(
                        (i+1, sample.upgrade_time, sample))
                    break
                sample.upgrade()
                if sample.upgrade_time == 5:
                    break
                upgrade_flag = self.p.judge(sample)

            if self.output_mode:
                print(f'   upgrade stop at {sample.upgrade_time} times')

            if sample.upgrade_time == 5:
                self.artifact_finish.append((i+1, sample))
                if self.output_mode:
                    print('finish upgrade:', sample)
                    self.print_max_combinations()
                # update possibility tolerance
                if not self.p.tolerance[sample.pos] and self.evaluate_artifact(sample) > self.p.threshold[sample.pos]:
                    self.p.tolerance[sample.pos] = 1
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
