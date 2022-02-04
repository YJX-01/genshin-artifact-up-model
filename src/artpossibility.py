import json
from copy import copy
from collections import Counter
from typing import Callable, Dict, Any
from itertools import accumulate, combinations, product
from art import Art


class ArtPossibility(object):
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    __positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']

    with open('./doc/distribution.json', 'r') as f:
        js = json.load(f)
        __distribution_sub: Dict[str, int] = js['sub']

    def __init__(self) -> None:
        '''
        set <main stat>, <threshold>, and <evaluate> function\n
        then the class can function
        '''
        self.main_stat: Dict[str, str] = {}
        self.evaluate: Callable[[Any], float] = None

        self.threshold: Dict[str, float] = {}
        # the value threshold decided by self.evaluate

        self.tolerance: Dict[str, int] = dict.fromkeys(self.__positions, 0)
        # the tolerance level [0,1,2,3]

        self.tolerance_p: Dict[str, Dict[str, Dict[str, Dict[int, float]]]]\
            = dict.fromkeys(self.__positions)
        # tolerance_p[pos][len][up][tolerance_lv] = p

        self.possibilities: Dict[str, Dict[str, Dict[str, Dict[tuple, float]]]]\
            = dict.fromkeys(self.__positions)
        # possibilities[pos][len][up][tuple] = p

    def get_p(self, art: Art):
        return self.possibilities[art.pos][len(art)][art.upgrade_time].get(tuple(art.list))

    def get_t(self, art: Art):
        return self.tolerance_p[art.pos][len(art)][art.upgrade_time].get(self.tolerance[art.pos])

    def recursive_possibilities(self, art: Art) -> float:
        if art.upgrade_time == 5:
            return self.evaluate(art) > self.threshold[art.pos]
        elif (p := self.get_p(art)) != None:
            return p
        else:
            p = 0
            for s in art.exist_stat:
                a = copy(art)
                a.__dict__[s] += 1
                a.upgrade_time += 1
                p += self.recursive_possibilities(a) / 4
            self.possibilities[art.pos][len(
                art)][art.upgrade_time][tuple(art.list)] = p
            return p

    def generate_possibilities(self, position: str):
        # initialize
        self.possibilities[position] = {}
        for length in range(3, 9):
            self.possibilities[position][length] = {}
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                self.possibilities[position][length][upgrade_time] = {}

        for length in range(3, 9):
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                if length == 3:
                    # list all combinations
                    exist_stat = combinations(
                        [s for s in self.__subs if s != self.main_stat[position]], 3)
                    # recursively generate possibilities
                    for exist_s in exist_stat:
                        # initialize corresponding artifact
                        art = Art()
                        for s in exist_s:
                            art.__setattr__(s, 1)
                        art.pos = position
                        # weight sampling
                        weight = self.__distribution_sub.copy()
                        weight.pop(self.main_stat[position], None)
                        list(map(lambda x: weight.pop(x), exist_s))
                        weight_sum = sum(weight.values())
                        # calculate p
                        p = 0
                        for k, v in weight.items():
                            a = copy(art)
                            a.__setattr__(k, 1)
                            a.upgrade_time += 1
                            p += (self.recursive_possibilities(a)*v/weight_sum)
                        self.possibilities[position][3][0][tuple(art.list)] = p
                else:
                    # list all combinations
                    exist_stat = combinations(
                        [s for s in self.__subs if s != self.main_stat[position]], 4)
                    for exist_s in exist_stat:
                        # initialize corresponding init artifact
                        art_init = Art()
                        for s in exist_s:
                            art_init.__setattr__(s, 1)
                        art_init.pos = position
                        art_init.upgrade_time = upgrade_time
                        # list all upgrade combinations
                        up_choice = product(exist_s, repeat=length-4)
                        for tup_up in up_choice:
                            art = copy(art_init)
                            for k, v in Counter(tup_up).items():
                                art.__dict__[k] += v
                            if self.get_p(art) != None:
                                continue
                            # calculate p
                            p = 0
                            for s in art.exist_stat:
                                a = copy(art)
                                a.__dict__[s] += 1
                                a.upgrade_time += 1
                                p += (self.recursive_possibilities(a)/4)
                                self.possibilities[position][length][upgrade_time][tuple(
                                    art.list)] = p

    def generate_tolerance(self, position):
        # initialize
        self.tolerance_p[position] = {}
        for length in range(3, 9):
            self.tolerance_p[position][length] = {}
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                self.tolerance_p[position][length][upgrade_time] = {}

        for length in range(3, 9):
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                order_p = sorted(list(
                    Counter(self.possibilities[position][length][upgrade_time].values()).items()))
                sum_num = sum([i[1] for i in order_p])
                accum_list = [d/sum_num for d in
                              accumulate([i[1] for i in order_p])]
                pos_index = [0, 0, 0, 0]
                for index in range(len(accum_list)):
                    if accum_list[index] >= 0.8 and not pos_index[3]:
                        pos_index[3] = order_p[index][0]
                    elif accum_list[index] >= 0.7 and not pos_index[2]:
                        pos_index[2] = order_p[index][0]
                    elif accum_list[index] >= 0.6 and not pos_index[1]:
                        pos_index[1] = order_p[index][0]
                    elif accum_list[index] >= 0.5 and not pos_index[0]:
                        pos_index[0] = order_p[index][0]
                for tol_lv in range(len(pos_index)):
                    self.tolerance_p[position][length][upgrade_time][tol_lv] = pos_index[tol_lv]

    def initialize(self) -> None:
        if not self.threshold or not self.main_stat or not self.evaluate:
            print('something not set')
            return
        for position in self.__positions:
            self.generate_possibilities(position)
            self.generate_tolerance(position)

    def judge(self, art: Art) -> bool:
        if art.sets == 0 and self.get_p(art) >= self.get_t(art):
            return True
        elif art.sets == 1:
            self.tolerance[art.pos] += 1
            if self.get_p(art) >= self.get_t(art):
                self.tolerance[art.pos] -= 1
                return True
            else:
                self.tolerance[art.pos] -= 1
                return False
        else:
            return False


if __name__ == "__main__":
    def evaluate_artifact(a: Art, value_vector=[0.2, 0.6, 0, 0, 0, 0, 0.4, 0, 1, 1]) -> float:
        return sum(map(lambda x, y: x*y, value_vector, a.list)) if a else 0

    ap = ArtPossibility()
    ap.main_stat = dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                            ['HP', 'ATK', 'ER', 'ELECTRO_DMG', 'CR']))
    ap.threshold = dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                            [5, 5, 4, 4, 3]))
    ap.evaluate = evaluate_artifact
    ap.initialize()
