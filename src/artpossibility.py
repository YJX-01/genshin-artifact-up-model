from copy import copy
from collections import Counter
from typing import Callable, Dict, Any
from itertools import accumulate, combinations, product
from art import Art, ArtpositionType


class ArtPossibility(object):
    __subs = ['ATK_PER', 'ATK_CONST',
              'DEF_PER', 'DEF_CONST',
              'HP_PER', 'HP_CONST',
              'EM', 'ER',
              'CRIT_RATE', 'CRIT_DMG']

    __positions = ['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET']

    __distribution_sub: Dict[str, int] = {
        "ATK_PER": 100,
        "ATK_CONST": 150,
        "DEF_PER": 100,
        "DEF_CONST": 150,
        "HP_PER": 100,
        "HP_CONST": 150,
        "EM": 100,
        "ER": 100,
        "CRIT_RATE": 75,
        "CRIT_DMG": 75}

    def __init__(self) -> None:
        '''
        set <main stat>, <threshold>, and <evaluate> function\n
        then the class can function
        '''
        self.main_stat: Dict[str, str] = {}
        self.evaluate: Callable[[Any], float] = None

        self.threshold: Dict[str, float] = {}
        # the value threshold decided by self.evaluate

        self.tolerance_p: Dict[str, Dict[str, Dict[str, float]]]\
            = dict.fromkeys(self.__positions)
        # tolerance_p[pos][len][up] = p

        self.possibilities: Dict[str, Dict[str, Dict[str, Dict[tuple, float]]]]\
            = dict.fromkeys(self.__positions)
        # possibilities[pos][len][up][tuple] = p
        
        self.percent: float = 0.2
        # the last 20% or more will not be upgraded

    def get_p(self, art: Art):
        return self.possibilities[art.pos_type.name][len(art)][art.upgrade_time].get(tuple(art.list))

    def get_t(self, art: Art):
        return self.tolerance_p[art.pos_type.name][len(art)][art.upgrade_time]

    def set_p(self, art: Art, p):
        self.possibilities[art.pos_type.name][len(art)][art.upgrade_time][tuple(art.list)] = p

    def recursive_possibilities(self, art: Art) -> float:
        if (p := self.get_p(art)) != None:
            return p
        elif art.upgrade_time == 5:
            p = int(self.evaluate(art) >= self.threshold[art.pos_type.name])
            self.set_p(art, p)
            return p
        else:
            p = 0
            for s in art.exist_stat:
                a = copy(art)
                a.sub_stat = art.sub_stat.copy()
                a.sub_stat[s] += 1
                a.upgrade_time += 1
                p += self.recursive_possibilities(a) / 4
            self.set_p(art, p)
            return p

    def generate_possibilities(self, position: str):
        # initialize
        self.possibilities[position] = {}
        for length in range(3, 10):
            self.possibilities[position][length] = {}
            uptime = [i for i in range(length-4, length-2) if 0 <= i <= 5]
            for upgrade_time in uptime:
                self.possibilities[position][length][upgrade_time] = {}

        # init 3
        # list all combinations
        exist_stat = combinations([s for s in self.__subs if s != self.main_stat[position]], 3)
        # recursively generate possibilities
        for exist_s in exist_stat:
            # initialize corresponding artifact
            art = Art()
            for s in exist_s:
                art.sub_stat[s] = 1
            art.pos_type = ArtpositionType[position]
            # weight sampling
            weight = self.__distribution_sub.copy()
            weight.pop(self.main_stat[position], None)
            list(map(lambda x: weight.pop(x), exist_s))
            weight_sum = sum(weight.values())
            # calculate p
            p = 0
            for k, v in weight.items():
                a = copy(art)
                a.sub_stat = art.sub_stat.copy()
                a.sub_stat[k] = 1
                a.upgrade_time += 1
                p += (self.recursive_possibilities(a)*v/weight_sum)
            self.set_p(art, p)

        # init 4
        # list all combinations
        exist_stat = combinations([s for s in self.__subs if s != self.main_stat[position]], 4)
        for exist_s in exist_stat:
            # initialize corresponding artifact
            art = Art()
            for s in exist_s:
                art.sub_stat[s] = 1
            art.pos_type = ArtpositionType[position]
            self.recursive_possibilities(art)
            

    def generate_tolerance(self, position):
        # initialize
        self.tolerance_p[position] = {}
        for length in range(3, 9):
            self.tolerance_p[position][length] = {}
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                self.tolerance_p[position][length][upgrade_time] = None

        for length in range(3, 9):
            uptime = [i for i in range(length-4, length-2) if 0 <= i < 5]
            for upgrade_time in uptime:
                order_p = sorted(list(
                    Counter(self.possibilities[position][length][upgrade_time].values()).items()))
                sum_num = sum([i[1] for i in order_p])
                accum_list = [d/sum_num for d in
                              accumulate([i[1] for i in order_p])]
                for index in range(len(accum_list)):
                    if accum_list[index] >= self.percent:
                        if order_p[index][0] == 0:
                            index += 1
                        self.tolerance_p[position][length][upgrade_time] = order_p[index][0]
                        break

    def initialize(self) -> None:
        if not self.threshold or not self.main_stat or not self.evaluate:
            print('something not set')
            return
        for position in self.__positions:
            self.generate_possibilities(position)
            self.generate_tolerance(position)

    def judge(self, art: Art) -> bool:
        return self.get_p(art) >= self.get_t(art)


if __name__ == "__main__":
    def evaluate_artifact(a: Art, value_vector=[0.6, 0.2, 0, 0, 0, 0, 0, 0.4, 1, 1]) -> float:
        return sum(map(lambda x, y: x*y, value_vector, a.list)) if a else 0

    ap = ArtPossibility()
    ap.main_stat = dict(zip(['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET'],
                            ['HP_CONST', 'ATK_CONST', 'ER', 'ELECTRO_DMG', 'CRIT_DMG']))
    ap.threshold = dict(zip(['FLOWER', 'PLUME', 'SANDS', 'GOBLET', 'CIRCLET'],
                            [5, 5, 4, 4, 3]))
    ap.evaluate = evaluate_artifact
    ap.initialize()
