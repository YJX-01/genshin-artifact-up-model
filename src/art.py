import random
import json
import heapq
from typing import Dict, Sequence
import numpy as np


class Art(object):
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    with open('./doc/distribution.json', 'r') as f:
        js = json.load(f)
        __distribution_sub: Dict[str, int] = js['sub']

    def __init__(self) -> None:
        for s in self.__subs:
            self.__setattr__(s, 0)
        self.pos: str = ''
        self.sets: int = -1  # default target set is 0
        self.main: str = ''
        self.upgrade_time: int = 0

    def set_dict(self, in_dict: dict):
        for k, v in in_dict.items():
            self.__setattr__(k, v)

    @property
    def list(self) -> Sequence[int]:
        return [self.__getattribute__(s) for s in self.__subs]

    @property
    def array(self) -> Sequence:
        return np.array([self.__getattribute__(s) for s in self.__subs])

    @property
    def exist_stat(self) -> Sequence[str]:
        return [s for s in self.__subs if self.__getattribute__(s)]

    def generate(self, main: str, sets: int,  pos: str):
        p_dict = self.__distribution_sub.copy()
        p_dict.pop(main, None)
        init_num = 3+int(random.random() <= 0.2)
        heap = []
        for k, v in p_dict.items():
            ui = random.random()
            ki = ui**(1/v)
            if len(heap) < init_num:
                heapq.heappush(heap, (ki, k))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, k))
                if len(heap) > init_num:
                    heapq.heappop(heap)
        list(map(lambda s: self.__setattr__(s[1], 1), heap))
        self.pos = pos
        self.sets = sets
        self.main = main

    def upgrade(self):
        exist_stat = self.exist_stat
        if len(exist_stat) == 3:
            p_dict = self.__distribution_sub.copy()
            list(map(lambda x: p_dict.pop(x, None), exist_stat+[self.main]))
            upgrade_stat = random.choices(
                list(p_dict.keys()), list(p_dict.values()))[0]
        else:
            upgrade_stat = random.choice(exist_stat)
        self.__dict__[upgrade_stat] += 1
        self.upgrade_time += 1

    @property
    def chinese(self) -> str:
        with open('./doc/chinese_dict.json', 'r', encoding='utf8') as f:
            __chinese_dict: dict = json.load(f)
        str_list = ['{:<5}: {}'.format(__chinese_dict.get(k, k), v) for k, v in self.__dict__.items()
                    if v and k in self.__subs]
        if len(str_list) < 4:
            str_list.append('     :  ')
        for k in ['pos', 'sets', 'main']:
            str_list.append('{:<4}: {:<7}'.format(
                __chinese_dict.get(k, k), __chinese_dict.get(self.__dict__[k], self.__dict__[k])))
        return ' | '.join(str_list)

    def __repr__(self) -> str:
        str_list = ['{:<5}: {}'.format(k, v) for k, v in self.__dict__.items()
                    if v and k in self.__subs]
        if len(str_list) < 4:
            str_list.append('     :  ')
        for k in ['pos', 'sets', 'main']:
            str_list.append('{:<4}: {:<7}'.format(k, self.__dict__[k]))
        return ' | '.join(str_list)

    def __len__(self) -> int:
        return sum([self.__getattribute__(s) for s in self.__subs
                    if self.__getattribute__(s)])
