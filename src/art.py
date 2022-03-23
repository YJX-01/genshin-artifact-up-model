import random
import json
import heapq
from collections import Counter
from enum import Enum
from typing import Dict, Sequence
import numpy as np


class Art(object):
    __subs = ['ATK_PER', 'ATK_CONST',
              'DEF_PER', 'DEF_CONST',
              'HP_PER', 'HP_CONST',
              'EM', 'ER',
              'CRIT_RATE', 'CRIT_DMG']

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
    
    with open('./doc/chinese_dict.json', 'r', encoding='utf8') as f:
        __chinese_dict: dict = json.load(f)

    __random_repo = np.random.random(1_000_000)
    __random_pointer = 0

    @classmethod
    def get_r(cls) -> float:
        cls.__random_pointer += 1
        if cls.__random_pointer >= 1_000_000:
            cls.__random_repo = np.random.random(1_000_000)
            cls.__random_pointer = 0
        return cls.__random_repo[cls.__random_pointer]

    def __init__(self):
        self.set_type: SetType = SetType(0)
        self.pos_type: ArtpositionType = ArtpositionType(1)
        self.main_stat: StatType = StatType(1)
        self.sub_stat: Counter[str, int] = Counter(dict.fromkeys(self.__subs, 0))
        self.upgrade_time: int = 0

    @property
    def list(self) -> Sequence[int]:
        return [self.sub_stat[s] for s in self.__subs]

    @property
    def array(self) -> Sequence:
        return np.array(self.list)

    @property
    def exist_stat(self) -> Sequence[str]:
        return [s for s in self.__subs if self.sub_stat[s]]

    def generate(self, main: str, sets: int, pos: int):
        p_dict = self.__distribution_sub.copy()
        p_dict.pop(main, None)
        init_num = 3+int(self.get_r() <= 0.2)
        heap = []
        for k, v in p_dict.items():
            ui = self.get_r()
            ki = ui**(1/v)
            if len(heap) < init_num:
                heapq.heappush(heap, (ki, k))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, k))
                if len(heap) > init_num:
                    heapq.heappop(heap)
        for s in heap:
            self.sub_stat[s[1]] = 1
        self.pos_type = ArtpositionType(pos)
        self.set_type = SetType(sets)
        self.main_stat = StatType[main]

    def upgrade(self):
        exist_stat = self.exist_stat
        if len(exist_stat) == 3:
            p_dict = self.__distribution_sub.copy()
            list(map(lambda x: p_dict.pop(x, None), exist_stat+[self.main_stat.name]))
            upgrade_stat = random.choices(list(p_dict.keys()), list(p_dict.values()))[0]
        else:
            upgrade_stat = exist_stat[int(self.get_r()*4)]
        self.sub_stat[upgrade_stat] += 1
        self.upgrade_time += 1

    @property
    def chinese(self) -> str:
        str_list = ['{:<9}: {}'.format(self.__chinese_dict.get(k, k), v) if v else '         :  '
                    for k, v in self.sub_stat.most_common(4)]
        s = self.__chinese_dict.get(self.pos_type.name, self.pos_type.name)
        str_list.append('{:<4}: {:<7}'.format('位置', s))
        s = self.__chinese_dict.get(self.main_stat.name, self.main_stat.name)
        str_list.append('{:<4}: {:<11}'.format('主词条', s))
        s = self.__chinese_dict.get(self.set_type.name, self.set_type.name)
        str_list.append('{:<4}: {:<7}'.format('套装', s))
        return ' | '.join(str_list)

    def __repr__(self) -> str:
        str_list = ['{:<9}: {}'.format(k, v) if v else '         :  ' 
                    for k, v in self.sub_stat.most_common(4)]
        str_list.append('{:<4}: {:<7}'.format('pos', self.pos_type.name))
        str_list.append('{:<4}: {:<11}'.format('main', self.main_stat.name))
        str_list.append('{:<4}: {:<7}'.format('sets', self.set_type.value))
        return ' | '.join(str_list)

    def __len__(self) -> int:
        return sum(self.sub_stat.values())


class SetType(Enum):
    '''圣遗物套装类型'''
    NONE = 0
    GLADIATORS_FINALE = 1  # 角斗士的终幕礼
    JUE_DOU_SHI = 1
    WANDERERS_TROUPE = 2  # 流浪大地的乐团
    YUE_TUAN = 2
    BLOODSTAINED_CHIVALRY = 3  # 染血的骑士道
    RAN_XUE = 3
    NOBLESSE_OBLIGE = 4  # 昔日宗室之仪
    ZONG_SHI = 4
    VIRIDESCENT_VENERER = 5  # 翠绿之影
    FENG_TAO = 5
    MAIDEN_BELOVED = 6  # 被怜爱的少女
    SHAO_NV = 6
    THUNDERSOOTHER = 7  # 平息鸣雷的尊者
    PING_LEI = 7
    THUNDERING_FURY = 8  # 如雷的盛怒
    RU_LEI = 8
    LAVAWALKER = 9  # 渡过烈火的贤人
    DU_HUO = 9
    CRIMSON_WITCH_OF_FLAMES = 10  # 炽烈的炎之魔女
    MO_NV = 10
    ARCHAIC_PETRA = 11  # 悠古的磐岩
    YAN_TAO = 11
    RETRACING_BOLIDE = 12  # 逆飞的流星
    NI_FEI = 12
    BLIZZARD_STRAYER = 13  # 冰风迷途的勇士
    BING_TAO = 13
    HEART_OF_DEPTH = 14  # 沉沦之心
    SHUI_TAO = 14
    TENACITY_OF_THE_MILLELITH = 15  # 千岩牢固
    QIAN_YAN = 15
    PALE_FLAME = 16  # 苍白之火
    CANG_BAI = 16
    SHIMENAWAS_REMINISCENCE = 17  # 追忆之注连
    ZHUI_YI = 17
    EMBLEM_OF_SEVERED_FATE = 18  # 绝缘之旗印
    JUE_YUAN = 18
    OCEANHUED_CLAM = 19  # 海染砗磲
    HAI_RAN = 19
    HUSK_OF_OPULENT_DREAMS = 20  # 华馆梦醒形骸记
    HUA_GUAN = 20


class ArtpositionType(Enum):
    '''圣遗物位置类型'''
    FLOWER = 1  # 生之花
    PLUME = 2  # 死之羽
    SANDS = 3  # 时之沙
    GOBLET = 4  # 空之杯
    CIRCLET = 5  # 理之冠


class StatType(Enum):
    ATK_PER = 1
    ATK_CONST = 2
    DEF_PER = 3
    DEF_CONST = 4
    HP_PER = 5
    HP_CONST = 6
    EM = 7
    ER = 8
    CRIT_RATE = 9
    CRIT_DMG = 10
    HEAL_BONUS = 11
    ANEMO_DMG = 12
    GEO_DMG = 13
    ELECTRO_DMG = 14
    HYDRO_DMG = 15
    PYRO_DMG = 16
    CRYO_DMG = 17
    DENDRO_DMG = 18
    PHYSICAL_DMG = 19
