import json
from typing import Mapping, Sequence
from itertools import combinations, product
import numpy as np
from classifier import Art


class DataGenerator(object):
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    __positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']

    def __init__(self):
        self.data = {'flower': {}, 'plume': {},
                     'sands': {}, 'goblet': {}, 'circlet': {}}
        self.main_stat = {}
        self.target_stat = []
        self.ignore_stat = []
        self.output_path = ''
        self.chinese = False
        self.tmp_data = []

    def set_output_path(self, path):
        self.output_path = path
        try:
            with open(path, 'r') as f:
                self.data = json.load(f)
        except:
            return

    def set_chinese(self, chinese: bool = True):
        self.chinese = True

    def set_main_stat(self, main_stat: Sequence[str]):
        '''
        set artifact's main stat\n
        \tmain_stat: List[str]
        '''
        self.main_stat = main_stat

    def set_target_stat(self, target: Sequence[str]):
        '''
        set target sub stat\n
        \ttarget: List[str]
        '''
        self.target_stat = target
        self.ignore_stat = [s for s in self.__subs
                            if s not in target]

    def output(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f)

    def user_choose_for_one(self, stat_num, value_num, final_stat_dict, random_omission) -> int:
        a = Art()
        a.set_dict(final_stat_dict)
        if self.chinese:
            print('\n', a.chinese_ouput())
        else:
            print('\n', a)
        print(f'\n\tstat num = <{stat_num}>, target stat num = <{value_num}>\n',
              'exit this circulation(e);\n already done this part(a);\n save data and break(s);\n pass(p);\n',
              'Will you choose to upgrade this? yes/no (y/n) ',
              end='->')

        choose_option = ''

        if random_omission > 0:
            random_omission -= 1
            print('<omit>')
        else:
            choose_option = input()

        if 'y' in choose_option:
            tmp_l = a.to_list()+[1]
            self.tmp_data.append(tmp_l)
        elif 'n' in choose_option:
            tmp_l = a.to_list()+[-1]
            self.tmp_data.append(tmp_l)
        elif 'p' in choose_option:
            random_omission = np.random.randint(stat_num, 2*stat_num)
        elif 'a' in choose_option:
            return -3
        elif 's' in choose_option:
            return -2
        elif 'e' in choose_option:
            return -1
        return random_omission

    def user_choose(self, stat_num: int, target_stat_pos: list, ignore_stat_pos: list, position, main_stat_name):
        self.tmp_data = []
        user_option = ''
        for tar_stat_num in range(min(len(target_stat_pos), stat_num, 4), -1, -1):
            for tar_case in combinations(target_stat_pos, tar_stat_num):
                # impose random omission to ease the burden of user
                random_omission = 0
                for ignore_case in combinations(ignore_stat_pos, min(stat_num, 4)-tar_stat_num):
                    # set initial stat
                    final_stat = []
                    final_stat.extend(tar_case)
                    final_stat.extend(ignore_case)
                    final_stat_dict = dict(zip(final_stat,
                                               [1 for i in range(min(stat_num, 4))]))
                    final_stat_dict['pos'] = position
                    final_stat_dict['main'] = main_stat_name

                    upgrade_times = max(0, stat_num-4)

                    if not upgrade_times:
                        value_num = tar_stat_num
                        random_omission = self.user_choose_for_one(
                            stat_num, value_num, final_stat_dict, random_omission)
                        if random_omission == -1:
                            break
                        elif random_omission == -2:
                            user_option = 'save'
                            return self.tmp_data, user_option
                        elif random_omission == -3:
                            user_option = 'already'
                            return self.tmp_data, user_option
                        continue

                    for upgrade_choice in product(final_stat, repeat=upgrade_times):
                        tmp_dict = final_stat_dict.copy()
                        for u in upgrade_choice:
                            tmp_dict[u] += 1

                        # change you assist evaluation function here:
                        def value(a, b, c):
                            return a + sum([1 for u in c if u in b])

                        value_num = value(
                            tar_stat_num, target_stat_pos, upgrade_choice)
                        random_omission = self.user_choose_for_one(
                            stat_num, value_num, tmp_dict, random_omission)
                        if random_omission == -1:
                            break
                        elif random_omission == -2:
                            user_option = 'save'
                            return self.tmp_data, user_option
                        elif random_omission == -3:
                            user_option = 'already'
                            return self.tmp_data, user_option

        return self.tmp_data, user_option

    def start_generate(self):
        if not self.output_path or not self.main_stat or not self.target_stat:
            print('something not set')
            return
        for position in self.__positions:
            main_stat_name = self.main_stat[position]
            target_stat_pos = [i for i in self.target_stat
                               if i != main_stat_name and i in self.__subs]
            ignore_stat_pos = [i for i in self.ignore_stat
                               if i != main_stat_name and i in self.__subs]
            # start to list combinations
            for stat_num in range(3, 9):
                print('-'*50, '\n\n',
                      f'start data generate for position: <{position}> , stat num = <{stat_num}>', '\n\n', '-'*50)
                tmp_data, user_option = self.user_choose(
                    stat_num, target_stat_pos, ignore_stat_pos, position, main_stat_name)
                if user_option == 'save':
                    self.data[position][str(stat_num)] = tmp_data
                    print('\nsave training data and break')
                    break
                elif user_option == 'already':
                    print('\nalready have training data in this part')
                    continue
                else:
                    self.data[position][str(stat_num)] = tmp_data
        self.output()


if __name__ == '__main__':
    data_generator = DataGenerator()
    data_generator.set_output_path('./data/training_data.json')
    data_generator.set_main_stat(dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                                          ['HP', 'ATK', 'ER', 'ELECTRO_DMG', 'CR'])))
    data_generator.set_target_stat(['CD', 'CR', 'ATK_P'])
    data_generator.set_chinese()
    data_generator.start_generate()
