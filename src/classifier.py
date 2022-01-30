import json
import time
import random
from typing import Callable, Sequence, Mapping, Any, Dict, Tuple
import numpy as np


class Art(object):
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    def __init__(self) -> None:
        for s in self.__subs:
            self.__setattr__(s, 0)
        self.pos = ''
        # default target set is 0
        self.sets = -1
        self.main = ''

    def set_dict(self, in_dict: dict):
        for k, v in in_dict.items():
            self.__setattr__(k, v)

    def to_list(self) -> list:
        return [self.__getattribute__(s) for s in self.__subs]

    def to_array(self):
        return np.array([self.__getattribute__(s) for s in self.__subs])

    def generate(self, main: str, sets: int,  pos: str, p_dict: dict):
        stat_key = list(p_dict.keys())
        weight = list(p_dict.values())
        init_num = 3+int(random.random() <= 0.2)
        for init_time in range(init_num):
            stat = random.choices(stat_key, weight)[0]
            weight.pop(stat_key.index(stat))
            stat_key.remove(stat)
            self.__setattr__(stat, 1)
        self.pos = pos
        self.sets = sets
        self.main = main

    def upgrade(self):
        exist_stat = [k for k, v in self.__dict__.items()
                      if v and k != 'main' and k != 'pos' and k != 'sets']
        if len(exist_stat) == 3:
            remain_stat = [s for s in self.__subs
                           if s not in exist_stat and s != self.main]
            upgrade_stat = random.choice(remain_stat)
        else:
            upgrade_stat = random.choice(exist_stat)
        self.__dict__[upgrade_stat] += 1

    def chinese_ouput(self) -> str:
        with open('./doc/chinese_dict.json', 'r', encoding='utf8') as f:
            __chinese_dict: dict = json.load(f)
        str_list = ['{:<5}: {}'.format(__chinese_dict.get(k, k), v) for k, v in self.__dict__.items()
                    if v and k != 'main' and k != 'pos' and k != 'sets']
        if len(str_list) < 4:
            str_list.append('     :  ')
        for k in ['pos', 'sets', 'main']:
            str_list.append('{:<4}: {:<7}'.format(
                __chinese_dict.get(k, k), __chinese_dict.get(self.__dict__[k], self.__dict__[k])))
        return ' | '.join(str_list)

    def __repr__(self) -> str:
        str_list = ['{:<5}: {}'.format(k, v) for k, v in self.__dict__.items()
                    if v and k != 'main' and k != 'pos' and k != 'sets']
        if len(str_list) < 4:
            str_list.append('     :  ')
        for k in ['pos', 'sets', 'main']:
            str_list.append('{:<4}: {:<7}'.format(k, self.__dict__[k]))
        return ' | '.join(str_list)

    def __len__(self) -> int:
        return sum([v for k, v in self.__dict__.items()
                    if v and k != 'main' and k != 'pos' and k != 'sets'])


class ArtClassifier(object):
    '''
    you should set <main stat>, <stat criterion>, and <stop criterion>\n
    then start simulation\n
    '''
    __subs = ['ATK', 'ATK_P', 'DEF', 'DEF_P',
              'HP', 'HP_P', 'ER', 'EM', 'CR', 'CD']

    __positions = ['flower', 'plume', 'sands', 'goblet', 'circlet']

    with open('./doc/distribution.json', 'r') as f:
        js = json.load(f)
        __distribution_main: Dict[str, Dict[str, int]] = js['main']
        __distribution_sub: Dict[str, int] = js['sub']

    def __init__(self) -> None:
        self.artifact_finish: Sequence[Tuple[int, Art]] = []
        self.artifact_abandon: Sequence[Tuple[int, int, Art]] = []
        # in the training data use last digit as classifier digit, so len(array) = 11
        # below is training data and optimal w* for SVM
        self.training_data: Mapping[str,
                                    Mapping[str, Sequence[Sequence[int]]]] = {}
        self.w = {}
        # below is you need to set
        self.main_stat = {}
        self.target_stat = []
        self.ignore_stat = []
        self.stop_criterion = None
        self.output_mode = False

    def set_main_stat(self, main_stat: Mapping[str, str]):
        '''
        set artifact's main stat\n
        \tmain_stat: Dict[str, str]
        '''
        self.main_stat = main_stat

    def set_stat_criterion(self, target: Sequence[str]):
        '''
        set target sub stat\n
        \ttarget: List[str]
        '''
        self.target_stat = target
        self.ignore_stat = [s for s in self.__subs
                            if s not in self.target_stat]

    def set_stop_criterion(self, f: Callable[[Any], bool]):
        '''
        set stopping criterion\n
        \tf: Callable:\t(np.array[...]) -> bool\n
        '''
        self.stop_criterion = f

    def judge_stop(self, array) -> bool:
        # get the top value artifact combination
        return self.stop_criterion(array)

    def set_output(self, flag: bool = True):
        self.output_mode = flag

    def clear_result(self):
        self.artifact_finish.clear()
        self.artifact_abandon.clear()

    def get_max_combinations(self) -> Dict[str, Any]:
        '''
        choose the combination that has max target stat\n
        \treturn: np.array
        '''
        # the default target set is 0, and there is a uncertain postion for set1
        # to simplify, set1 only occur in 'sands' / 'goblet' / 'circlet'
        # the sort method is simply to count target stat which is naive and need discussion
        def evaluate_artifact(a: Art) -> float:
            if not a:
                return 0
            v_a = a.to_array()
            v_w = np.array([0.2, 0.6, 0, 0, 0, 0, 0, 0, 1, 1])
            return np.matmul(v_a, v_w)

        def find_max_set1(stat_list: list, top_dict: dict) -> str:
            max_stat = ('', 0)
            for stat in stat_list:
                n1 = evaluate_artifact(top_dict[stat][1])
                n0 = evaluate_artifact(top_dict[stat][0])
                if n1-n0 > max_stat[1]:
                    max_stat = (stat, n1-n0)
            return max_stat

        result_artifact = dict.fromkeys(self.__positions)
        top_for_each_set = dict.fromkeys(self.__positions)
        for pos in self.__positions:
            top_for_each_set[pos] = [None, None]
        for tup in self.artifact_finish:
            art = tup[1]
            if art.sets == 0 and evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos][0]):
                top_for_each_set[art.pos][0] = art
            elif art.sets == 1 and evaluate_artifact(art) > evaluate_artifact(top_for_each_set[art.pos][1]):
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
                result_array += v.to_array()
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

    def train_data_input(self, path):
        with open(path, 'r') as f:
            self.training_data = json.load(f)

    def w_data_input(self, path):
        with open(path, 'r') as f:
            js = json.load(f)
            self.w = dict.fromkeys(js.keys())
            for pos, stat in js.items():
                self.w[pos] = {}
                for stat_num, stat_w in stat.items():
                    self.w[pos][stat_num] = np.array(stat_w)

    def w_data_output(self, path):
        with open(path, 'w') as f:
            w = dict.fromkeys(self.w.keys())
            for pos, stat in self.w.items():
                w[pos] = {}
                for stat_num in stat.keys():
                    w[pos][stat_num] = list(self.w[pos][stat_num])
            json.dump(w, f, indent=4)

    def train(self, train_data: Sequence[Sequence[int]]):
        '''
        train a SVM from training data\n
        \treturn: np.array
        '''
        X = []
        y = []
        for art_vector in train_data:
            classifier_y = art_vector.pop()
            X.append(art_vector)
            y.append(classifier_y)

        X = np.array(X)
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        y = np.array(y)
        Xy = X * y.reshape((-1, 1))

        def sigmoid(z):
            return 1.0 / (1 + np.exp(-z))

        def fp(w):
            '''the gradient of f'''
            return -(1-sigmoid(Xy@w)) @ Xy

        def gd_const(fp, x0, stepsize, tol=1e-2, maxiter=100000):
            '''constant gradient descent'''
            x = np.array(x0)
            for k in range(maxiter):
                grad = fp(x)
                if np.linalg.norm(grad) < tol:
                    break
                x -= stepsize * grad
            return x, k

        w0 = np.zeros(11)
        stepsize = 0.05
        maxiter = 100000

        w, k = gd_const(fp, w0, stepsize=stepsize, maxiter=maxiter)
        if k == maxiter-1:
            k = 'MAX'

        y_hat = 2*(X@w > 0)-1

        accuracy = sum(
            [1 for i in range(len(y)) if y_hat[i] == y[i]]) / float(len(y))
        if self.output_mode:
            print("iter times: {}".format(k))
            print("accuracy = {:.2f}%".format(accuracy*100))
            print(f'\nw* = \n{w}\n', '-'*50)
        return w

    def start_training(self):
        # check the preparation
        if not self.training_data:
            print('training data missing')
            return

        # train SVM
        self.w = dict.fromkeys(self.__positions)
        for pos, pos_v in self.training_data.items():
            self.w[pos] = {}
            for stat_num, train_data in pos_v.items():
                if self.output_mode:
                    print(
                        f'\npos = {pos}, stat_num = {stat_num}, training start\n')
                self.w[pos][stat_num] = self.train(train_data)
        if self.output_mode:
            print('---training ends---')
        return

    def sample_generation(self) -> Art:
        # generate position
        position = random.choice(self.__positions)

        # generate sets and main stat
        main_dis: Dict[str, int] = self.__distribution_main[position]
        main = random.choices(list(main_dis.keys()),
                              list(main_dis.values()), k=1)[0]
        sets = random.randint(0, 1)

        # generate sub stat
        sub_dis: Dict[str, int] = self.__distribution_sub.copy()
        sub_dis.pop(main, None)

        sample = Art()
        sample.generate(main, sets, position, sub_dis)
        return sample

    def start_simulation(self, max_times: int = 1000):
        # check the preparation
        if not self.stop_criterion \
                or not self.main_stat \
                or not self.target_stat \
                or not self.w:
            print('something not set')
            return

        if self.output_mode:
            print('\n--- simulation start ---\n')

        # simulate for max_times
        for i in range(max_times):
            # generate an artifact
            sample: Art = self.sample_generation()
            if self.output_mode:
                print(f'in <{i}> simultion get artifact below:')
                print(sample)

            # initialize before upgrade
            stat_now = sum([v for k, v in sample.__dict__.items()
                            if k in self.__subs])
            upgrade_time = 0
            upgrade_flag = np.matmul(np.append(sample.to_array(), [1]),
                                     self.w[sample.pos][str(stat_now)])

            # if main stat doesn't match, abandon.
            # if set doesn't match and not in 'sands', 'goblet', 'circlet', abandon
            if sample.main != self.main_stat[sample.pos]:
                upgrade_flag = -1
            if sample.sets == 1 and sample.pos not in ['sands', 'goblet', 'circlet']:
                upgrade_flag = -1

            while(upgrade_time < 5):
                if upgrade_flag < 0:
                    self.artifact_abandon.append(
                        (i+1, upgrade_time, sample))
                    break
                sample.upgrade()
                stat_now += 1
                upgrade_time += 1
                if upgrade_time == 5:
                    break
                upgrade_flag = np.matmul(np.append(sample.to_array(), [1]),
                                         self.w[sample.pos][str(stat_now)])

            if self.output_mode:
                print(f'upgrade stop at {upgrade_time} times')

            if upgrade_time == 5:
                self.artifact_finish.append((i+1, sample))
                if self.output_mode:
                    print('finish upgrade:', sample)
                    self.print_max_combinations()
                # judge the stopping criterion
                if len(self.artifact_finish) >= 5 and self.judge_stop(self.get_max_combinations_array()):
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


def stopping(array):
    value_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    flag = np.matmul(array, value_vector) >= 24
    value_vector2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    flag2 = np.matmul(array, value_vector2) >= 19
    return flag and flag2


if __name__ == '__main__':
    sim = ArtClassifier()
    sim.set_main_stat(dict(zip(['flower', 'plume', 'sands', 'goblet', 'circlet'],
                               ['HP', 'ATK', 'ER', 'ELECTRO_DMG', 'CR'])))
    sim.set_stat_criterion(['CD', 'CR', 'ATK_P'])
    sim.set_stop_criterion(stopping)

    sim.set_output()

    # use SVM to train classifier
    sim.train_data_input('./data/training_data1.json')
    sim.start_training()
    sim.w_data_output('./data/optimal_w_train.json')

    # # get classifier directly
    # sim.w_data_input('./data/optimal_w.json')

    # # sim.set_output(False)
    # for i in range(1):
    #     sim.start_simulation()
    #     sim.clear_result()
