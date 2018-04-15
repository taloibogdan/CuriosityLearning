import random
import numpy as np
from model import StateActionPredictor as SAP
from model import ReinforcementLearner as RL
import time
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, size, start, nr_goals):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.viz = np.zeros(size)
        self.heatmap = self.ax1.imshow(np.random.random((size[0],size[1])))
        plt.show(block=False)
        self.map_size = size
        self.start = start
        self.nr_goals = nr_goals
        self.pos = start
        self.goals = []
        for i in range(nr_goals):
            self.goals.append([np.random.randint(0,size[0]),np.random.randint(0,size[1])])

    def plot(self,xs,ys):
        self.ax2.plot(xs,ys)
        self.fig.canvas.draw()

    def reset(self):
        self.pos = self.start
        self.goals = []
        for i in range(self.nr_goals):
            self.goals.append([np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1])])

    def get_state(self):
        map = []
        #print('=================================================================')
        for i in range(self.map_size[0]):
            row = []
            for j in range(self.map_size[1]):
                if self.pos in [[i,j],[i+1,j],[i-1,j],[i,j+1],[i,j-1]]:
                    row.append([1])
                #elif [i, j] in self.goals:
                #    row.append([0.5])
                else:
                    row.append([0])
            map.append(row)
            #print(row)
        map = np.array(map)
        #print('=================================================================')

        return map

    def do_action(self, action):
        if action == "W" and self.pos[0] > 1:
            self.pos[0] -= 1
        elif action == "A" and self.pos[1] > 1:
            self.pos[1] -= 1
        elif action == "S" and self.pos[0] < self.map_size[0] - 2:
            self.pos[0] += 1
        elif action == "D" and self.pos[1] < self.map_size[1] - 2:
            self.pos[1] += 1
        self.viz[self.pos[0]][self.pos[1]] += 1
        if self.pos in self.goals:
            self.goals.remove(self.pos)
            return 1
        else:
            return 0

    def show_heatmap(self):
        h = self.viz / np.max(self.viz) * .8
        h[self.pos[0]][self.pos[1]] = 1
        h[self.pos[0]-1][self.pos[1]] = 1
        h[self.pos[0]+1][self.pos[1]] = 1
        h[self.pos[0]][self.pos[1]-1] = 1
        h[self.pos[0]][self.pos[1]+1] = 1
        for p in self.goals:
            h[p[0]][p[1]] = 1
        """i = 1
        while r > 0.1 and i < 10:
            r -= 0.1
            h[-i][-1] = h[-i][-2] = 1
            i += 1
        if r > 0:
            h[-i][-1] = h[-i][-2] = r*10"""
        #h[-12][-1] = h[-12][-2] = 1
        #h[-14][-1] = h[-14][-2] = 1
        self.heatmap.set_array(h)
        self.fig.canvas.draw()

    def check_action(self, action):
        if (action == "W" and self.pos[0] > 1) or\
           (action == "A" and self.pos[1] > 1) or\
           (action == "S" and self.pos[0] < self.map_size[0] - 2) or\
           (action == "D" and self.pos[1] < self.map_size[1] - 2):
            return True
        return False

    def print(self):
        for i in range(self.map_size[0]):
            row = ''
            for j in range(self.map_size[1]):
                if [i, j] == self.pos:
                    row+='X'
                elif [i, j] in self.goals:
                    row+='R'
                else:
                    row+='O'
            print(row)

    @staticmethod
    def printstate(state):
        for i in range(len(state)):
            row = ''
            for j in range(len(state[i])):
                if state[i][j] == 1:
                    row += 'X'
                else:
                    row += 'O'
            print(row)
        print('--------------------------------')

    def get_available_action(self, check=True):
        acts = []
        for a in ["W","A","S","D"]:
            if self.check_action(a) or (not check):
                acts.append(a)
        return random.choice(acts)


testing = False
ACTS = ['W','A','S','D']
l = 42
s = Simulator([l,l], [2, 2], 0)
rl = RL([None,l,l,1],4)
sap = SAP([l,l,1],4,load_from_file=False)
rl.set_session(sap.sess)
sQ = None
s1 = s.get_state()


batch = []
xs = []
ys = []
tempys = []
for i in range(1000000000):

    e = 5 / ((i/5000)+10)
    """if i > 1000000:
        if i > 8000:
            time.sleep(0.5)
        act = rl.get_action([s1])
        k = ACTS[act[0]]
    else:"""
    k = s.get_available_action(False)
        #if np.random.rand() > e:
        #    act = rl.get_action([s1])
        #    k = ACTS[act[0]]
    if testing:
        k = input()
    k = k.upper()
    if k == "X":
        break

    r = s.do_action(k)
    s2 = s.get_state()

    """if sQ is None:
        sQ = rl.get_Qs([s1])
    sQ1 = rl.get_Qs([s2])"""
    act = [0, 0, 0, 0]
    k = ACTS.index(k)
    act[k] = 1
    act = np.array(act)

    if testing:
        print([round(x,2) for x in sap.pred_act([s1],[s2])])
    else:
        batch.append((s1,s2,act))
        if i % 1000 == 0 and i > 0:
            #print("Step ", i)
            #print("--------------------------------")
            #r1 = sap.pred_act([s1], [s2])
            #print(r1)
            s1s = [b[0] for b in batch]
            s2s = [b[1] for b in batch]
            acts = [b[2] for b in batch]
            for j in range(1):
                r2 = sap.train(s1s, s2s, acts, i/1000)
                print("Step",int(i/1000),".",j,r2[2])
                #print(r2[3])
                tempys.append(r2[2])
            if i % 1000 == 0:
                xs.append(int(i/1000))
                ys.append(np.average(tempys))
                s.plot(xs,ys)
                tempys = []
                sap.save_weights()
            if i>1000000:
                testing = True
            batch = []
        #print("--------------------------------")
    """r = r/2 + r2[0]/2
    if r > 1:
        r = 1
    rl.train([s1], sQ, sQ1, k, r, e)"""
    #print(ACTS[k])
    #s.print()
    #print(r2)
    #print(r)
    s1 = s2
    #sQ = sQ1
    if len(s.goals) < 0:
        s.reset()
        s1 = s.get_state()
        sQ = None
    if testing or i % 1000 == 0:
        s.show_heatmap()