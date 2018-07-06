import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CircleDrive2D:
    # r: the radius of the circle
    # theta: the rotation of the control
    # friforce: the friction force
    # dt: delta time
    # rg: eligible error range
    def __init__(self, r=1.0, theta=0.0, friforce=0.0, dt=0.1, rg=1.0, nsteps=1000):
        self.radius = r
        self.trans_theta = theta
        self.friction_force = friforce
        self.dt = dt
        self.upper_x = 2 * r
        self.lower_x = -2 * r
        self.upper_y = 3 * r
        self.lower_y = -r
        print('x ~ [%.3f, %.3f], y ~ [%.3f, %.3f]\n' % (self.lower_x, self.upper_x,
            self.lower_y, self.upper_y))
        self.range = rg
        self.nsteps = nsteps

        self.rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def reset(self):
        self.pos_list = []
        self.pos = np.zeros((2))
        self.v = np.zeros((2))
        self.pos_list.append(self.pos)


    def step(self, a):
        old_pos = self.pos
        old_v = self.v
        
        # rotate the a_v
        cur_a = np.array(a)
        cur_a = np.reshape(cur_a, (2, 1))
        cur_a = np.dot(self.rotation, cur_a)
        cur_a = np.reshape(cur_a, (2,))
        # effect of friction force
        magnitude = np.sqrt(np.sum(cur_a * cur_a))
        magnitude = magnitude - self.friction_force
        if magnitude < 1e-9:
            cur_a = np.zeros((2))
        else:
            print(magnitude / (magnitude + self.friction_force))
            cur_a *= magnitude / (magnitude + self.friction_force)

        print('refactor a: [%.3f, %.3f]' % (cur_a[0], cur_a[1]))

        new_v = old_v + cur_a * self.dt
        new_pos = old_pos + old_v * self.dt + 0.5 * self.dt * self.dt * cur_a
        self.pos = new_pos
        self.v = new_v
        self.pos_list.append(new_pos)

        return self.state(), self.reward(), self.check()

    def state(self):
        return np.concatenate([self.pos, self.v])


    def reward(self):
        dif = self.pos - np.array([0.0, self.radius])
        dist = np.sqrt(np.sum(dif * dif))
        if np.abs(dist - self.radius) < self.range:
            return 1.0
        else:
            return 0.0
            bias = (np.abs(dist - self.radius) - self.range)
            return 1 - bias * bias


    def set_state(self, pos, v):
        self.pos = pos
        self.v = v
        self.pos_list.append(self.pos)


    def check(self):
        if (len(self.pos_list) > self.nsteps):
            return False
        if (self.pos[0] > self.upper_x) or (self.pos[0] < self.lower_x) or \
            (self.pos[1] > self.upper_y) or (self.pos[1] < self.lower_x):
            return False
        return True


    def output_img(self, filepath=None, is3D=False):
        xdata = [pos[0] for pos in self.pos_list]
        ydata = [pos[1] for pos in self.pos_list]

        steps = len(self.pos_list)
        zdata = [(i + 0.0) / steps for i in range(steps)]
        if is3D:
            figure = plt.figure(figsize=(7, 7))
            ax = figure.add_subplot(111, projection='3d')
            ax.plot(xdata, ydata, zdata)
        else:
            plt.figure(figsize=(7, 7))
            plt.plot(xdata, ydata, color = 'red', linewidth=1)
        plt.xlim(self.lower_x, self.upper_x)
        plt.ylim(self.lower_y, self.upper_y)
        if filepath == None:
            plt.show()
        else:
            plt.savefig(filepath)


if __name__ == '__main__':
    target = np.array([0.0, 1.5])
    env = CircleDrive2D(r=1.5, dt=0.02, nsteps=2000)
    env.reset()
    env.set_state(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    pos = np.array([0.0, 0.0])
    while True:
        a = (target - pos) / (1.5 * 1.5)
        state, rd, cont = env.step(a)
        pos = state[0:2]
        print(pos)
        if cont == False:
            break
    env.output_img(is3D=True)


