import numpy as np
import pyglet
from Pistas import pista


pyglet.clock.set_fps_limit(10000)

#Coordenada x dos checkpoints
checkpoint_a = 200
checkpoint_b = 700


class CarEnv(object):
    n_sensor = 5 #numero de sensores
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (1366, 740)
    sensor_max = 45.
    start_point = [1200, 50]
    speed = 110.
    dt = 0.1
    melhorTempo = [np.infty,0] #[ultimo melhor tempo,primeiro tempo de finalização registrado]

    # Coordenada Y grafico
    yGraf = 550

    def __init__(self, discrete_action=True):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]

        self.terminal = False
        # node1 (x, y, r, w, l),
        self.car_info = np.array([0, 0, 0, 10, 23], dtype=np.float64)   # car coordination
        self.pista_a = np.array([ #superior esquerdo
            [0, 220],
            [50, 330],
            [180, 400],
            [0, 400],

        ])
        self.pista_b = np.array([ #inferior esquerdo
            [0, 180],
            [50, 70],
            [180, 0],
            [0, 0],
        ])

        self.pista_c = np.array([ #inferior central
            [300, 0],
            [683, 100],
            [1066, 0],
            [300, 0],
        ])

        self.pista_d = np.array([ #superior central
            [300, 400],
            [683, 300],
            [1066, 400],
            [300, 400],
        ])

        self.pista_e = np.array([ #central
            [160, 100],
            [100, 200],
            [160, 300],
            [700, 200],
        ])

        self.pista_f = np.array([
            [1366, 60],
            [1366, 100],
            [1366, 340],
            [600, 200],
        ])

        self.pista_g = np.array([
            [1366, 400],
            [1366, 750],
            [0, 750],
            [0, 400],
        ])

        self.check_box = np.array([
            [1366, 400],
            [1366, 300],
            [1266, 300],
            [1266, 400],
        ])

        #Coordenadas para renderização das linhas
        self.check_lines = np.array([
            [checkpoint_a, 0],
            [checkpoint_a, 400],
            [checkpoint_b, 400],
            [checkpoint_b, 0],
        ])

        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor, 3))  # n sensors, (distance, end_x, end_y)

    def alteraPistas(self, n):
        self.pista_a, self.pista_b, self.pista_c, self.pista_d, self.pista_e, self.pista_f, self.pista_g, self.check_box, self.start_point = pista(n)
        self.viewer = None
        self.terminal = True

    def step(self, action, time):
        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30  # max r = 6 degree
        self.car_info[:2] = self.car_info[:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        self._update_sensor()
        s = self._get_state()
        r = 1

        if self.terminal:
            r = (time-250)/50

        if (self.car_info[0] > checkpoint_a and self.car_info[0] < checkpoint_a + 11) or (self.car_info[0] > checkpoint_b and self.car_info[0] < checkpoint_b + 11):
            r = 10

        if self.car_info[0] > self.check_box[2][0] and self.car_info[1] > self.check_box[2][1] \
                and self.car_info[0] < self.check_box[0][0] and self.car_info[1] < self.check_box[0][1]:

            if self.melhorTempo[1] == 0:
                self.melhorTempo[1] = time

            difTempo = time - self.melhorTempo[1]

            r = ( 40 / (1 + np.exp(((8/7)*difTempo)+4)) ) + 30

            if self.melhorTempo[0] > time:
                self.melhorTempo[0] = time

            self.terminal = True

        return s, r, self.terminal

    def reset(self):
        self.terminal = False
        self.car_info[:3] = np.array([*self.start_point, -np.pi])
        self._update_sensor()
        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.pista_a, self.pista_b, self.pista_c, self.pista_d, self.pista_e, self.pista_f, self.pista_g, self.check_box, self.check_lines)
        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def _update_sensor(self):
        cx, cy, rotation = self.car_info[:3]

        n_sensors = len(self.sensor_info)
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)

        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si, -2:]]

            # obstacle collision
            for oi in range(len(self.pista_a)):
                p = self.pista_a[oi]
                r = self.pista_a[(oi + 1) % len(self.pista_a)] - self.pista_a[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_b)):
                p = self.pista_b[oi]
                r = self.pista_b[(oi + 1) % len(self.pista_b)] - self.pista_b[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_c)):
                p = self.pista_c[oi]
                r = self.pista_c[(oi + 1) % len(self.pista_c)] - self.pista_c[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_d)):
                p = self.pista_d[oi]
                r = self.pista_d[(oi + 1) % len(self.pista_d)] - self.pista_d[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_e)):
                p = self.pista_e[oi]
                r = self.pista_e[(oi + 1) % len(self.pista_e)] - self.pista_e[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_f)):
                p = self.pista_f[oi]
                r = self.pista_f[(oi + 1) % len(self.pista_f)] - self.pista_f[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi in range(len(self.pista_g)):
                p = self.pista_g[oi]
                r = self.pista_g[(oi + 1) % len(self.pista_g)] - self.pista_g[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            # window collision
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for oi in range(4):
                p = win_coord[oi]
                r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = p + t * r
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(intersection - q))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]
            if distance < self.car_info[-1]/2:
                self.terminal = True


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, car_info, sensor_info, obstacle_coords_a, obstacle_coords_b, obstacle_coords_c, obstacle_coords_d, obstacle_coords_e, obstacle_coords_f, obstacle_coords_g, chegada_coords, checkpoint_coords):
        super(Viewer, self).__init__(width, height, resizable=False, caption='IA Veículo 2D', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=0, y=25)
        pyglet.gl.glClearColor(*self.color['background'])

        self.car_info = car_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        midground = pyglet.graphics.OrderedGroup(1)
        foreground = pyglet.graphics.OrderedGroup(2)

        self.sensors = []
        line_coord = [0, 0] * 2
        c = (255, 0, 0) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))

        car_box = [0, 0] * 4
        c = (49, 86, 255) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', c))

        self.pistaE = obstacle_coords_e

        #LINHA DE CHEGADA
        c = (50, 255, 50) * 4
        self.checkpoint = self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', chegada_coords.flatten()),
                                         ('c3B', c))

        #CHECKPOINTS
        self.checkpoints = self.batch.add(4, pyglet.gl.GL_LINE_LOOP, background, ('v2f', checkpoint_coords.flatten()),
                                         ('c3B', c))

        c = (100, 100, 100) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_a.flatten()), ('c3B', c))
        self.obstacle2 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_b.flatten()), ('c3B', c))
        self.obstacle3 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_c.flatten()), ('c3B', c))
        self.obstacle4 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_d.flatten()), ('c3B', c))
        self.obstacle5 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', self.pistaE.flatten()), ('c3B', c))
        self.obstacle6 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_f.flatten()), ('c3B', c))
        self.obstacle7 = self.batch.add(4, pyglet.gl.GL_QUADS, midground, ('v2f', obstacle_coords_g.flatten()), ('c3B', c))

        self.grafico = []
        self.grafico_rec = []
        self.melhorTempoGraf = 0
        self.legenda = " "
        self.melhorPtGraf = 0

    def render(self):
        self.batch.draw()
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')

        self.info = pyglet.text.Label(self.legenda,
                                      font_name='Arial Black',
                                      font_size=15,
                                      color=(255, 255, 255, 120),
                                      width=300,
                                      multiline=True,
                                      x=1000, y=680)

        verde = (50, 200, 50)
        vermelho = (200, 50, 50)
        pyglet.gl.glLineWidth(3)
        pyglet.graphics.draw(int(len(self.grafico_rec) / 2), pyglet.gl.GL_LINE_LOOP, ('v2i', self.grafico_rec), ('c3B', vermelho * int(len(self.grafico_rec)/2)))
        pyglet.graphics.draw(int(len(self.grafico)/2), pyglet.gl.GL_LINE_LOOP, ('v2i', self.grafico))
        pyglet.gl.glLineWidth(1)
        for i in range(7):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINE_LOOP, ('v2i', (100,CarEnv.yGraf+(int(i*50/2)),900,CarEnv.yGraf+(int(i*50/2)))))
        pyglet.graphics.draw(2, pyglet.gl.GL_LINE_LOOP, ('v2i', (100, CarEnv.yGraf + self.melhorTempoGraf, 900, CarEnv.yGraf + self.melhorTempoGraf)), ('c3B', verde * 2))
        pyglet.graphics.draw(2, pyglet.gl.GL_LINE_LOOP, ('v2i', (100, CarEnv.yGraf + self.melhorPtGraf, 900, CarEnv.yGraf + self.melhorPtGraf)), ('c3B', vermelho * 2))
        self.info.draw()

        self.flip()


    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def on_close(self):
        self.close()

    def _update(self):
        cx, cy, r, w, l = self.car_info

        # sensores
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # carro
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys

if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done = env.step(env.sample_action())
            if done:
                break