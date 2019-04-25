'''
TODO:
    Consertar ganho de pontos na finalização do tempo de execução
    Melhorar taxa de aprendizagem
    Melhorar pista
    Melhorar grafico
'''



import ambiente as env
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt


output_dir = "memoria" #TODO: Parcialmente implementado

env = env.CarEnv(discrete_action=True)
state_size = env.n_sensor
action_size = 3
batch_size = 32
n_ep = 1000 #numero de episodios
n_st_ep = 1000 #numero de passos por episodio
state = env._get_state()

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        #fator de desconto
        self.gamma = 0.95
        #taxa de exploração
        self.epsilon = 1.0
        #taxa de diminuição da exploração
        self.epsilon_decay = 0.9999
        #randomização minima
        self.epsilon_min = 0.01
        #learning rate
        self.learning_rate = 0.001

        self.model = self._build_model()



    def _build_model(self):

        model = Sequential()

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)



#############################################################################################

done = False

scores = []
ultimos = []
counter = 0
medias = []

for e in range(n_ep):

    env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(n_st_ep):

        env.render()


        action = agent.act(state)

        next_state, reward, done = env.step(action)

        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state



        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_ep, time, agent.epsilon))
            scores.append(time)
            ultimos.append(time)
            counter = counter + 1
            if ultimos.__len__() >= 50:
                ultimos.pop(0)
            medias.append(np.mean(ultimos))
            if counter > 100:
                plt.plot(scores, 'ro')
                plt.plot(medias, 'k')
                plt.xlabel('Episódios')
                plt.ylabel('Pontuação')
                counter = 0
                plt.show()
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")