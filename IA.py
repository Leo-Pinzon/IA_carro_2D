'''
TODO:
    Consertar ganho de pontos na finalização do tempo de execução??
    Melhorar taxa de aprendizagem
    Melhorar pista
    Melhorar grafico
'''



import ambiente as env
import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt




output_dir = "data/DNNW.hdf5"
fatores_dir = "data/fatores.csv"

carregar_dados = True

env = env.CarEnv(discrete_action=True)
state_size = env.n_sensor
action_size = 3
batch_size = 1024
n_ep = 10000                          #numero de episodios
n_st_ep = 500                        #numero de passos por episodio
state = env._get_state()
n_graf = 800                         #numero maximo de dados apresentados no grafico

#----------------------PARAMETROS DA IA----------------------#

#fator de desconto das recompensas futuras estimadas
GAMMA = 0.95

#taxa de decisões aleatórias (taxa de exploração)
EPISILON = 0.8

#fator de diminuição de EPISILON
EPISILON_DECAY = 0.9999

#valor minimo alcançavel de EPISILON
EPISILON_MINIMO = 0.01

TAXA_DE_APRENDIZADO = 0.005

#----------------------PARAMETROS DA IA----------------------#

E_INI = 0 #episodio inicial


if carregar_dados and os.path.exists(fatores_dir):
    E_INI, GAMMA, EPISILON, EPISILON_DECAY, EPISILON_MINIMO, TAXA_DE_APRENDIZADO = np.loadtxt(fatores_dir, delimiter=",")
    E_INI = int(E_INI)



class DQNAgent:
    def __init__(self, state_size, action_size):


        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=15000)


        self.gamma = GAMMA
        self.epsilon = EPISILON
        self.epsilon_decay = EPISILON_DECAY
        self.epsilon_min = EPISILON_MINIMO
        self.learning_rate = TAXA_DE_APRENDIZADO

        self.model = self._build_model()



    def _build_model(self):

        model = Sequential()

        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) #TODO: Alterar modo de ativação?

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

    def save(self, name, name_, e, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate):
        #salva pesos da rede neural
        self.model.save_weights(name)

        #salva os fatores
        np.savetxt(name_, (e, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate), delimiter=',')


agent = DQNAgent(state_size, action_size)

legenda = "\n\nLEARNING RATE: {:.5f}\n" \
                  "GAMMA: {:.5f}\n" \
                  "ϵ: {:.5f}\n" \
                  "ϵ_DECAY: {:.5f}\n" \
                  "ϵ_MIN: {:.5f}\n".format(agent.learning_rate, agent.gamma, agent.epsilon, agent.epsilon_decay, agent.epsilon_min)




#############################################################################################

done = False

scores = []
ultimos = []
counter = 0
medias = []
lista_e = []
RENDER = 1

if(carregar_dados == True) and os.path.exists(output_dir):
    agent.model.load_weights(output_dir)

####
plt.ion()
fig = plt.figure(figsize=(20,2))
####

for e in range(E_INI,n_ep):

    env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(n_st_ep):

        env.render()

        action = agent.act(state)

        next_state, reward, done = env.step(action, time)

        reward = reward if not done else -50

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state


        if done or time == n_st_ep-1:
            print("Episódio: {}/{}, Tempo decorrido: {}, Ações aleatórias: {:.2%}".format(e, n_ep, time, agent.epsilon))

            user_feedback = "Ep. {}\nϵ = {:.3%}".format(e+1, agent.epsilon)

            env.viewer.legenda = user_feedback

            scores.append(time)
            ultimos.append(time)
            counter = counter + 1
            if ultimos.__len__() >= 50:
                ultimos.pop(0)
            medias.append(np.mean(ultimos))
            lista_e.append(e)

            if lista_e.__len__() > n_graf:
                lista_e.pop(0)
                medias.pop(0)
                scores.pop(0)
            plt.axis([lista_e[0], e, 0, n_st_ep])
            plt.scatter(lista_e, scores, c='blue', marker=".")
            plt.scatter(lista_e,medias,c='black', marker=".")
            plt.show()
            plt.pause(0.005)

            #if counter >= n_graf:
            #    plt.plot(scores, 'ro')
            #    plt.plot(medias, 'k')
            #    plt.xlabel('Episódios')
            #    plt.ylabel('Tempo até colisão')
            #    plt.text(0,(np.amax(scores)+np.amin(scores))/2,legenda)
            #    counter = 0
            #    plt.show()

            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir,fatores_dir,e,agent.gamma,agent.epsilon,agent.epsilon_decay,agent.epsilon_min,agent.learning_rate)

plt.show()