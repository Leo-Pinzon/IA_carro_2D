import ambiente as env
import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import csv
import keyboard
import winsound

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

output_dir = "data/DNNW.hdf5"
fatores_dir = "data/fatores.csv"
mostrar_grafico_tempo = True
mostrar_grafico_recompensa = True
carregar_dados = False
treinarCircuito = False

env = env.CarEnv(discrete_action=True)
state_size = env.n_sensor
action_size = 3
batch_size = 1024
n_ep = 2000                 #numero maximo de episodios
n_st_ep = 300               #numero de passos por episodio
state = env._get_state()
n_graf = 500                #numero maximo de dados apresentados no grafico

#----------------------PARAMETROS DA IA----------------------#
#fator de desconto das recompensas futuras estimadas
GAMMA = 0.99995
#taxa de decisões aleatórias (taxa de exploração)
EPISILON = 0.9
#fator de diminuição de EPISILON
EPISILON_DECAY = 0.999995
#valor minimo de EPISILON
EPISILON_MINIMO = 0.005
#taxa de aprendizado
TAXA_DE_APRENDIZADO = 0.009
#----------------------PARAMETROS DA IA----------------------#

E_INI = 0 #episodio inicial

if carregar_dados and os.path.exists(fatores_dir):
    E_INI, GAMMA, EPISILON, EPISILON_DECAY, EPISILON_MINIMO, TAXA_DE_APRENDIZADO = np.loadtxt(fatores_dir, delimiter=",")
    E_INI = int(E_INI)

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=7000)

        self.gamma = GAMMA
        self.epsilon = EPISILON
        self.epsilon_decay = EPISILON_DECAY
        self.epsilon_min = EPISILON_MINIMO
        self.learning_rate = TAXA_DE_APRENDIZADO

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()

        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))

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

keyboard.add_hotkey('1', env.alteraPistas, args='1')
keyboard.add_hotkey('2', env.alteraPistas, args='2')
keyboard.add_hotkey('3', env.alteraPistas, args='3')

auxPista = '2'


def salva_grafico(tempo):
    tempo = tempo * 1.0
    with open('data/grafico.txt', mode='a') as arq:
        arq = csv.writer(arq, delimiter=',')
        arq.writerow([tempo])

done = False

finalizaTreinamento = False

pontuacaoAcumulada = 0
melhorPt = 0
scores = []
ultimos = []
ultimos_recompensas = []
med_ultimos = 0.
counter = 0
medias = []
lista_e = []
RENDER = 1

if(carregar_dados == True) and os.path.exists(output_dir):
    agent.model.load_weights(output_dir)

for e in range(E_INI,n_ep):

    env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(n_st_ep):

        if keyboard.is_pressed('ctrl+j'):
            while (keyboard.is_pressed('ctrl+j')):
                print()
            agent.save(output_dir, fatores_dir, e, agent.gamma, agent.epsilon, agent.epsilon_decay, agent.epsilon_min,
                       agent.learning_rate)
            env.viewer.close()
            exit()

        env.render()

        action = agent.act(state)

        next_state, reward, done = env.step(action, time)

        reward = reward

        pontuacaoAcumulada += reward

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        #LEGENDA
        user_feedback = "Ep. {}\nϵ = {:.3%}\nLr = {}\nMelhor Tempo = {}\nRecompensa = {:.0f}".format(e + 1,
                                                                                                           agent.epsilon,
                                                                                                           agent.learning_rate,
                                                                                                           env.melhorTempo[
                                                                                                               0],
                                                                                                           pontuacaoAcumulada)

        if env.viewer != None:
            env.viewer.legenda = user_feedback

        if done or time == n_st_ep-1:

            print("Episódio: {}/{}, Tempo decorrido: {}, Ações aleatórias: {:.2%}".format(e, n_ep, time, agent.epsilon))

            # LEGENDA
            user_feedback = "Ep. {}\nϵ = {:.3%}\nLr = {}\nMelhor Tempo = {}\nRecompensa = {:.0f}".format(e + 1,
                                                                                                               agent.epsilon,
                                                                                                               agent.learning_rate,
                                                                                                               env.melhorTempo[
                                                                                                                   0],
                                                                                                               pontuacaoAcumulada)

            if env.viewer != None:
                env.viewer.legenda = user_feedback
            #

            scores.append(time)
            ultimos.append(time)
            ultimos_recompensas.append(pontuacaoAcumulada*(2/3))
            if pontuacaoAcumulada > melhorPt:
                melhorPt = pontuacaoAcumulada
            counter = counter + 1
            if ultimos.__len__() > 40:
                ultimos.pop(0)
                ultimos_recompensas.pop(0)
            med_ultimos = np.mean(ultimos)

            if abs(med_ultimos - env.melhorTempo[0]) < 20:
                agent.save(output_dir, fatores_dir, e, agent.gamma, agent.epsilon, agent.epsilon_decay, agent.epsilon_min, agent.learning_rate)
                print("----------TREINAMENTO ENCERRADO----------")

                winsound.Beep(frequency, duration)

                print("EP:",e)
                exit()

            medias.append(med_ultimos)
            lista_e.append(e)

            pontuacaoAcumulada = 0

            salva_grafico(time)

            if lista_e.__len__() > n_graf:
                lista_e.pop(0)
                medias.pop(0)
                scores.pop(0)

            if env.viewer != None:
                if mostrar_grafico_tempo:
                    env.viewer.grafico.clear()
                    env.viewer.grafico.append(100)
                    env.viewer.grafico.append(env.yGraf)
                    for i in range(len(ultimos)):
                        env.viewer.grafico.append(int((i*20)+100))
                        env.viewer.grafico.append(int(ultimos[i]/2+env.yGraf))
                    env.viewer.grafico.append(((len(ultimos)-1)*20)+100)
                    env.viewer.grafico.append(env.yGraf)
                    if(env.melhorTempo[0] != np.inf):
                        env.viewer.melhorTempoGraf = int(env.melhorTempo[0]/2)

                if mostrar_grafico_recompensa:
                    env.viewer.grafico_rec.clear()
                    env.viewer.grafico_rec.append(100)
                    env.viewer.grafico_rec.append(env.yGraf)
                    for i in range(len(ultimos_recompensas)):
                        env.viewer.grafico_rec.append(int((i*20)+100))
                        env.viewer.grafico_rec.append(int(ultimos_recompensas[i]/2+env.yGraf))
                    env.viewer.grafico_rec.append(((len(ultimos_recompensas)-1)*20)+100)
                    env.viewer.grafico_rec.append(env.yGraf)
                    env.viewer.melhorPtGraf = int(melhorPt/3)

            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 10 == 0 and treinarCircuito:
        winsound.Beep(600, 250)
        if treinarCircuito:
            if auxPista == '2':
                env.alteraPistas('3')
                auxPista = '3'
            else:
                env.alteraPistas('2')
                auxPista = '2'

    if e % 50 == 0:
        agent.save(output_dir,fatores_dir,e,agent.gamma,agent.epsilon,agent.epsilon_decay,agent.epsilon_min,agent.learning_rate)