import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data1 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\asgn1_data.csv')
data2 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\diabetes1.csv')
data3 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\heart_data.csv')

Y = data1['target']
X = data1.drop('target', axis = 1)
accuracies = []


#Defining node count
i_nd = 2
hid_nd = 5
out_nd = 2

class PSO_Part:
    def __init__(self, w_count, pos, vel):
        self.pos = np.random.uniform(
            pos[0], pos[1], (w_count,)
        )
        self.vel = np.random.uniform(
            vel[0], vel[1], (w_count,)
        )
        self.pers_best = np.inf
        self.pers_best_pos = np.zeros((w_count,))


def fit_func(X, Y, wgt):
    if isinstance(wgt, PSO_Part):
        wgt = wgt.pos

    w1 = wgt[0 : i_nd * hid_nd].reshape((i_nd, hid_nd))
    b1 = wgt[i_nd * hid_nd : (i_nd * hid_nd) + hid_nd].reshape((hid_nd,))
    w2 = wgt[(i_nd * hid_nd) + hid_nd : (i_nd * hid_nd) + hid_nd + (hid_nd * out_nd)].reshape((hid_nd, out_nd))
    b2 = wgt[(i_nd * hid_nd) + hid_nd + (hid_nd * out_nd) : (i_nd * hid_nd) + hid_nd + (hid_nd * out_nd) +
        out_nd].reshape((out_nd,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    exps = np.exp(logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    smp = len(probs)
    l_probs = -np.log(probs[range(smp), Y])

    return np.sum(l_probs) / smp

def predict(X, wgt):
    w1 = wgt[0 : i_nd * hid_nd].reshape((i_nd, hid_nd))
    b1 = wgt[i_nd * hid_nd : (i_nd * hid_nd) + hid_nd].reshape((hid_nd,))
    w2 = wgt[(i_nd * hid_nd) + hid_nd : (i_nd * hid_nd) + hid_nd + (hid_nd * out_nd)].reshape((hid_nd, out_nd))
    b2 = wgt[(i_nd * hid_nd) + hid_nd + (hid_nd * out_nd) : (i_nd * hid_nd) + hid_nd + (hid_nd * out_nd)
        + out_nd].reshape((out_nd,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    exps = np.exp(logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    Y_pred = np.argmax(probs, axis=1)
    return Y_pred


class PSO:
    def __init__(self, no_part, w_count, pos, vel, pso_inert, pso_c0, pso_c1):
        self.part = np.array(
            [PSO_Part(w_count, pos, vel) for i in range(no_part)]
        )
        self.global_best = np.inf
        self.global_best_pos = np.zeros((w_count,))
        self.pos = pos
        self.vel = vel
        self.inert = pso_inert
        self.c0 = pso_c0
        self.c1 = pso_c1
        self.w_count = w_count


    def run(self, fit_func, X, Y, run_count):

        for i in range(run_count):

            for particle in self.part:
                fitness = fit_func(X, Y, particle.pos)

                if fitness < particle.pers_best:
                    particle.pers_best = fitness
                    particle.pers_best_pos = particle.pos.copy()

                if fitness < self.global_best:
                    self.global_best = fitness
                    self.global_best_pos = particle.pos.copy()

            for particle in self.part:

                iw = np.random.uniform(self.inert[0], self.inert[1], 1)[0]
                particle.vel = (
                    iw * particle.vel
                    + (
                        self.c0
                        * np.random.uniform(0.0, 1.0, (self.w_count,))
                        * (particle.pers_best_pos - particle.pos)
                    )
                    + (
                        self.c1
                        * np.random.uniform(0.0, 1.0, (self.w_count,))
                        * (self.global_best_pos - particle.pos)
                    )
                )
                particle.pos = particle.pos + particle.vel

            if i % 100 == 0:
                final_wgt = self.global_best_pos
                Y_pred = predict(X, final_wgt)
                accuracy = find_accr(Y, Y_pred)
                print("Run #: ", i + 1, " loss: ", fitness, " accuracy: ", accuracy)
                accuracies.append(accuracy)

        print("global best loss: ", self.global_best)

    def final_global_best(self):
        return self.global_best_pos


def find_accr(Y, Y_pred):
    return (Y == Y_pred).mean()


if __name__ == "__main__":
    no_part = 100
    run_count = 1000
    w_count = ((i_nd * hid_nd) + hid_nd + (hid_nd * out_nd) + out_nd)

    wgt = (0.0, 1.0)
    lrn_rate = (0.0, 1.0)
    pso_inert = (0.85, 0.85)
    pso_c0 = 0.7
    pso_c1 = 0.9

    opt = PSO(no_part, w_count, wgt, lrn_rate, pso_inert, pso_c0, pso_c1)
    opt.run(fit_func, X, Y, run_count)
    final_wgt = opt.final_global_best()
    Y_pred = predict(X, final_wgt)
    accuracy = find_accr(Y, Y_pred)
    print("Accuracy: %.3f" % accuracy)

    plt.plot(accuracies)
    plt.xlabel('No. of Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.show()

