import tensorflow.keras
import pygad.kerasga
import pygad
import pandas as pd
import matplotlib.pyplot as plt


def fitness_func(solution, sol_idx):
    global X, Y, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=X)

    bce = tensorflow.keras.losses.BinaryCrossentropy()
    solution_fitness = 1.0 / (bce(Y, predictions).numpy() + 0.00000001)

    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

data1 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\asgn1_data.csv')
data2 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\diabetes1.csv')
data3 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\heart_data.csv')

Y = data3['target']
X = data3.drop('target', axis = 1)
accuracies = []


# Build the keras model using the functional API.
input_layer  = tensorflow.keras.layers.Input(13)
dense1_layer = tensorflow.keras.layers.Dense(4, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="sigmoid")(dense1_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model,num_solutions=40)


# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 5 # Number of generations.
num_parents_mating = 20 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights
# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

for k in range(20):
    # Start the genetic algorithm evolution.
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    predictions = pygad.kerasga.predict(model=model, solution=solution, data=X)
    ba = tensorflow.keras.metrics.BinaryAccuracy()
    ba.update_state(Y, predictions)
    accuracy = ba.result().numpy()
    print("Accuracy : ", accuracy)
    accuracies.append(accuracy)

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="Accuracy vs generations", linewidth=4)

# Plot the accuracies
plt.plot(accuracies)
plt.xlabel('No. of Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iteration')
plt.show()
