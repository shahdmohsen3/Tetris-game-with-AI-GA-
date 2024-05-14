import random
random.seed(200)
import PlayGame
import tetris_base as game
import time
import PlayGame as playGame

class TetrisGeneticAlgorithm:

    def __init__(self, weights=None, population=None):
        self.weights = weights if weights else []
        self.population = population if population else []
        self.score = 0



    def initialize_population(self, population_size, num_chromosomes):
        population = []
        for _ in range(population_size):
            individual = [random.uniform(-1.0, 1.0) for _ in range(num_chromosomes)]
            population.append(individual)
        return population

    def selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [score / total_fitness for score in fitness_scores]

        parent1 = self.roulette_wheel_select_individual(population, selection_probs)
        parent2 = self.roulette_wheel_select_individual(population, selection_probs)

        return parent1, parent2

    def roulette_wheel_select_individual(self, population, selection_probs):
        r = random.random()
        cumulative_prob = 0.0
        for i, prob in enumerate(selection_probs):
            cumulative_prob += prob
            if r < cumulative_prob:
                return population[i]

    def single_point_crossover(self, parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1, parent2

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutation(self, individual, mutation_rate):
        mutated_individual = []
        for gene in individual:
            if random.random() < mutation_rate:
                if isinstance(gene, float):
                    mutated_gene = gene + random.uniform(-0.1, 0.1)
                else:
                    mutated_gene = gene
                mutated_individual.append(mutated_gene)
            else:
                mutated_individual.append(gene)
        return mutated_individual

    def calculate_fitness(self, scores):
        normalized_scores = [score / max(scores) for score in scores]
        fitness_values = [score ** 2 for score in normalized_scores]
        return fitness_values

    def evaluate_chromosome(self, chromosome, population):
        # Run the game with the provided weights and return the score as fitness
        self.weights = chromosome
        self.population = population
        print("Weights: ",chromosome)
        ans=playGame.run_game_with_ai(chromosome)
        print("Score: ",ans)
        return ans

if __name__ == "__main__":
    # Genetic Algorithm Parameters
    POPULATION_SIZE = 12
    NUM_CHROMOSOMES = 7
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.1
    NUM_GENERATIONS = 600

    # Initialize genetic algorithm
    ga = TetrisGeneticAlgorithm()

    best_weights = [0, 0]  # Initialize a list to store the weights of the top two scores
    best_scores = [float('-inf'), float('-inf')]

    # Initialize population
    population = ga.initialize_population(POPULATION_SIZE, NUM_CHROMOSOMES)
    cnt_of_generation=1
    # Evolution loop
    for generation in range(NUM_GENERATIONS):
        print("Generation: ",cnt_of_generation)
        cnt_of_generation=cnt_of_generation+1
        # Evaluate fitness of each chromosome
        fitness_scores = [ga.evaluate_chromosome(chromosome, population) for chromosome in population]


        # Selection
        selected_population = []
        for _ in range(len(population)):
            parent1, parent2 = ga.selection(population, fitness_scores)
            selected_population.append(parent1)
            selected_population.append(parent2)

        # Crossover
        next_generation = []
        for i in range(0, len(selected_population), 2):
            child1, child2 = ga.single_point_crossover(selected_population[i], selected_population[i + 1],
                                                       CROSSOVER_RATE)
            next_generation.append(child1)
            next_generation.append(child2)

        # Mutation
        population = [ga.mutation(chromosome, MUTATION_RATE) for chromosome in next_generation]



        print("After Selection , Crossover and Mutation: ")
        scores =[ga.evaluate_chromosome(chromosome, population) for chromosome in population]


        for score, weights in zip(scores, population):
            if score > best_scores[0]:
                best_scores[1] = best_scores[0]
                best_weights[1] = best_weights[0]
                best_scores[0] = score
                best_weights[0] = weights
            elif score > best_scores[1]:
                best_scores[1] = score
                best_weights[1] = weights



        print(f"Generation {cnt_of_generation - 1}: Best weights = {best_weights}")
        print(f"Generation {cnt_of_generation - 1}: Best scores = {best_scores}")


        # best_score = playGame.run_game_with_ai(best_weights)
        print(f"Generation {cnt_of_generation-1 }: Best weights = {best_weights}  ")
        print("_________________________________________________________________________________________________")


    print("Best weights found:", best_weights[0])
    print("Score with best weights:", best_scores[0])
