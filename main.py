from matplotlib import pyplot as plt
import time
from random import randint, uniform

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class DecryptTextGA:
    def __init__(self):
        # GA Parameters
        self.generations = 500
        self.population_size = 300
        self.tournament_size = 20
        self.tournament_winner_probability = 0.75
        self.crossover_probability = 0.65
        self.crossover_points_count = 5
        self.mutation_probability = 0.2
        self.elitism_percentage = 0.15
        self.terminate = 50
        self.count_fitness_calls = 0

        self.letter_weight = 0.5
        self.pair_weight = 1

        self.mutation_count = 20

        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.elitism_count = int(self.elitism_percentage * self.population_size)
        self.crossover_count = self.population_size - self.elitism_count

        self.tournament_probabilities = [self.tournament_winner_probability]

        for i in range(1, self.tournament_size):
            probability = self.tournament_probabilities[i - 1] * (1.0 - self.tournament_winner_probability)
            self.tournament_probabilities.append(probability)

        self.dict_words = self.get_common_words()

    def get_ngram_frequency(self, filename):
        ngram_frequency = {}

        file = open(filename, 'r')
        content = file.readlines()
        for row in content:
            row = row.strip()
            freq, letter = row.split('\t')
            ngram_frequency[letter] = float(freq)

        return ngram_frequency

    def get_common_words(self):
        common_words = {}

        file = open("dict.txt", 'r')
        content = file.readlines()
        for i, row in enumerate(content):
            row = row.strip()
            common_words[row] = i

        return common_words

    def generate_ngrams(self, word, n):
        ngrams = [word[i:i + n] for i in range(len(word) - n + 1)]

        processed_ngrams = []
        for ngram in ngrams:
            if ngram.isalpha():
                ngram_upper = ngram.upper()
                processed_ngrams.append(ngram_upper)

        return processed_ngrams

    def decrypt(self, key):
        letter_mapping = {}

        for i in range(26):
            k = LETTERS[i]
            v = key.upper()[i]

            letter_mapping[k] = v

        decrypted_text = ''
        for character in self.ciphertext:
            if character not in LETTERS:
                decrypted_text += character
            else:
                decrypted_text += letter_mapping[character]

        return decrypted_text

    def calculate_key_fitness(self, text):
        letters = self.generate_ngrams(text, 1)
        pairs = self.generate_ngrams(text, 2)

        letter_fitness = 0
        for letter in letters:
            frequency = self.letter_frequency[letter]
            letter_fitness += frequency

        pair_fitness = 0
        for pair in pairs:
            frequency = self.pair_frequency[pair]
            pair_fitness += frequency


        fitness = (letter_fitness * self.letter_weight) + (pair_fitness * self.pair_weight)
        return fitness

    def merge_keys(self, parent1, parent2):
        child = [None] * 26
        count = 0

        # set for child same letters as parent1 has in random points
        while count < self.crossover_points_count:
            idx = randint(0, len(parent1) - 1)

            if not child[idx]:  # if the child doesnt have a letter in this index yet
                child[idx] = parent1[idx]
                count += 1

        for letter in parent2:
            if letter not in child:
                for i in range(len(child)):
                    if not child[i]:
                        child[i] = letter
                        break
        return ''.join(child)

    def many_mutate_key(self, key):
        for i in range(self.mutation_count):
            a = randint(0, len(key) - 1)
            b = randint(0, len(key) - 1)

            key = list(key)
            # swap
            temp = key[a]
            key[a] = key[b]
            key[b] = temp
        return ''.join(key)

    def mutate_key(self, key):
        a = randint(0, len(key) - 1)
        b = randint(0, len(key) - 1)

        key = list(key)
        # swap
        temp = key[a]
        key[a] = key[b]
        key[b] = temp

        return ''.join(key)

    def initialization(self):
        population = []

        for _ in range(self.population_size):
            key = ''
            while len(key) < 26:
                idx = randint(0, 25)
                if self.letters[idx] not in key:
                    key += LETTERS[idx]
            population.append(key)
        return population

    def evaluation(self, population):
        fitness = []

        for key in population:
            decrypted_text = self.decrypt(key)
            key_fitness = self.calculate_key_fitness(decrypted_text)
            self.count_fitness_calls += 1
            fitness.append(key_fitness)
        return fitness

    def elitism(self, population, fitness):
        population_fitness = {}

        for i in range(self.population_size):
            key = population[i]
            value = fitness[i]
            population_fitness[key] = value

        population_fitness = {k: v for k, v in sorted(population_fitness.items(), key=lambda item: item[1])}
        sorted_population = list(population_fitness.keys())

        elite_population = sorted_population[-self.elitism_count:]
        return elite_population

    def tournament_selection(self, population, fitness):
        population_copy = population.copy()
        selected_keys = []

        for a in range(2):
            tournament_population = {}

            for _ in range(self.tournament_size):
                r = randint(0, len(population_copy) - 1)
                key = population_copy[r]
                key_fitness = fitness[r]

                tournament_population[key] = key_fitness
                population_copy.pop(r)

            sorted_tournament_population = {k: v for k, v in
                                            sorted(tournament_population.items(), key=lambda item: item[1],
                                                   reverse=True)}
            tournament_keys = list(sorted_tournament_population.keys())

            index = -1
            selected = False
            while not selected:
                index = randint(0, self.tournament_size - 1)
                probability = self.tournament_probabilities[index]

                r = uniform(0, self.tournament_winner_probability)
                selected = (r < probability)

            selected_keys.append(tournament_keys[index])

        return selected_keys[0], selected_keys[1]

    def reproduction(self, population, fitness):
        crossover_population = []

        while len(crossover_population) < self.crossover_count:

            parent_one, parent_two = self.tournament_selection(population, fitness)

            child_one = self.merge_keys(parent_one, parent_two)
            child_two = self.merge_keys(parent_two, parent_one)

            crossover_population += [child_one, child_two]

        crossover_population = self.mutation(crossover_population, self.crossover_count)

        return crossover_population

    def mutation(self, population, population_size):
        for i in range(population_size):
            r = uniform(0, 1)

            if r < self.mutation_probability:
                key = population[i]
                mutated_key = self.mutate_key(key)

                population[i] = mutated_key

        return population

    def additional_mutations(self, population, N):
        for _ in range(N):
            for i in range(len(population)):
                key = population[i]
                mutated_key = self.mutate_key(key)

                population[i] = mutated_key

        return population

    def convert_to_plaintext(self, decrypted_text):
        plaintext = list(decrypted_text)
        for i in range(len(plaintext)):
            if self.lettercase[i]:
                plaintext[i] = plaintext[i].lower()
        plaintext = ''.join(plaintext)

        return plaintext

    def solve(self, ciphertext=''):
        # Defining ciphertext
        self.ciphertext = ciphertext

        # Formatting ciphertext
        self.lettercase = [ch.islower() and ch.isalpha() for ch in self.ciphertext]
        self.ciphertext = self.ciphertext.upper()

        # Getting letters and pair-letters frequency
        letters = 'Letter_Freq.txt'
        self.letter_frequency = self.get_ngram_frequency(letters)

        pairs = 'Letter2_Freq.txt'
        self.pair_frequency = self.get_ngram_frequency(pairs)

        # run
        population = self.initialization()

        highest_fitness = 0
        count_stable_maximum = 0
        avg_fitness_list = []
        max_fitness_list = []
        lamark_fitness_list = []
        darwin_fitness_list = []
        for no in range(self.generations + 1):
            fitness = self.evaluation(population)
            elitist_population = self.elitism(population, fitness)
            crossover_population = self.reproduction(population, fitness)

            population = elitist_population + crossover_population

            #for i in range(len(population) - 1):
            #    mutant = self.many_mutate_key(population[i])
            #    new_fitness = self.calculate_key_fitness(mutant)
            #    if new_fitness > fitness[i]:
                    # fitness[i] = new_fitness
                    # population[i] = mutant      # lamarck
                    # fitness[i] = new_fitness  # darwin

            # Terminate if highest_fitness not increasing
            if highest_fitness == max(fitness):
                count_stable_maximum += 1
            else:
                count_stable_maximum = 0

            if count_stable_maximum >= self.terminate:
                break

            highest_fitness = max(fitness)
            max_fitness_list.append(highest_fitness)
            average_fitness = sum(fitness) / self.population_size
            avg_fitness_list.append(average_fitness)

            index = fitness.index(highest_fitness)
            key = population[index]
            decrypted_text = self.decrypt(key)
            plaintext = self.convert_to_plaintext(decrypted_text)

            print('[Generation ' + str(no) + ']', )
            print('Average Fitness:', average_fitness)
            print('Max Fitness:', highest_fitness)
            print('Key:', key)
            print('Decrypted Text:\n' + plaintext + '\n')
        print('Number of calls to fitness function: ', self.count_fitness_calls)
        plaintext = self.convert_to_plaintext(decrypted_text)
        return plaintext, key, avg_fitness_list, max_fitness_list


def main():
    with open("enc.txt") as encrypt_file:
        ciphertext = encrypt_file.read().strip()

    start_time = time.time()

    plaintext, key, avg_fitness_list, max_fitness_list = DecryptTextGA().solve(ciphertext)

    end_time = time.time()

    # write the decrypted text to file
    with open("plain.txt", "w") as decrypted_file:
        decrypted_file.write(plaintext)

    # write the key to file
    perm = ''
    ab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(26):
        perm = perm + ab[i] + ' ' + key[i] + '\n'

    with open("perm.txt", "w") as perm_file:
        perm_file.write(perm)

    running_time = end_time - start_time
    print("Running time: {:.4f} seconds".format(running_time))

    x1 = range(len(avg_fitness_list))
    x2 = range(len(max_fitness_list))
    # Plot the values
    plt.plot(x1, avg_fitness_list, color='red', label='average')

    plt.plot(x2, max_fitness_list, color='blue', label='highest')

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Average Fitness Per Generation')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()

