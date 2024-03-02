import io
import time
import numpy as np


def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    num_words, vec_size = map(int, fin.readline().split())

    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data


#######################################################
# These 3 functions calculate the cosine similarity for
# any 2 word vectors
def dot_product(vec_a, vec_b):
    dot_prod = 0.0
    for i in range(len(vec_a)):
        dot_prod += vec_a[i] * vec_b[i]

    return dot_prod


import math


def magnitude(vector):
    return math.sqrt(dot_product(vector, vector))


# The entry point function
def cosine_similarity(vec_a, vec_b):
    dot_prod = dot_product(vec_a, vec_b)
    magnitude_a = magnitude(vec_a)
    magnitude_b = magnitude(vec_b)

    return dot_prod / (magnitude_a * magnitude_b)


def calc_match(input_array, dictionary):
    vectors = []

    # Retrieves the vectors for the input words using the word as the key
    for word in input_array:
        vectors.append(dictionary[word])

    # Vector calculations
    vector4 = np.subtract(vectors[1], vectors[0])
    vector5 = np.add(vector4, vectors[2])

    probs = []

    # This loop calculates the cosine similarity for all words in dictionary except the
    # except for the word in question
    for i in dictionary:
        if input_array[-1] != i:
            probs.append([cosine_similarity(vector5, dictionary[i]), i])

    cosines = {}
    # Stores the cosine values and word in a new dictionary called cosines
    for item in probs:
        cosines[item[0]] = list(map(str, item[1:]))
    sorted_dict = dict(sorted(cosines.items()))  # Sorts the dictionary

    print(
        f"╔════════════════════════════════════════════════════════════╗\n║ {input_array[0]} is to {input_array[1]} as {input_array[2]} is to {list(sorted_dict.values())[-1][0]}\n╚════════════════════════════════════════════════════════════╝"
    )

    print("""╔═══════════════════════╤═══════════════════════════════╗""")
    print(
        """║PREDICTION VALUE\t│\tWORD\t\t\t║\n╚═══════════════════════╪═══════════════════════════════╝"""
    )

    # Print top 20 similarities
    for i in reversed(range(-20, 0)):
        key = list(sorted_dict.keys())[i]
        value = sorted_dict[key][0]

        print(f"│ {format(key,'.16f')}\t│\t{value}")

    # return list(sorted_dict.values())[-2][0]


def start():
    begin = True
    print(
        "\n╭─────────────────────────╮\n│  Word vector Loading... │\n╰─────────────────────────╯"
    )

    start_time = time.strftime("%H:%M:%S", time.localtime())
    word_vectors = load_vectors("FastText100K.txt")
    end_time = time.strftime("%H:%M:%S", time.localtime())
    print(
        f"╒═══════════════════════╕\n│Start time: {start_time}\t│\n│End time: {end_time}\t│\n╰───────────────────────╯"
    )
    print(
        "\t╭─────────────────────────╮\n\t│  Word vector loaded...  │\n\t╰─────────────────────────╯"
    )

    while begin == True:
        userInput = (
            input("\nEnter 3 analogy word tokens or <ENTER> to exit: ").strip().split()
        )

        dictionary_check = []

        for item in userInput:
            if item not in word_vectors and len(userInput) != 0:
                dictionary_check.append(item)

        if len(dictionary_check) == 0:
            if len(userInput) == 0:
                print(
                    "╭───────────────────────────────╮\n│   EXITING APPLICATION!!!\t│\n╰───────────────────────────────╯"
                )
                begin = False
            elif len(userInput) > 0 and len(userInput) != 3:
                print("\nYou must enter only 3 single word analogy words")
            else:
                print(
                    "\n╭──────────────────────╮\n│Processing analogy... │\n╰──────────────────────╯"
                )
                analogy = calc_match(userInput, word_vectors)
        else:
            print(
                f"\n{dictionary_check} word(s) not in dictionary!! Perhaps try something else"
            )


if __name__ == "__main__":
    start()
