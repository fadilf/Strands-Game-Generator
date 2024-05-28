# from FlagEmbedding import FlagModel
import requests
import pickle
from scipy.spatial import KDTree
import random
import numpy as np
import itertools
from dotenv import dotenv_values

config = dotenv_values(".env")

def generate_n_cont_set(word, n):
    n_letters_word = set()
    for i in range(len(word) - n + 1):
        n_letters_word.add(word[i:i+n])
    return n_letters_word

def rotate_grid(grid):
    return [[tuple(item) for item in row] for row in np.rot90(np.array(grid))]

class Model:
    API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
    API_TOKEN = config["API_TOKEN"]
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    def encode(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return np.array(response.json(), dtype=np.float32)
model = Model()
# model = FlagModel('BAAI/bge-small-en-v1.5',
#                 query_instruction_for_retrieval="Generate a representation for this word for retrieving related words:",
#                 use_fp16=True)

with open("google-10000-english-no-swears.txt") as f:
    all_words = f.readlines()

words = []
for word in all_words:
    word_filtered = word.strip()
    if len(word_filtered) > 3:
        words.append(word_filtered)

lens = {}
for word in words:
    word_len = len(word)
    if word_len not in lens:
        lens[word_len] = 1
    else:
        lens[word_len] += 1

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

tree = KDTree(embeddings)

def generator(query:str=None):

    if query is None:
        query = random.choice(words)

    query_embedding = model.encode(query)

    # Get the 1000 most relevant words/phrases
    dd, ii = tree.query(query_embedding, k=1000)

    # Choose spangrammables
    spangrammable = []
    spangram_weights = []
    for weight, i in zip(dd, ii):
        candidate = words[i].strip()
        candidate_alpha_only = "".join([char for char in candidate if char.isalpha()])
        if 6 <= len(candidate_alpha_only) <= 13:
            spangrammable.append(candidate)
            spangram_weights.append(weight ** 60)

    spangram_weights = spangram_weights[:100]
    spangram_weights /= sum(spangram_weights)
    spangram = random.choices(spangrammable[:100], spangram_weights, k=1)[0]

    subembeddings = [embeddings[i] for i in ii]
    subwords = [words[i] for i in ii]
    subtree = KDTree(subembeddings)

    # Get words/phrases the most relevant to spangram
    ddd, iii = subtree.query(model.encode(spangram), k=250)

    candidates: list[str] = []
    n = 4
    candidate_n_cont_pool: set[str] = set()
    candidate_weights = {}
    for weight, i in zip(ddd, iii):
        word = subwords[i].strip().lower()
        word_set = generate_n_cont_set(word, n)
        unique = True
        for item in word_set:
            if item in candidate_n_cont_pool:
                unique = False
                break
        if unique:
            candidates.append(word)
            candidate_n_cont_pool.update(word_set)
            candidate_weights[word] = weight ** 60

    # Separate the candidates into plain words and phrases
    plain_words = []
    phrases = []
    for candidate in candidates[1:]:
        if len(candidate) <= 3:
            continue
        if candidate.isalpha():
            plain_words.append(candidate)
        else:
            phrases.append(candidate)

    plain_lengths = [len(plain_word) for plain_word in plain_words]

    spangram = "".join([char for char in spangram if char.isalpha()])

    budget = 48 - len(spangram)

    chosen = []
    plain_lengths_cpy = plain_lengths[:]
    while sum(chosen) != budget:
        remaining = budget - sum(chosen)
        if sum(chosen) < budget:
            if remaining in plain_lengths_cpy:
                chosen.append(remaining)
                break
            else:
                new_chosen = random.choice(plain_lengths_cpy)
                plain_lengths_cpy.remove(new_chosen)
                chosen.append(new_chosen)
        elif sum(chosen) > budget:
            if (-remaining) in chosen:
                chosen.remove(-remaining)
                break
            else:
                for i in range(random.choice([1,1,1,2,2,3])):
                    to_remove = random.choice(chosen)
                    chosen.remove(to_remove)
                    plain_lengths_cpy.append(to_remove)

    word_lens: dict[int, list[str]] = {}
    for word in plain_words:
        word_len = len(word)
        if word_len not in word_lens:
            word_lens[word_len] = [word]
        else:
            word_lens[word_len].append(word)

    chosen_words = [spangram]
    for length in chosen:
        weights = [candidate_weights[word] for word in word_lens[length]]
        weights /= sum(weights)
        chosen_word = random.choices(word_lens[length], weights, k=1)[0]
        word_lens[length].remove(chosen_word)
        chosen_words.append(chosen_word)

    if len(spangram) < 8:
        spangram_direction = "ltr"
    else:
        spangram_direction = random.choice(["ltr", "ttb"])

    grid = []
    for i in range(8):
        row = []
        for j in range(6):
            row.append((i,j))
        grid.append(row)

    if spangram_direction == "ttb":
        grid = rotate_grid(grid)
    coord_lst = grid.pop(0)
    row = 1
    while len(grid) > 0:
        if row % 2 == 0:
            coord_lst.extend(grid.pop(0))
        else:
            coord_lst.extend(reversed(grid.pop(0)))
        row += 1

    non_spangrams = chosen_words[:]
    non_spangrams.remove(spangram)
    valid_combos = []
    for n in range(len(non_spangrams) + 1):
        for combo in itertools.combinations(non_spangrams, n):
            prefix_len = sum([len(item) for item in combo])
            if spangram_direction == "ltr":
                dir_len = 6
            else:
                dir_len = 8
            row_prefix_len = prefix_len % dir_len
            if (row_prefix_len == 0) or ((row_prefix_len + len(spangram)) >= (2 * dir_len)):
                valid_combos.append(combo)

    chosen_combo = list(random.choice(valid_combos[1:-1]))
    random.shuffle(chosen_combo)
    for word in chosen_combo:
        non_spangrams.remove(word)
    random.shuffle(non_spangrams)
    chosen_words = chosen_combo + [spangram] + non_spangrams
    spangram_idx = chosen_words.index(spangram)


    coords: list[list[tuple[int,int]]] = []

    for word in chosen_words:
        word_coords = []
        for char in word:
            word_coords.append(coord_lst.pop(0))
        if random.choice([True, False]):
            coords.append(word_coords)
        else:
            coords.append(list(reversed(word_coords)))


    def get_word_letter_idx(coords, letter_coords, words):
        for word_idx, word_coords in enumerate(coords):
            if letter_coords in word_coords:
                letter_idx = word_coords.index(letter_coords)
                word = words[word_idx]
                letter = word[letter_idx]
                return letter_idx, letter, word_idx, word


    def check_word_continuity(word_coords:list[list[tuple[int,int]]]):
        for coords_a, coords_b in zip(word_coords[:-1], word_coords[1:]):
            if abs(coords_a[0] - coords_b[0]) > 1 or abs(coords_a[1] - coords_b[1]) > 1:
                return False
        return True


    def spangram_valid(word_coords:list[tuple[int,int]], direction: str):
        if direction == "ltr":
            return 0 in [coord[1] for coord in word_coords] and 5 in [coord[1] for coord in word_coords]
        else:
            return 0 in [coord[0] for coord in word_coords] and 7 in [coord[0] for coord in word_coords]


    def shuffle_grid(letter_coords, words, n):
        letter_coords_cpy = [word[:] for word in letter_coords]
        shuffles = 0

        while shuffles < n:
            a_coord = random.randint(0, 7), random.randint(0, 5)
            a_letter_idx, a_letter, a_word_idx, a_word = get_word_letter_idx(letter_coords_cpy, a_coord, words)

            b_candidate_coords = []
            for i in range(-1,2):
                row = a_coord[0] + i
                if row < 0 or row >= 8:
                    continue
                for j in range(-1,2):
                    col = a_coord[1] + j
                    if col < 0 or col >= 6 or a_coord == (row, col):
                        continue

                    b_candidate_coords.append((row, col))
            
            random.shuffle(b_candidate_coords)
            for b_coord in b_candidate_coords:
                b_letter_idx, b_letter, b_word_idx, b_word = get_word_letter_idx(letter_coords_cpy, b_coord, chosen_words)
                if a_word_idx == b_word_idx:
                    possible_word_coords = [item[:] for item in letter_coords_cpy[a_word_idx]]
                    possible_word_coords[a_letter_idx] = b_coord
                    possible_word_coords[b_letter_idx] = a_coord
                    if a_word_idx == spangram_idx and not spangram_valid(possible_word_coords, spangram_direction):
                        continue
                    if check_word_continuity(possible_word_coords):
                        letter_coords_cpy[a_word_idx] = possible_word_coords
                        # print(f"swapping: {a_coord, b_coord} in same word")
                        shuffles += 1
                        break
                else:
                    possible_a_word_coords = letter_coords_cpy[a_word_idx][:]
                    possible_a_word_coords[a_letter_idx] = b_coord

                    possible_b_word_coords = letter_coords_cpy[b_word_idx][:]
                    possible_b_word_coords[b_letter_idx] = a_coord

                    if a_word_idx == spangram_idx and not spangram_valid(possible_a_word_coords, spangram_direction):
                        continue

                    if b_word_idx == spangram_idx and not spangram_valid(possible_b_word_coords, spangram_direction):
                        continue

                    if check_word_continuity(possible_a_word_coords) and check_word_continuity(possible_b_word_coords):
                        letter_coords_cpy[a_word_idx] = possible_a_word_coords
                        letter_coords_cpy[b_word_idx] = possible_b_word_coords
                        # print(f"swapping: {a_coord, b_coord} in different words")
                        shuffles += 1
                        break
        return letter_coords_cpy

    new_letter_coords = shuffle_grid(coords, chosen_words, 100000)

    grid = [["" for col in range(6)] for row in range(8)]
    for word_coords, word in zip(new_letter_coords, chosen_words):
        for letter_coord, letter in zip(word_coords, word):
            grid[letter_coord[0]][letter_coord[1]] = letter
    
    return grid, new_letter_coords, spangram_idx, query