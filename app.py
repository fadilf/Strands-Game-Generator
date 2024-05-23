from flask import Flask, request
from generator import generator
import json

app = Flask(__name__)

with open("home.html") as f:
    home = f.read()

with open("template.html") as f:
    template = f.read()

@app.route("/")
def hello_world():
    theme = request.args.get('theme')
    if theme is None:
        return home
    else:
        if theme in ["RANDOM", ""]:
            grid, letter_coords, spangram_idx, query = generator()
        else:
            grid, letter_coords, spangram_idx, query = generator(theme)
        # grid = [['p', 'u', 'l', 't', 'a', 'g'],
        #         ['f', 'o', 'o', 's', 'e', 'e'],
        #         ['g', 'n', 'd', 'c', 'e', 'r'],
        #         ['s', 'i', 'e', 'r', 'o', 'y'],
        #         ['l', 'i', 'd', 's', 'c', 'n'],
        #         ['s', 'a', 'n', 'p', 'r', 'e'],
        #         ['r', 't', 'r', 's', 'a', 'l'],
        #         ['a', 'r', 'i', 'v', 'a', 's']]
        # letter_coords = [
        #     [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4)],
        #     [(1, 0), (1, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 5)],
        #     [(2, 5), (1, 4), (2, 3), (3, 4), (3, 3), (2, 2), (3, 1), (2, 1), (2, 0), (3, 0)],
        #     [(5, 0), (4, 0), (4, 1), (4, 2), (3, 2), (4, 3)],
        #     [(6, 1), (6, 2), (5, 1), (5, 2), (6, 3), (5, 3), (6, 4), (5, 4), (5, 5), (4, 5), (4, 4), (3, 5)],
        #     [(7, 0), (6, 0), (7, 1), (7, 2), (7, 3), (7, 4), (6, 5), (7, 5)]
        # ]
        # spangram_idx = 2
        # query = theme
        template_filled = template.replace("{GRID_HERE}", json.dumps(grid))
        template_filled = template_filled.replace("{LETTER_COORDS_HERE}", json.dumps(letter_coords))
        template_filled = template_filled.replace("{SPANGRAM_IDX_HERE}", json.dumps(spangram_idx))
        template_filled = template_filled.replace("{CHOSEN_THEME_HERE}", query)
        return template_filled