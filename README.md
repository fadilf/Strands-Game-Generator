# Strands-Game-Generator
Automatically generate a game of Strands by NYT based on a theme of your choice.
Play the game here: [https://fadilfaizal.pythonanywhere.com/](https://fadilfaizal.pythonanywhere.com/)

## Instructions to run locally
- Clone this repository
- Set up a Python virtual environment and activate it
- Install the dependencies in `requirements.txt`
- Add a `.env` file and set `API_TOKEN` to be your HuggingFace Inference API token
- Run `flask --app app run -p 3000`
- Open `http://127.0.0.1:3000` in your browser and enjoy!