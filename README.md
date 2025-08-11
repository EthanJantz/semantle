# Semantle

This is a recreation of the game [Semantle](semantle.com) in Python and an accompanying solver. Semantle is a word guessing game where the user attempts to guess a word using the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of their most recent guess to the target word.

This implementation uses the Google News Word2Vec model with 300-dimensional vectors trained on billions of words, providing comprehensive semantic similarity calculations.

## The Solver

The original Semantle game provides information on the Word2Vec model used to power the game in its [FAQ](https://semantle.com/faq). With this information and the Google News Word2Vec model, it's possible to create a solver that efficiently finds the answer through an iterative filtering approach. `solver.py` implements this by starting with all potential target words as candidates, then makes a random guess from the candidates. After each guess, it takes the target cosine similarity that is returned and filters out any words from the potential target word list that doesn't have the same cosine similarity from the guess word (within a specified tolerance). This process repeats, progressively narrowing down the potential words until only the correct answer remains.

## Running the Application

The project uses the pre-trained Google News Word2Vec model, so no custom dataset is required. The model will be automatically downloaded on first run.

Set up your dependencies with `uv sync` and run the application with `uv run python main.py`. 

You will be able to then choose either `play` mode (to play the game yourself) or `solve` mode (to watch the solver find the answer automatically).