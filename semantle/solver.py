from semantle.semantle import Semantle
from gensim.models import KeyedVectors
import random


class Solver:
    def __init__(self, semantle: Semantle):
        self.semantle = semantle
        self.wv: KeyedVectors = (
            semantle.model
        )  # Google News model is already a KeyedVectors object
        self.similarity_score: float  # Will be set by get_similarity_score

    def get_similarity_score(self, guessed_word: str) -> None:
        """
        Assigns the similarity score of a given guess to a variable.
        """
        score = self.semantle.check_guess(guessed_word)
        if score is None:
            raise ValueError(f"Word '{guessed_word}' not found in vocabulary")
        self.similarity_score = score

    def solve_game(self, tolerance_decimals=5):
        """
        A solver that can guess the Semantle word of the day.
        Starts with all potential words and narrows them down with each guess, filtering out any word that doesn't match the 
        target distance.
        
        Args:
            tolerance_decimals (int): The number of decimal places to use for the tolerance within which the target distance must be.
        """
        
        # Keep a list of potential words that might be the Word of the Day. 
        # Starts off with all possible words, and whittles it down.
        potential_words = list(self.wv.index_to_key)
        
        guess_num = 0
        
        # When the potential words is only 1 word long, we have our answer
        while len(potential_words) > 1:
            guess = random.choice(potential_words)
            print(f"\nGuess: {guess}")
            
            # Get the similarity score
            try:
                self.get_similarity_score(guess)
                similarity_of_current_guess = self.similarity_score
                
            except ValueError:
                # If the word isn't in vocabulary, remove it and try again
                potential_words.remove(guess)
                continue
            
            # Switch to cosine distance so we can use distances function
            target_dist = 1 - similarity_of_current_guess
            
            # Calculate the distances of the guess to all other words that are still potential words
            distances = self.wv.distances(guess, other_words=potential_words)
            tolerance = 1.0 * 10**(-tolerance_decimals)
            temp_potential_words = []
            print(f"Searching through {len(potential_words)} potential words, looking for words with distance {target_dist} +- {tolerance}")
            
            # Now we know the cosine distance from the guess to the target word, we can filter out any word that doesn't have the same 
            # distance from our guess. Only the remaining words are kept as potential target words.
            for i in range(len(potential_words)):
                w = potential_words[i]
                d = distances[i]
                if abs(d - target_dist) < tolerance:
                    print(f"Found a match: {w} with distance {d}")
                    temp_potential_words.append(w)
            
            potential_words = temp_potential_words
            guess_num += 1
        
        if len(potential_words) == 1:
            print(f"Answer is {potential_words[0]} found after {guess_num} attempt(s)")
            return potential_words[0]
        else:
            raise ValueError("No solution found.")
