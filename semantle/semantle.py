import time
import os
import gensim
from gensim import utils
import gensim.downloader as api
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import csv
import random
import argparse


class MyCorpus:
    '''An iterator that yields sentences (lists of str).'''

    def __iter__(self):
        corpus_path = os.path.relpath('./data/simpsons_script_lines.csv')
        with open(corpus_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter = ',', quotechar = '"')
            for row in reader:
                yield(row['normalized_text'].split())

class Semantle:

    def __init__(self):
        # initialize model
        # TODO: Figure out how to use GoogleNews300 model
        # When loaded using gensim.downloader.load('word2vec-google-news-300') the model we get out is a KeyedVectors object, which doesn't work the same as a Word2Vec object
        self.use_local = True
        if self.use_local:
            print("Using Simpsons")
            self.corpus_name = "simpsons-script" # To start
            self.sentences = MyCorpus()
            self.model = gensim.models.Word2Vec(sentences = self.sentences, vector_size=300, min_count=5)
            print(type(self.model))
        self.word_of_the_day = 'school'
        self.guesses_dict = {}
        self.guesses_in_order = []
        self.endgame = False
    
    def player_guess(self):
        '''
        Handles the user input for a guess.
        '''
        guess = input("Guess: ")
        cleaned_guess = guess.lower()
        return cleaned_guess

    def take_turn(self):
        '''
        The core turn structure of the game.
        '''
        taking_guess = True
        while taking_guess:     
            current_guess = self.player_guess()
            try:
                similarity_of_current_guess = self.model.wv.similarity(current_guess, self.word_of_the_day)
                taking_guess = False
            except:
                taking_guess = True
        self.guesses_dict[current_guess] = similarity_of_current_guess
        
        self.guesses_in_order.append(current_guess)
        print(f"Guess: {current_guess}, Similarity: {similarity_of_current_guess*100:.2f}")
        self.update_game_state(current_guess)
    
    def play_game(self):
        while not self.endgame:
            self.take_turn()
            
    def update_game_state(self, current_guess):
        self.endgame = current_guess == self.word_of_the_day
        
    def solve_game(self, tolerance_decimals=6):
        ''' a solver that can guess the word of the day.
        '''
        #TODO: right now this gets the distance to a Word of the day given in the class- we can adapt this to work where 
        # you input provided distances (e.g. with real semantle)
        
        # keep a list of potential words that might be the Word of the Day. Start off with all possible words, and whittle it down.
        potential_words = list(self.model.wv.index_to_key)
        
        guess_num = 0

        # when the potential words is only 1 word long, we should have our answer!

        while len(potential_words) >1:

            guess = random.choice(potential_words)
            print(f"\nGuess: {guess}")
            similarity_of_current_guess = self.model.wv.similarity(guess, self.word_of_the_day)
            
            # Switch to cosine distance so we can use distances function
            target_dist = 1 - similarity_of_current_guess
            
            # Calculate the distances of the guess to all other words that are still potential words
            distances = self.model.wv.distances(guess, other_words=potential_words)     
            tolerance = 1.0*10**(-tolerance_decimals)
            temp_potential_words = []
            print(f"Searching through {len(potential_words)} potential words, looking for words with distance {target_dist} +- {tolerance}")
            
            # Cycle through distances, keep the potential word if the distance matches.
            for i in range(len(potential_words)):
                w = potential_words[i]
                d = distances[i]
                if abs(d - target_dist) < tolerance:
                    print(f"Found a match: {w} with distance {d}")
                    temp_potential_words.append(w)
            
            potential_words = temp_potential_words
            guess_num += 1
        

        print(f"Answer is {potential_words[0]} found after {guess_num} attempt(s)")
                
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['play', 'solve'], help="Game mode, either 'play to guess the word of the day, or solve to find the word of the day using a solver.")
    args=parser.parse_args()
    
    if args.mode:
        mode = args.mode
    else: # if a mode isn't provided, get it from user input
        while True:
            
            mode = input("Enter mode (play/solve): ").strip().lower()
            if mode in ['play', 'solve']:
                break
            else:
                print("Invalid mode. Please enter 'play' or 'solve'.")
    print(f"\nPlaying in {mode} mode")


    print("\nLoading...")
    ct = time.time()
    semantle = Semantle()
    ft = time.time()
    print(f"Loaded in {ft - ct}")
    
    print('-------------------------')
    print('Starting game')
    
    if mode == 'solve':
        semantle.solve_game(tolerance_decimals = 6)
    elif mode == 'play':
        semantle.play_game()
                
                
if __name__ == "__main__":
    main() 
    

    

            


