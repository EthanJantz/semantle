#!/usr/bin/env python3
"""
Main entry point for the Semantle solver application.
"""

from semantle.semantle import Semantle
from semantle.solver import Solver
import argparse

def main():
    """
    Run the Semantle solver to find today's word.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['play', 'solve'], help="Game mode, either 'play to guess the word of the day, or solve to find the word of the day using a solver.")
    args=parser.parse_args()
    
    if args.mode:
        mode=args.mode 
    else: # If a mode isn't provided, get it from user input
        while True:
            mode = input("Enter mode (play/solve): ").strip().lower()
            if mode in ['play', 'solve']:
                break
            else:
                print("Invalid mode. Please enter 'play' or solve. ")
            print(f"\nPlaying in {mode}")
        
    
    print("Initializing Semantle...")
    semantle = Semantle()

    if mode == "play":
        print("Play Game")
        semantle.play_game()
        
    elif mode == "solve":
        print("Creating solver...")
        solver = Solver(semantle)
    
        print("Starting solver...")
        solution = solver.solve_game()
        print(f"\n{solution} is the solution to today's Semantle!")

if __name__ == "__main__":
    main()
