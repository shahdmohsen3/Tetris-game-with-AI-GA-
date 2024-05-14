import random, time, pygame, sys
from pygame.locals import *
import tetris_base as game
import pygame as pg
import time
import random
random.seed(25)


def find_Best_Move(board, piece, weights):
    x_access = 0
    rotation = 0
    final_score = -100000

    n_holes, n_block = game.calc_initial_move_info(board)
    for r in range(len(game.PIECES[piece['shape']])):
        # Iterate through every possible rotation
        for x in range(-2, game.BOARDWIDTH - 2):
            # Iterate through every possible position
            inof = game.calc_move_info(board, piece, x, r, \
                                                n_holes, \
                                                n_block)

            if (inof[0]):  #valid info
                temp = 0
                for i in range(1, len(inof)):
                    temp += weights[i - 1] * inof[i]

                # Update best movement
                if (temp > final_score):
                    final_score = temp
                    x_access = x
                    rotation = r


        piece['y'] = -2

    piece['x'] = x_access
    piece['rotation'] = rotation

    return x_access, rotation


def run_game_with_ai(weights):
    game.FPS = int(500000)
    game.main()
    max_score=10000
    board = game.get_blank_board()
    Ftime= time.time()
    score = 0
    level_num, fall_freq = game.calc_level_and_fall_freq(score)
    falling_piece= None
    next_piece= game.get_new_piece()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("exit")
                exit()

        if falling_piece == None:
            falling_piece = next_piece
            next_piece    = game.get_new_piece()

            find_Best_Move(board, falling_piece, weights)  # Changed here

            score           += 1
            Ftime = time.time()
            level_num=level_num+1
            if (not game.is_valid_position(board, falling_piece)):
                break

        if time.time() - Ftime > fall_freq:
            if (not game.is_valid_position(board, falling_piece, adj_Y=1)):
                game.add_to_board(board, falling_piece)
                num_removed_lines = game.remove_complete_lines(board)
                if(num_removed_lines == 1):
                    score += 40
                elif (num_removed_lines == 2):
                    score += 120
                elif (num_removed_lines == 3):
                    score += 300
                elif (num_removed_lines == 4):
                    score += 1200
                falling_piece = None
            else:
                falling_piece['y'] += 1
                Ftime = time.time()

        game.Show(board,score,level_num,next_piece ,falling_piece)

        # Stop condition
        if (score > max_score):
            print("You Won!!!")
            break
    # print("Score: ",score)
    #
    return score

if __name__ == "__main__":
    optimal_weights = [-0.6212072122673682, -0.6781350133025787, -0.13852429009525494, -0.8293786110462142,
0.12615561588634705, -0.049539930847982305, 0.11916512916845212]

    run_game_with_ai(optimal_weights)
