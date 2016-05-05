import copy
import sys
__author__ = 'Sharath_GM'


NODE_INDICES = []
infinity = 999999

LABEL_MATRIX = [[0 for x in range(5)] for x in range(5)]
LABEL_MATRIX[-1][-1] = "root"
LETTER_LIST = ['A', 'B', 'C', 'D', 'E']
FILE_STR = ""


class Board:

    NONE = '*'

    def __init__(self, initial_state, initial_values, start_player):
        self.empty_board = [['*' for x in range(5)] for x in range(5)]
        self.state_matrix = initial_state
        self.value_matrix = initial_values
        self.start_player = start_player
        self.eval_function_total = 0
        self.final_node = [-1, -1]

    def move(self, player_to_move, index1, index2):
        self.state_matrix[index1][index2] = player_to_move

    def get_candidate_moves(self):
        outlist = []
        for i in range(0, 5):
            for j in range(0, 5):
                if self.state_matrix[i][j] == Board.NONE:
                    outlist.append([i,j])
        return outlist

    def proper_print(self):
        matrix_str = ''
        for i in range(0, 5):
            for j in range(0, 5):
                matrix_str += (self.state_matrix[i][j]).rstrip('\r\n')
            if not i == 4:
                matrix_str += '\r\n'
        return matrix_str

    def proper_print_simulate(self):
        matrix_str = ''
        for i in range(0, 5):
            for j in range(0, 5):
                matrix_str += (self.state_matrix[i][j]).rstrip('\r\n')
            if self.is_full() and i == 4:
                return matrix_str
            matrix_str += '\r\n'
        return matrix_str

    def compute_eval_total(self):
        player1_total = 0
        player2_total = 0
        for i in range(0, 5):
            for j in range(0, 5):
                if self.state_matrix[i][j] == self.start_player:
                    player1_total += self.value_matrix[i][j]
                elif self.state_matrix[i][j] == get_opponent(self.start_player):
                    player2_total += self.value_matrix[i][j]
        self.eval_function_total = player1_total - player2_total
        return player1_total - player2_total

    def is_full(self):
        for i in range(0, 5):
            for j in range(0, 5):
                if self.state_matrix[i][j] == Board.NONE:
                    return False
        return True

    def set_game_player(self, this_player):
        self.start_player = this_player

    def set_indices(self, node):
        self.final_node = node


def str_name(value):
    if value == infinity:
        return "Infinity"
    if value == -infinity:
        return "-Infinity"

    return str(value)


def get_neighbours_to_conquer(i, j, this_player, present_board):
    this_opponent = get_opponent(this_player)
    node_list = []
    raid = "false"
    for x in range(max(0, i - 1), min(5, i + 2)):
        for y in range(max(0, j - 1), min(5, j + 2)):
            if (i, j) == (x, y) or (i != x and j != y):
                continue
            if present_board.state_matrix[x][y] == this_player:
                raid = "true"
                break
    if raid == "true":
        for x in range(max(0, i - 1), min(5, i + 2)):
            for y in range(max(0, j - 1), min(5, j + 2)):
                if (i, j) == (x, y) or (i != x and j != y):
                    continue
                if present_board.state_matrix[x][y] == this_opponent:
                    node_list.append([x, y])
    return node_list


def get_opponent(this_player):
    this_opponent = 'O'
    if this_player == 'O':
        this_opponent = 'X'
    return this_opponent


def compute_score(present_board):
    return present_board.compute_eval_total()


def minimax(curr_board, max_depth, curr_depth, p, q):
    global NODE_INDICES, FILE_STR

    if curr_depth >= max_depth:
        temp_value = compute_score(curr_board)
        FILE_STR += str(LABEL_MATRIX[p][q]) + "," + str(curr_depth) + "," + str(temp_value) + "\n"
        return compute_score(curr_board)
    if curr_depth % 2 == 0:
        curr_player = curr_board.start_player
    else:
        curr_player = get_opponent(curr_board.start_player)

    scores = []
    moves = []

    eval_score = -infinity
    temporary_list = [-infinity]

    move_list = curr_board.get_candidate_moves()

    for node in move_list:
        new_board = copy.deepcopy(curr_board)
        new_board.move(curr_player, node[0], node[1])
        for neighbour_node in get_neighbours_to_conquer(node[0], node[1], curr_player, new_board):
            new_board.move(curr_player, neighbour_node[0], neighbour_node[1])
        #print new_board.proper_print()
        if curr_depth == 0:
            FILE_STR += "root" + "," + str(curr_depth) + "," + str_name(max(temporary_list)) + "\n"
        elif curr_player == curr_board.start_player:
            FILE_STR += str(LABEL_MATRIX[p][q]) + "," + str(curr_depth) + "," + str_name(max(temporary_list)) + "\n"
        else:
            FILE_STR += str(LABEL_MATRIX[node[0]][node[1]]) + "," + str(curr_depth) + "," + str_name(min(temporary_list)) + "\n"
        eval_score = minimax(new_board, max_depth, curr_depth + 1, node[0], node[1])

        scores.append(eval_score)
        temporary_list.append(eval_score)
        moves.append(node)

    if curr_depth == 0:
        FILE_STR += "root" + "," + str(curr_depth) + "," + str(max(temporary_list)) + "\n"
    else:
        FILE_STR += str(LABEL_MATRIX[node[0]][node[1]]) + "," + str(curr_depth + 1) + "," + str(min(temporary_list)) + "\n"

    if curr_player == curr_board.start_player:
        i = 0
        largest = scores[0]
        max_score_index = 0
        for val in scores:
            if scores[i] > largest:
                largest = scores[i]
                max_score_index = i
            i += 1
        choice = moves[max_score_index]
        NODE_INDICES = choice
        curr_board.set_indices(choice)
        return scores[max_score_index]

    else:
        FILE_STR += str(LABEL_MATRIX[p][q]) + "," + str(curr_depth) + "," + str(min(scores)) + "\n"
        j = 0
        smallest = scores[0]
        min_score_index = 0
        for val in scores:
            if scores[j] < smallest:
                smallest = scores[j]
                min_score_index = j
            j += 1
        choice = moves[min_score_index]
        NODE_INDICES = choice
        curr_board.set_indices(choice)
        return scores[min_score_index]



def alphabeta(curr_board, max_depth, curr_depth):
    def max_value(curr_board, alpha, beta, max_depth, curr_depth, p, q):
        global NODE_INDICES, FILE_STR
        if curr_depth >= max_depth:
            return compute_score(curr_board)
        if curr_depth % 2 == 0:
            curr_player = curr_board.start_player
        else:
            curr_player = get_opponent(curr_board.start_player)
        eval_score = -infinity
        moves = []
        scores = []

        move_list = curr_board.get_candidate_moves()

        for node in move_list:

            new_board = copy.deepcopy(curr_board)
            new_board.move(curr_player, node[0], node[1])
            for neighbour_node in get_neighbours_to_conquer(node[0], node[1], curr_player, new_board):
                new_board.move(curr_player, neighbour_node[0], neighbour_node[1])
            #print new_board.proper_print()
            temp_eval = min_value(new_board, alpha, beta, max_depth, curr_depth + 1, node[0], node[1])
            eval_score = max(eval_score, temp_eval)
            FILE_STR += str(LABEL_MATRIX[node[0]][node[1]]) + "," + str(curr_depth + 1) + "," + str(temp_eval) + ',' + str_name(alpha) + ',' + str_name(beta) + "\n"
            moves.append(node)
            scores.append(eval_score)
            FILE_STR += str(LABEL_MATRIX[p][q]) + "," + str(curr_depth) + "," + str(min(scores)) + ',' + str_name(alpha) + ',' + str_name(beta) + "\n"
            max_value_index = scores.index(max(scores))
            NODE_INDICES = moves[max_value_index]
            curr_board.set_indices(moves[max_value_index])
            if eval_score >= beta:
                return eval_score
            alpha = max(alpha, eval_score)
        return eval_score

    def min_value(curr_board, alpha, beta, max_depth, curr_depth, p, q):
        global NODE_INDICES, FILE_STR
        if curr_depth >= max_depth:
            return compute_score(curr_board)
        if curr_depth % 2 == 0:
            curr_player = curr_board.start_player
        else:
            curr_player = get_opponent(curr_board.start_player)
        eval_score = infinity
        moves = []
        scores = []

        move_list = curr_board.get_candidate_moves()

        for node in move_list:
            new_board = copy.deepcopy(curr_board)
            new_board.move(curr_player, node[0], node[1])
            for neighbour_node in get_neighbours_to_conquer(node[0], node[1], curr_player, new_board):
                new_board.move(curr_player, neighbour_node[0], neighbour_node[1])
            #print new_board.proper_print()
            temp_eval = max_value(new_board, alpha, beta, max_depth, curr_depth + 1, node[0], node[1])
            eval_score = min(eval_score, temp_eval)
            FILE_STR += str(LABEL_MATRIX[node[0]][node[1]]) + "," + str(curr_depth + 1) + "," + str(temp_eval) + ',' + str_name(alpha) + ',' + str_name(beta) + "\n"
            moves.append(node)
            scores.append(eval_score)
            FILE_STR += str(LABEL_MATRIX[p][q]) + "," + str(curr_depth) + "," + str(min(scores)) + ',' + str_name(alpha) + ',' + str_name(beta) + "\n"
            min_value_index = scores.index(min(scores))
            NODE_INDICES = moves[min_value_index]
            if eval_score <= alpha:
                return eval_score
            beta = min(beta, eval_score)
        return eval_score

    val = max_value(curr_board, -infinity, +infinity, max_depth, curr_depth, -1, -1)
    return val


def do_greedy(curr_board):
    move_list = curr_board.get_candidate_moves()
    scores = []
    moves = []

    for node in move_list:
        new_board = copy.deepcopy(curr_board)
        new_board.move(game_player, node[0], node[1])
        for neighbour_node in get_neighbours_to_conquer(node[0], node[1], game_player, new_board):
            new_board.move(game_player, neighbour_node[0], neighbour_node[1])
        #print new_board.proper_print()
        scores.append(compute_score(new_board))
        moves.append(node)

    if game_player == curr_board.start_player:
        i = 0
        largest = scores[0]
        max_score_index = 0
        for val in scores:
            if scores[i] > largest:
                largest = scores[i]
                max_score_index = i
            i += 1
        choice = moves[max_score_index]
        curr_board.set_indices(choice)


def play_game(game_depth, algo_choice, present_board):
    if algo_choice == 1:
        do_greedy(present_board)

    elif algo_choice == 2:
        minimax(present_board, game_depth, 0, -1, -1)

    elif algo_choice == 3:
        alphabeta(present_board, game_depth, 0)

#MAIN FUNCTION

for i in range(0, 5):
    for j in range(0, 5):
        LABEL_MATRIX[i][j] = LETTER_LIST[j] + str(i+1)

ValueMatrix = [[0 for x in range(5)] for x in range(5)]
StateMatrix = [[0 for x in range(5)] for x in range(5)]

f = open('input.txt', 'r')

algorithmChoice = int(f.readline())

if algorithmChoice != 4:
    game_player = f.readline().rstrip()
    game_opponent = get_opponent(game_player)

    cutOffDepth = int(f.readline().rstrip())

    i = 0

    # Populate Heuristics
    for line in f:
        line = line.rstrip()
        valueList = line.split(' ')
        j = 0
        for value in valueList:
            ValueMatrix[i][j] = int(value.replace('\r\n', ''))
            j += 1
        i += 1
        if i == 5:
            break

    # Populate Given State of Board
    i = 0
    for line in f:
        line = line.rstrip()
        j = 0
        line = line[:5]
        for character in line:
            if character != '\n':
                StateMatrix[i][j] = character.rstrip('\r\n')
            j += 1
        i += 1

f.close()

#Greedy Best First Search

if algorithmChoice == 1:

    board = Board(StateMatrix, ValueMatrix, game_player)

    do_greedy(board)

    f = open('next_state.txt', 'w')
    board.move(game_player, board.final_node[0] , board.final_node[1])
    for neighbour_node in get_neighbours_to_conquer(board.final_node[0], board.final_node[1], game_player, board):
            board.move(game_player, neighbour_node[0], neighbour_node[1])
    #print board.proper_print()
    f.write(board.proper_print())
    del board
    f.close()

#END OF GBFS


# Minimax

if algorithmChoice == 2:
    board2 = Board(StateMatrix, ValueMatrix, game_player)

    file_log = open('traverse_log.txt', 'w')
    file_log.write("Node,Depth,Value\n")

    final_value = minimax(board2, cutOffDepth, 0, -1, -1)

    file_log.write(FILE_STR)
    file_log.close()

    f = open('next_state.txt', 'w')
    board2.move(game_player, board2.final_node[0], board2.final_node[1])
    for neighbour_node in get_neighbours_to_conquer(board2.final_node[0], board2.final_node[1], game_player, board2):
        board2.move(game_player, neighbour_node[0], neighbour_node[1])
    #print board2.proper_print()
    #print "Minimax"

    f.write(board2.proper_print())
    del board2
    f.close()

# End of Minimax


if algorithmChoice == 3:
    board = Board(StateMatrix, ValueMatrix, game_player)

    file_log = open('traverse_log.txt', 'w')
    file_log.write("Node,Depth,Value,Alpha,Beta\n")

    final_value = alphabeta(board, cutOffDepth, 0)

    file_log.write(FILE_STR)
    file_log.close()

    f = open('next_state.txt', 'w')
    board.move(game_player, NODE_INDICES[0], NODE_INDICES[1])
    for neighbour_node in get_neighbours_to_conquer(NODE_INDICES[0], NODE_INDICES[1], game_player, board):
        board.move(game_player, neighbour_node[0], neighbour_node[1])
    #print board.proper_print()
    #print "alpha"
    f.write(board.proper_print())
    del board
    f.close()

if algorithmChoice == 4:
    f = open('input.txt', 'r')
    f.readline()
    game_player = f.readline().rstrip()
    player_algorithm = int(f.readline().rstrip())
    player_max_depth = int(f.readline().rstrip())
    game_opponent = f.readline().rstrip()
    opponent_algorithm = int(f.readline().rstrip())
    opponent_max_depth = int(f.readline().rstrip())

    # Populate Heuristics
    i = 0
    for line in f:
        line = line.rstrip()
        valueList = line.split(' ')
        j = 0
        for value in valueList:
            ValueMatrix[i][j] = int(value.replace('\n', ''))
            j += 1
        i += 1
        if i == 5:
            break

    # Populate Given State of Board
    i = 0
    for line in f:
        line = line.rstrip()
        j = 0
        line = line[:5]
        for character in line:
            StateMatrix[i][j] = character.replace('\r\n', '')
            j += 1
        i += 1
        if i == 5:
            break

    board = Board(StateMatrix, ValueMatrix, game_player)
    f = open('trace_state.txt', 'w')

    while not board.is_full():
        board.set_game_player(game_player)
        play_game(player_max_depth, player_algorithm, board)
        if player_algorithm == 3:
            board.move(game_player, NODE_INDICES[0], NODE_INDICES[1])
            for neighbour_node in get_neighbours_to_conquer(NODE_INDICES[0], NODE_INDICES[1], game_player, board):
                board.move(game_player, neighbour_node[0], neighbour_node[1])
        else:
            board.move(game_player, board.final_node[0], board.final_node[1])
            for neighbour_node in get_neighbours_to_conquer(board.final_node[0], board.final_node[1], game_player, board):
                board.move(game_player, neighbour_node[0], neighbour_node[1])
        #print board.proper_print_simulate()
        f.write(board.proper_print_simulate())

        if board.is_full():
            break

        board.set_game_player(game_opponent)
        play_game(opponent_max_depth, opponent_algorithm, board)
        if player_algorithm == 3:
            board.move(game_opponent, NODE_INDICES[0], NODE_INDICES[1])
            for neighbour_node in get_neighbours_to_conquer(NODE_INDICES[0], NODE_INDICES[1], game_opponent, board):
                board.move(game_opponent, neighbour_node[0], neighbour_node[1])
        else:
            board.move(game_opponent, board.final_node[0], board.final_node[1])
            for neighbour_node in get_neighbours_to_conquer(board.final_node[0], board.final_node[1], game_opponent, board):
                board.move(game_opponent, neighbour_node[0], neighbour_node[1])

        print board.proper_print_simulate()
        f.write(board.proper_print_simulate())


