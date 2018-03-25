import sys
from copy import deepcopy
from operator import itemgetter

moves = {'U': [-1, 0], 'R': [0, 1], 'D': [1, 0], 'L': [0, -1]}


class Node(object):

    def __init__(self,
                 matrix,
                 parent=None,
                 g_val=0,
                 h_val=0,
                 card=None,
                 dir_from_parent=''):
        self.matrix = matrix
        self.card = card or len(matrix)
        self.parent = parent
        self.g_val = g_val
        self.h_val = h_val
        self.f_val = 0
        self.dir = dir_from_parent

    def __hash__(self):
        return self.hash_matrix()

    def __eq__(self, other):
        return type(other) is type(self) and self.matrix == other.matrix

    def hash_matrix(self):
        wsum = 0
        fact = 1
        for i in range(self.card):
            for j in range(self.card):
                wsum += self.matrix[i][j] * fact
                fact *= 5
        return wsum

    # NOTE: __ne__ not implemented as python 3 takes care of it

    @staticmethod
    def find(num, matrix):
        # find index of given value in goal matrix
        for i, row in enumerate(matrix):
            try:
                j = row.index(num)
            except ValueError:
                continue
            return i, j

        return -1, -1

    def manhattan_dist(self, goal):
        # find manhattan distance from self node to goal node
        _sum = 0
        for i, row in enumerate(self.matrix):
            for j, tile in enumerate(row):
                if tile == 0:
                    continue
                i1, j1 = self.find(tile, goal.matrix)
                _sum += abs(i - i1) + abs(j - j1)

        return _sum

    def misplaced_tiles(self, goal):
        _sum = 0
        for i, row in enumerate(self.matrix):
            for j, tile in enumerate(row):
                if tile == 0:
                    continue
                _sum += 1 if self.matrix[i][j] != goal.matrix[i][j] else 0

        return _sum

    def find_gap(self):
        # find index of 0 (gap) in cur matrix
        return self.find(0, self.matrix)

    def is_position_legal(self, x, y):
        # find if position is legal in matrix
        return (x >= 0) and (x < self.card) and (y >= 0) and (y < self.card)

    def possible_moves(self, x, y):
        # get all possible moves
        global moves
        for dir, mv in moves.items():
            x2, y2 = self.next_pos(x, y, mv)
            if self.is_position_legal(x2, y2):
                yield dir, (x2, y2)

    @staticmethod
    def next_pos(x, y, move):
        return x + move[0], y + move[1]

    def neighbours(self):
        cur_matrix = self.matrix
        # find the gap
        x, y = self.find_gap()
        # formulate the next legal move
        for _dir, coord in self.possible_moves(x, y):
            # create a new node
            x2, y2 = coord
            new_node = Node(
                deepcopy(cur_matrix),
                card=self.card,
                parent=self,
                g_val=self.g_val + 1,
                dir_from_parent=_dir)
            # swap tiles for the new node
            new_node.matrix[x][y], new_node.matrix[x2][y2] = new_node.matrix[
                x2][y2], new_node.matrix[x][y]
            yield new_node


class AStar(object):
    """ Astar algorithm """

    def __init__(self, src_node, goal_node, card, heuristic):
        self.source = src_node
        self.goal = goal_node
        self.card = card
        self.heuristic = heuristic

    @staticmethod
    def get_min_node(node_list):
        return min(
            map(lambda x: (x, x.g_val + x.h_val), node_list),
            key=itemgetter(1))[0]

    def rearrange(self):
        # The dict of nodes already evaluated
        # dict of hashed_matrix & node
        closed = dict()

        # The dict of currently discovered nodes that are not evaluated yet.
        # dict of hashed_matrix & node
        sopen = dict()

        # initialise source_tile
        self.source.g_val = 0
        # get callback function
        self.source.h_val = getattr(self.source, self.heuristic)(self.goal)
        sopen.update({self.source.hash_matrix(): self.source})

        while sopen:
            # get min node from open list
            current = self.get_min_node(sopen.values())
            # add to closed
            closed.update({current.hash_matrix(): current})

            # if goal reached
            if current == self.goal:
                rev_path = []
                while current.parent:
                    rev_path.append(current.dir)
                    current = current.parent
                rev_path.reverse()
                return ','.join(rev_path)

            # remove current from open
            del (sopen[current.hash_matrix()])

            # find neighbours of current
            for new_node in current.neighbours():
                hash_new_node = new_node.hash_matrix()
                # do not alter if it is already evaluated
                if hash_new_node in closed:
                    continue

                # if it has not yet been evaluated
                if hash_new_node not in sopen:
                    new_node.h_val = getattr(new_node,
                                             self.heuristic)(self.goal)
                    sopen.update({hash_new_node: new_node})
                # if it has already been evaluated, then check for g_val
                elif hash_new_node in sopen and sopen[hash_new_node].g_val > new_node.g_val:
                    sopen.update({hash_new_node: new_node})

        return ''


class IDAStar(object):

    def __init__(self, src_node, goal_node, heuristic):
        self.source = src_node
        self.goal = goal_node
        self.heuristic = heuristic

    def rearrange(self):
        # get callback function
        bound = getattr(self.source, self.heuristic)(self.goal)
        path = [self.source]
        while True:
            t = self.search(path, bound)
            if t == True:
                return ','.join(node.dir for node in path[1:])
            elif t == float('inf'):
                return ''
            bound = t

    def search(self, path, bound):
        current = path[-1]

        # get callback function value
        current.h_val = getattr(current, self.heuristic)(self.goal)
        current.f_val = current.g_val + current.h_val
        if current.f_val > bound:
            return current.f_val

        if current == self.goal:
            return True

        _min = float('inf')
        for new_node in current.neighbours():
            if new_node not in path:
                path.append(new_node)
                t = self.search(path, bound)
                if t == True:
                    return True
                if t < _min:
                    _min = t
                path.pop()

        return _min


def main():
    if len(sys.argv) == 6:
        algo = int(sys.argv[1])
        card = int(sys.argv[2])
        heuristic = int(sys.argv[3])
        in_file = sys.argv[4]
        out_file = sys.argv[5]
    else:
        print("Invalid Input, please try again")
        return

    source = []
    with open(in_file) as f:
        for i in f.readlines():
            source.append(
                list(
                    map(lambda x: int(x) if x not in ['\n', ''] else 0,
                        i.split(','))))

    if heuristic == 1:
        callback = 'manhattan_dist'
    elif heuristic == 2:
        callback = 'misplaced_tiles'
    else:
        print("Invalid Input, ensure value of H is 1 or 2")
        return

    if card == 3:
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    elif card == 4:
        goal = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
    else:
        print("Invalid Input, ensure value of N is 3 or 4")
        return

    output = ''
    if algo == 1:
        astar = AStar(
            Node(source, card=card), Node(goal, card=card), card, callback)
        output = astar.rearrange()
    elif algo == 2:
        idastar = IDAStar(
            Node(source, card=card), Node(goal, card=card), callback)
        output = idastar.rearrange()
    else:
        print("Invalid Input, ensure algorithm is 1 or 2")
        return

    with open(out_file, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    main()
