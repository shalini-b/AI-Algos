from heapq import heappush, heappop


class Node(object):

    def __init__(self, val):
        self.val = val
        # TODO: check if this is correct
        self.g_val = 99999
        self.h_val = 99999
        # node already evaluated
        self.in_closed = False
        # discovered node but not evaluated yet
        self.in_open = False
        # Maintain neighbour nodes
        self.neighbours = []

    def __hash__(self):
        return self.val

    def __eq__(self, other):
        return type(other) is type(self) and hash(other) == hash(self)

    # NOTE: __ne__ not implemented as python 3 takes care of it

    def __lt__(self, other):
        return self.g_val + self.h_val < other.g_val + other.h_val

    def manhattan_dist(self, goal):
        # find manhattan distance from self node to goal node
        cur_x, cur_y = self.val//card, self.val % card
        goal_x, goal_y = goal.val//card, goal.val % card
        return max(abs(cur_x - goal_x), abs(cur_y - goal_y))


class AStar(object):
    """ Astar algorithm """

    def __init__(self, src_node, goal_node):
        self.source = src_node
        self.goal = goal_node
        self.heuristic = 'manhattan_dist'

    @staticmethod
    def heap_push(sopen, node):
        heappush(sopen, node)
        node.in_open = True

    @staticmethod
    def heap_pop(sopen):
        node = heappop(sopen)
        node.in_open = False
        return node

    def rearrange(self):
        # The list of currently discovered nodes that are not evaluated yet.
        sopen = []

        # initialise source_tile
        self.source.g_val = 0
        # get callback function and evaluate h_val
        self.source.h_val = getattr(self.source, self.heuristic)(self.goal)
        self.heap_push(sopen, self.source)

        while sopen:
            # get min node from open list
            current = self.heap_pop(sopen)

            # if goal reached
            if current == self.goal:
                return current.g_val + current.h_val

            # Mark as closed
            current.in_closed = True

            # find neighbours of current
            for new_node in current.neighbours:
                # do not alter if it is already evaluated
                if new_node.in_closed:
                    continue

                tmp_g_val = current.g_val + 1
                if tmp_g_val >= new_node.g_val:
                    continue

                new_node.g_val = tmp_g_val
                # if it has not yet been evaluated
                if new_node not in sopen:
                    new_node.h_val = getattr(new_node,
                                             self.heuristic)(self.goal)
                    self.heap_push(sopen, new_node)

        return -1


if __name__ == '__main__':
    # TODO: Given a list of edges
    edges = [(0, 4), (1, 2), (2, 5), (0, 1), (4, 8), (5, 8)]

    # Cardinality
    card = 3

    # List of all nodes
    nodes_list = []
    for i in range(pow(card, 2)):
        nodes_list.append(Node(i))

    for node_val, neighbour_val in edges:
        nodes_list[node_val].neighbours.append(nodes_list[neighbour_val])

    astar = AStar(nodes_list[0], nodes_list[-1])
    output = astar.rearrange()
    if output == -1:
        raise Exception('Could not find path. Please try again')
    else:
        print(output)
