import sys
import time
from copy import deepcopy
from random import shuffle, choice
from collections import defaultdict


class DfsB(object):
    def __init__(self, num_vars, domain, adj_matrix):
        self.num_vars = num_vars
        self.adj_matrix = adj_matrix
        self.domain = domain
        self.vars = [i for i in range(num_vars)]
        self.assignment = {}
        self.recursions = 0

    def recurse_dfsb(self):
        # Increment number of steps
        self.recursions += 1
        # When assignment is complete
        if len(self.assignment) == self.num_vars:
            return self.assignment

        # Fetch first unassigned variable
        node = choice(list(set(self.vars) - set(self.assignment)))
        for i in range(self.domain):
            # Check for constraints
            if not self.is_compliant(node, i):
                continue

            # Make the selection
            self.assignment.update({node: i})
            res = self.recurse_dfsb()
            if res:
                return res

            # Backtrack from this assignment and try next value
            self.assignment.pop(node)

        return False

    def is_compliant(self, node, color):
        # Check if constrained nodes have same color assigned already
        return not any(map(lambda x: self.assignment.get(x) == color, self.adj_matrix[node]))


class DfsBPlus(object):
    def __init__(self, num_vars, domain, adj_matrix):
        self.num_vars = num_vars
        self.domain = domain
        self.vars = [i for i in range(num_vars)]
        # gives neighbours for each variable
        self.adj_matrix = adj_matrix
        # gives final assignment
        self.assignment = {}
        # Fetch statistics
        self.arc_pruning_steps = 0
        self.search_steps = 0

    def dfsb(self):
        # initialises var-domain scenario
        available_values = dict.fromkeys(self.vars, [i for i in range(self.domain)])
        return self.recurse_dfsb(available_values)

    def recurse_dfsb(self, cur_domain):
        self.search_steps += 1
        # When assignment is complete
        if len(self.assignment) == self.num_vars:
            return self.assignment

        # Select most constrained variable
        node = self.get_mcv(cur_domain)
        # Iterate over list of least constrained colors for chosen node
        for i in self.lcv(node, cur_domain):
            # Check for constraints
            if not self.is_compliant(node, i):
                continue

            # take backup of cur_domain
            next_domain = deepcopy(cur_domain)
            next_domain.update({node: [i]})

            # Check AC3 and prune domains
            if not self.check_arc_consistency(next_domain):
                continue

            # Make the selection
            self.assignment.update({node: i})

            # Go to next node
            res = self.recurse_dfsb(next_domain)
            if res:
                return res
            else:
                # Backtrack from this assignment and try next value
                self.assignment.pop(node)

        return False

    def check_arc_consistency(self, domain):
        # Check for AC3 arc consistency wrt to given assignment
        arcs = [(i, j) for i in self.vars for j in self.adj_matrix[i]]
        while arcs:
            n1, n2 = arcs.pop()
            if not self.inconsistent(n1, n2, domain):
                # if values are consistent, continue to next arc
                continue

            if not domain[n1]:
                # if there are no values left for n1
                return False

            # Change detected in domain of n1,
            # add back its neighbours to check
            for n3 in self.adj_matrix[n1]:
                if n3 != n2 and (n3, n1) not in arcs:
                    arcs.append((n3, n1))

        return True

    @staticmethod
    def inconsistent(n1, n2, domain):
        changed = False
        remove = []
        for n1c in domain[n1]:
            if not any(map(lambda n2c: n2c != n1c, domain[n2])):
                remove.append(n1c)
                changed = True

        if changed:
            domain[n1] = set(domain[n1]) - set(remove)

        return changed

    def get_mcv(self, cur_domain):
        # TODO: Degree heuristic can be used to break contention
        # Get differing MCV values if there is a contention
        shuffle(self.vars)
        # Select the variable which has least number of domain values
        return min(list(set(self.vars) - set(self.assignment)), key=lambda x: len(cur_domain[x]))

    def lcv(self, node, cur_domain):
        # Sort the color options in ascending order
        # of total possible domain values for other nodes
        return sorted(cur_domain[node], key=lambda x: self.num_possible(node, x, cur_domain), reverse=True)

    def num_possible(self, node, color, cur_domain):
        # Counts total number of domain values given a color assigned to a node
        cnt = 0

        for n in self.adj_matrix[node]:
            # Skip the assigned nodes
            if n in self.assignment:
                continue

            self.arc_pruning_steps += 1
            # Check for constraints
            cnt += len(list(set(cur_domain[n]) - {color}))

        return cnt

    def is_compliant(self, node, color):
        # Check if constrained nodes have same color assigned already
        return not any(map(lambda x: self.assignment.get(x) == color, self.adj_matrix[node]))


def run():
    # To find out time taken for running algo
    start = time.time()
    if len(sys.argv) == 4:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
        algo = int(sys.argv[3])
    else:
        print("Invalid Input, please try again")
        return

    with open(in_file) as f:
        num_var, num_const, domain = map(int, f.readline().split())
        constraints = f.readlines()

    adj_matrix = defaultdict(set)
    for c in constraints:
        cnst = list(map(int, c.split()))
        adj_matrix[cnst[0]].add(cnst[1])
        adj_matrix[cnst[1]].add(cnst[0])

    for n, nodes in adj_matrix.items():
        adj_matrix[n] = list(nodes)

    print(num_var, num_const, domain, adj_matrix)

    if algo == 0:
        dfsb = DfsB(num_var, domain, adj_matrix)
        res = dfsb.recurse_dfsb()
        # Number of recursions
        print(dfsb.recursions)
    elif algo == 1:
        dfsbp = DfsBPlus(num_var, domain, adj_matrix)
        res = dfsbp.dfsb()
        print(dfsbp.arc_pruning_steps + dfsbp.search_steps)
    else:
        print("Invalid Input for mode, please try again")
        return

    # Output results
    if isinstance(res, dict):
        print([i[1] for i in sorted(res.items())])
    else:
        print('No answer')

    with open(out_file, 'w') as f:
        if isinstance(res, dict):
            for i in sorted(res.items()):
                f.write(str(i[1]) + '\n')
        else:
            f.write('No answer \n')

    end = time.time()
    print('Time taken: ', end - start)

    # Verify results
    if isinstance(res, dict):
        verify_result(adj_matrix, res)


def verify_result(adj_matrix, res):
    for node, neighbours in adj_matrix.items():
        if any(map(lambda x: res[x] == res[node], neighbours)):
            print('Alert! Constraint violated for node {}'.format(node))

    print('Verification complete!')


if __name__ == '__main__':
    run()
