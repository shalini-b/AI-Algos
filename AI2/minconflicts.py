import sys
import time
from random import shuffle, randint, choice
from collections import defaultdict


class MinConflicts(object):
    def __init__(self, num_vars, domain, adj_matrix):
        self.num_vars = num_vars
        self.adj_matrix = adj_matrix
        self.domain = domain
        self.vars = [i for i in range(num_vars)]
        self.domain_vals = [i for i in range(self.domain)]
        self.assignment = {}
        self.steps_req = 0

    def solve_csp(self, max_steps, prob):
        # Complete formulation of problem using greedy approach
        self.make_initial_assignment()

        # Goal check
        if self.num_graph_conflicts == 0:
            return self.assignment

        for i in range(max_steps):
            self.steps_req += 1
            # fetch conflicted variables
            cvars = self.get_conflicted_vars()
            # Goal check
            if not cvars:
                return self.assignment

            # select random conflicted variable
            shuffle(cvars)
            var = cvars.pop()

            # if prob = 7, Using random walk,
            # select min conflicted value 70% of times
            # and select a random value for variable
            # 30% times to avoid local
            # minima and plateau issues
            if randint(0, 9) < prob:
                # fetch min conflicted value for var
                val = self.min_conflicts_val(var)
            else:
                # select random value for variable
                val = choice(self.domain_vals)

            # assign
            self.assignment.update({var: val})

        return 'No answer'

    def get_conflicted_vars(self):
        return [n for n in self.vars if self.node_conflicts(n, self.assignment[n]) > 0]

    def make_initial_assignment(self):
        # Initial assignment by greedy approach of
        # choosing least conflicting val for each variable
        for var in self.vars:
            self.assignment.update({var: self.min_conflicts_val(var)})

    def min_conflicts_val(self, var):
        # Given a node, get its least conflicting value
        shuffle(self.domain_vals)
        return min(self.domain_vals, key=lambda x: self.node_conflicts(var, x))

    def node_conflicts(self, n1, val):
        # Given a node & its value, find the number of conflicts with its neighbours
        return list(map(lambda x: self.assignment.get(x) == val, self.adj_matrix[n1])).count(True)

    def num_graph_conflicts(self):
        cnt = 0
        for node in self.vars:
             cnt += self.node_conflicts(node, self.assignment.get(node))

        return cnt


def run():
    start = time.time()
    if len(sys.argv) == 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    else:
        print("Invalid Input, please try again")
        return

    with open(in_file) as f:
        num_var, num_const, domain = map(int, f.readline().split())
        constraints = f.readlines()

    adj_matrix = defaultdict(list)
    for c in constraints:
        cnst = list(map(int, c.split()))
        adj_matrix[cnst[0]].append(cnst[1])
        adj_matrix[cnst[1]].append(cnst[0])

    print(num_var, num_const, domain, adj_matrix)

    # Call algo
    max_steps = 10000
    prob = 6
    mc = MinConflicts(num_var, domain, adj_matrix)
    res = mc.solve_csp(max_steps, prob)
    print(mc.steps_req)

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