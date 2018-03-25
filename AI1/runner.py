import subprocess
import os
import random

if __name__ == '__main__':
    for i in range(10):
        d = random.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
        subprocess.check_output(['python', 'puzzleGenerator.py', '3', str(d), 'test.txt'])
        print("A* algorithm")
        os.system('python puzzleSolver.py 1 3 1 test.txt out.txt')
        os.system('python puzzleSolver.py 1 3 2 test.txt out.txt')
        print("IDA* algorithm")
        os.system('python puzzleSolver.py 2 3 1 test.txt out.txt')
        os.system('python puzzleSolver.py 2 3 2 test.txt out.txt')
    for i in range(10):
        d = random.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
        subprocess.check_output(['python', 'puzzleGenerator.py', '4', str(d), 'test.txt'])
        print("A* algorithm")
        os.system('python puzzleSolver.py 1 4 1 test.txt out.txt')
        os.system('python puzzleSolver.py 1 4 2 test.txt out.txt')
        print("IDA* algorithm")
        os.system('python puzzleSolver.py 2 4 1 test.txt out.txt')
        os.system('python puzzleSolver.py 2 4 2 test.txt out.txt')

