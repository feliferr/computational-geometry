  
import os
import pandas as pd
# from paa191t2.branch_and_bound.problem_set import ProblemSet


class Loader:
    def __init__(self, resource_file="minimum-circle/data/points.txt"):
        self.resource_file = resource_file
    
    def parser_from_file(self, filename):


    # def parse_from_file(self, filename):
    #     with open(os.path.join(self.resource_folder, filename)) as all_instances:
    #         n, m, best = all_instances.readline().strip().replace("/n", "").split()
    #         best_indexes = all_instances.readline().strip().replace("/n", "").split()[1:]

    #         instances = {}
    #         weights = dict()

    #         for raw_line in all_instances.readlines():
    #             line = raw_line.strip().replace("/n", "").split()
    #             ident = line[0]
    #             c = line[1]
    #             instances[ident] = line[3:]
    #             weights[ident] = float(c)

    #         return ProblemSet(n, m, best, best_indexes, instances, weights)