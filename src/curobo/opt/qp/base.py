
from abc import abstractmethod

class QPSolver:
    
    solver: None
    
    g_matrix: None 
    
    G_matrix: None 
    
    l_matrix: None
    
    h_matrix: None 
    
    def init_problem(self, G_matrix, l_matrix, h_matrix):
        self.G_matrix = G_matrix
        self.l_matrix = l_matrix
        self.h_matrix = h_matrix

    @abstractmethod
    def solve(self, Q_matrix, semi_Q_matrix, solution=None):
        return solution
    
    
