import matplotlib.pyplot as plt
import numpy as np

class SudokuVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.set_aspect('equal')
        
        # Draw grid lines with varying thickness for better clarity
        for i in range(10):
            linewidth = 3 if i % 3 == 0 else 1
            color = 'black' if i % 3 == 0 else 'gray'
            self.ax.plot([0, 9], [i, i], color=color, linewidth=linewidth)
            self.ax.plot([i, i], [0, 9], color=color, linewidth=linewidth)
    
    def draw_board(self, board, initial_board=None, solution_board=None):
        """
        Draw the Sudoku board.
        :param board: 9x9 numpy array representing the current Sudoku state.
        :param initial_board: 9x9 numpy array representing the initial Sudoku state (to distinguish user input).
        :param solution_board: 9x9 numpy array representing the correct solution (to mark correct and incorrect inputs).
        """
        self.ax.clear()
        
        # Hide axis ticks again
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.set_aspect('equal')

        # Draw grid lines
        for i in range(10):
            linewidth = 2 if i % 3 == 0 else 0.5
            self.ax.axhline(i, color='black', linewidth=linewidth)
            self.ax.axvline(i, color='black', linewidth=linewidth)
        
        # Fill numbers in the board
        for i in range(9):
            for j in range(9):
                if board[i, j] != 0:
                    # Use bold font for initial numbers
                    if initial_board is not None and initial_board[i, j] != 0:
                        weight = 'bold'
                        color = 'black'
                    else:
                        weight = 'normal'
                        # Determine color based on solution_board
                        if solution_board is not None:
                            if board[i, j] == solution_board[i, j]:
                                color = 'blue'  # Correct numbers in blue
                            else:
                                color = 'red'   # Incorrect numbers in red
                        else:
                            color = 'blue'  # Default color is blue
                    
                    # Ensure the number is within the range of 1-9
                    num = min(max(int(board[i, j]), 1), 9)
                    self.ax.text(j + 0.5, 8 - i + 0.5, str(num),
                                 fontsize=18, ha='center', va='center',
                                 color=color, weight=weight,
                                 fontfamily='monospace')

        plt.show(block=True)
    
    def save(self, filename):
        """Save the current image."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
    def close(self):
        """Close the image window."""
        plt.close(self.fig)
