from __future__ import annotations

from typing import Any
import numpy as np
from template import template_program, task_description
import itertools
from llm4ad.base import Evaluation

__all__ = ['CirclePackingEvaluation']


class CirclePackingEvaluation(Evaluation):
    """Evaluator for circle packing problem in a unit square."""

    def __init__(self,
                 timeout_seconds=30,
                 **kwargs):
        """
        Args:
            timeout_seconds: Time limit for evaluation
            n_instance: Number of problem instances to evaluate
            max_circles: Maximum number of circles to pack (n)
        Raises:
            ValueError: If invalid parameters are provided
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n = 26

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def verify_circles(self, circles: np.ndarray) -> bool:
        """Checks that the circles are disjoint and lie inside a unit square.

        Args:
            circles: A numpy array of shape (num_circles, 3), where each row is
                of the form (x, y, radius), specifying a circle.

        Returns:
            bool: True if valid, False otherwise
        """
        epsilon = 1e-5 # Tolerance for floating point comparisons
        try:
            # Check pairwise disjointness
            for circle1, circle2 in itertools.combinations(circles, 2):
                center_distance = np.sqrt((circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2)
                radii_sum = circle1[2] + circle2[2]
                if center_distance < radii_sum - epsilon: # Allow small overlap
                    # print(f"Overlap detected: dist={center_distance}, radii_sum={radii_sum}")
                    return False

            # Check all circles lie inside the unit square [0,1]x[0,1]
            for circle in circles:
                if not (0 - epsilon <= min(circle[0], circle[1]) - circle[2] and
                        max(circle[0], circle[1]) + circle[2] <= 1 + epsilon):
                    # print(f"Boundary violation: {circle}")
                    return False
            return True
        except Exception:
            return False



    def plot_circles(self,circles: np.ndarray):

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        """Plots the circles."""
        _, ax = plt.subplots(1, figsize=(7, 7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')  # Make axes scaled equally.

        # Draw unit square boundary.
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        # Draw the circles.
        for circle in circles:
            circ = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='blue', facecolor='skyblue', alpha=0.5)
            ax.add_patch(circ)

        plt.title(
            f'A collection of {len(circles)} disjoint circles packed inside a unit square to maximize the sum of radii')
        plt.show()

    def evaluate(self, eva: callable) -> float:
        """Evaluate the circle packing solution."""
        if eva is None:
            return -float('inf')
            
        try:
            # The initial code provided returns a tuple (centers, radii, sum_of_radii)
            # But the evaluation class expects something else or the template function signature is different?
            # Let's check template.py signature from original setup vs what we put in.
            
            # The original template.py had:
            # def pack_circles(n: int) -> np.ndarray: return circles (n x 3)
            
            # The code in start.txt has:
            # def run_packing():
            # Returns (centers, radii, sum_of_radii)
            
            # Since the LLM is initialized with the code in template.py, and the evaluation calls `eva(self.n)`,
            # we need to be careful.
            
            # If the LLM produces a function `run_packing` that takes no arguments, `eva(self.n)` will fail.
            # But the prompt/task description says "Implement a function ... to pack n circles".
            
            # However, the user provided `start.txt` explicitly with `def run_packing():` taking NO arguments, and hardcoding N=26.
            # This conflicts with `eva(self.n)`.
            
            # If I look at the `template.py` I created, it contains `def run_packing():`.
            # If I look at `evaluation.py` I am creating, it calls `circles = eva(self.n)`.
            
            # I must adjust `evaluation.py` to handle `run_packing()` if that's what the LLM generates or starts with.
            # Or I should adjust the template code signature to match what `evaluation.py` expects?
            
            # But the user said "EXACTLY COPYING the python at the bottom to use it as your initial code".
            # The initial code has `def run_packing():`.
            
            # So I should modify `evaluation.py` to handle the `run_packing` signature.
            # `eva` will be the function compiled from the code.
            
            # Let's see how `eva` is called.
            # `circles = eva(self.n)`
            
            # If `eva` is `run_packing`, it takes 0 args.
            
            # I will modify `evaluate` method in `example/cpack/evaluation.py` to try calling with `n` args, and if it fails (TypeError), try calling without args.
            # And also the return type is different!
            # run_packing returns (centers, radii, sum_of_radii).
            # original pack_circles returns np.ndarray of shape (n, 3).
            
            # I need to handle the return value of `run_packing` to convert it to what `verify_circles` expects, or just use the sum directly if valid.
            
            res = None
            try:
                res = eva(self.n)
            except TypeError:
                res = eva()
            
            # If res is tuple (centers, radii, sum_of_radii)
            if isinstance(res, tuple) and len(res) == 3:
                centers, radii, sum_r = res
                # Convert to (n, 3) format for verify_circles if needed, or just verify using centers and radii
                # verify_circles expects (n, 3) array: [x, y, r]
                circles_arr = np.hstack((centers, radii.reshape(-1, 1)))
                circles = circles_arr
            else:
                circles = res
                
        except Exception as e:
            print(f"Error during execution: {e}")
            return -float('inf')

        #self.plot_circles(circles)
        # Convert to numpy array if not already
        circles = np.array(circles, dtype=np.float64)

        # Verify the solution
        if not self.verify_circles(circles):
            print("Verification failed: Circles validation check returned False") 
            return -float('inf')
            
        if len(circles) != self.n:
            print(f"Verification failed: Expected {self.n} circles, got {len(circles)}")
            return -float('inf')

        # Sum of radii is our score
        score = np.sum(circles[:, 2])

        return score
