template_program = '''
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def run_packing():
    """
    Solves the circle packing problem for 26 circles in a unit square to maximize
    the sum of their radii.

    This implementation uses scipy.optimize.minimize with an SLSQP method.
    The optimization variables are the (x, y) coordinates of each circle's center
    and its radius.
    
    Key elements:
    - Initialization: Circles are initially placed in a slightly perturbed grid 
      pattern to provide a good starting point for the optimizer, preventing
      them from starting in highly overlapping or sub-optimal random positions.
    - Objective Function: Minimizes the negative sum of all circle radii,
      effectively maximizing the sum of radii.
    - Constraints:
        - Bounds: Center coordinates are within [0, 1], and radii are within
          a small positive minimum and 0.5 (maximum possible radius).
        - Nonlinear Constraints:
            - Wall Constraints: Ensure each circle is fully contained within
              the unit square (i.e., x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0).
            - Non-overlapping Constraints: Ensure the distance between any two
              circle centers is greater than or equal to the sum of their radii
              (i.e., (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0).
    - Optimizer: scipy.optimize.minimize with the 'SLSQP' method, which
      is suitable for problems with bounds and nonlinear inequality constraints.

    Returns:
      tuple: A tuple containing:
        - centers (np.ndarray): Shape (26, 2) with circle centers.
        - radii (np.ndarray): Shape (26,) with circle radii.
        - sum_of_radii (float): The sum of all radii.
    """
    N = 26 # Number of circles to pack

    # --- 1. Initial Guess Generation ---
    # We use a structured initial placement to help the optimizer find better solutions.
    # A 5x6 grid has 30 positions, we'll take the first 26 and slightly perturb them.
    grid_rows = 6
    grid_cols = 5
    
    initial_centers = []
    for r_idx in range(grid_rows):
        for c_idx in range(grid_cols):
            if len(initial_centers) < N:
                # Calculate ideal center for grid cell
                cx = (c_idx + 0.5) / grid_cols
                cy = (r_idx + 0.5) / grid_rows
                
                # Add a small random perturbation to break perfect symmetry and explore
                # a wider solution space, helping to escape local optima.
                perturbation_scale = 0.02 # Scale of random perturbation
                cx += (np.random.rand() - 0.5) * perturbation_scale / grid_cols
                cy += (np.random.rand() - 0.5) * perturbation_scale / grid_rows
                
                # Ensure perturbed centers remain reasonable (not too close to edges initially)
                cx = np.clip(cx, 0.01, 0.99)
                cy = np.clip(cy, 0.01, 0.99)
                
                initial_centers.append([cx, cy])
            else:
                break
        if len(initial_centers) == N:
            break
            
    initial_centers = np.array(initial_centers[:N])
    # Start with a small, uniform radius for all circles to avoid initial overlaps.
    initial_radii = np.full(N, 0.02) 

    # Flatten the initial guess into a 1D array for the optimizer:
    # [c1_x, c1_y, r1, c2_x, c2_y, r2, ..., cN_x, cN_y, rN]
    x0 = np.hstack((initial_centers.flatten(), initial_radii))

    # --- 2. Objective Function ---
    # The goal is to maximize the sum of radii, so we minimize the negative sum.
    def objective(params):
        radii = params[2*N:]
        return -np.sum(radii)

    # --- 3. Constraints ---
    # Define bounds for all variables:
    # Centers (x, y) must be between 0 and 1.
    # Radii (r) must be positive (min_r) and cannot exceed 0.5 (to fit in the square).
    min_r = 1e-6 # Minimum radius to prevent circles from collapsing to zero
    max_r = 0.5  # Maximum theoretical radius for a single circle in a unit square
    
    lower_bounds = np.zeros(3 * N)
    upper_bounds = np.ones(3 * N)
    
    lower_bounds[2*N:] = min_r # Apply min_r to all radii
    upper_bounds[2*N:] = max_r # Apply max_r to all radii
    
    bounds = Bounds(lower_bounds, upper_bounds)

    # Define nonlinear constraints:
    # These include wall containment and non-overlapping conditions.
    def constraints_fun(params):
        # Reshape parameters back into centers and radii
        centers = params[:2*N].reshape(N, 2)
        radii = params[2*N:]
        
        # List to store all constraint values. All must be >= 0.
        violations = []

        # 3a. Wall Containment Constraints (4 per circle)
        # For each circle (cx, cy, r):
        #   cx - r >= 0
        #   1 - cx - r >= 0
        #   cy - r >= 0
        #   1 - cy - r >= 0
        for i in range(N):
            cx, cy = centers[i]
            r = radii[i]
            violations.append(cx - r)
            violations.append(1 - cx - r)
            violations.append(cy - r)
            violations.append(1 - cy - r)

        # 3b. Non-overlapping Constraints (N * (N-1) / 2 pairs)
        # For any two circles (i, j) with centers (cxi, cyi) and (cxj, cyj)
        # and radii ri and rj:
        #   (cxi - cxj)^2 + (cyi - cyj)^2 - (ri + rj)^2 >= 0
        for i in range(N):
            for j in range(i + 1, N): # Iterate over unique pairs
                d_sq = np.sum((centers[i] - centers[j])**2) # Squared distance between centers
                r_sum_sq = (radii[i] + radii[j])**2         # Squared sum of radii
                violations.append(d_sq - r_sum_sq)
                
        return np.array(violations)

    # All constraints are C(x) >= 0, so the lower bound is 0, upper is infinity.
    num_wall_constraints = 4 * N
    num_overlap_constraints = N * (N - 1) // 2
    total_constraints = num_wall_constraints + num_overlap_constraints

    constraints = NonlinearConstraint(
        fun=constraints_fun,
        lb=np.zeros(total_constraints),
        ub=np.full(total_constraints, np.inf)
    )

    # --- 4. Optimization ---
    # Using 'SLSQP' (Sequential Least SQuares Programming) method.
    # It handles bounds and nonlinear inequality constraints well.
    # maxiter is increased for potentially better convergence on a complex problem.
    options = {'disp': False, 'maxiter': 10000, 'ftol': 1e-8} 

    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )

    # --- 5. Extract and Return Results ---
    if not res.success:
        print(f"Optimization did not converge successfully: {res.message}")
        print("Returning the best solution found so far.")
    
    optimized_params = res.x
    
    # Reshape the optimized parameters back into centers and radii
    final_centers = optimized_params[:2*N].reshape(N, 2)
    final_radii = optimized_params[2*N:]
    final_sum_of_radii = np.sum(final_radii)

    return (final_centers, final_radii, final_sum_of_radii)
'''

task_description = "Implement a function that uses a constructive heuristic to pack n non-overlapping circles iteratively within a unit square to maximize the sum of their radii"
