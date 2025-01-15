import torch
from torch import nn, Tensor
from scipy.optimize import linprog

class OTSolver:
    def __init__(
        self, 
        epsilon: float=1e-3, 
        beta: float=2, 
        max_iter: int=1000, 
        L: int=1, 
        use_path: bool=True, 
        tol: float=1e-9, 
        method: str='ipot'
    ) -> None:
        """
        Args:
            beta (float): Step size of proximal point iteration.
            L (int): Number of iterations for inner optimization.
            use_path (bool): Whether warm start method is used.
        """
        self.epsilon: float = epsilon
        self.beta = beta
        self.max_iter: int = max_iter
        self.L = L
        self.use_path = use_path
        self.tol: float = tol
        self.method: str = method

    def solve(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Solves the optimal transport problem based on the specified method.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        if self.method == 'sinkhorn':
            return self.sinkhorn_knopp(mu, nu, C)
        elif self.method == 'ipot':
            return self.ipot(mu, nu, C)
        elif self.method == 'linear':
            return self.linear_programming(mu, nu, C)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def sinkhorn_knopp(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Solves the optimal transport problem using the Sinkhorn-Knopp algorithm.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        K = torch.exp(-C / self.epsilon)  # Kernel matrix using entropy regularization
        u = torch.ones_like(mu, device=mu.device)
        v = torch.ones_like(nu, device=nu.device)
        
        for _ in range(self.max_iter):
            u_new = mu / (torch.matmul(K, v) + self.tol)  # Adding tolerance for numerical stability
            v_new = nu / (torch.matmul(K.T, u_new) + self.tol)
            
            # Check for convergence
            if torch.norm(u_new - u) < self.tol and torch.norm(v_new - v) < self.tol:
                break
            
            u, v = u_new, v_new
        
        # Transport plan is the element-wise multiplication of K, u, and v
        P = torch.diag(u) @ K @ torch.diag(v)
        
        return P
        
    def ipot(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Inexact Proximal Point (IPOT) algorithm for optimal transport.
        
        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        m = len(mu)
        n = len(nu)
        a = torch.ones([m,])
        b = torch.ones([n,])

        Gamma = torch.ones((m,n))/m*n
        G = torch.exp(-(C/self.beta))

        for _ in range(self.max_iter):
            Q = G*Gamma
            if self.use_path == False:
                a = torch.ones([m,])
                b = torch.ones([n,])
            
            for i in range(self.L):
                a = mu/torch.matmul(Q,b)
                b = nu/torch.matmul(torch.transpose(Q),a)
        
            Gamma = torch.expand_dims(a,axis=1) * Q * torch.expand_dims(b,axis=0)
                
        return Gamma
    
    def linear_programming(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Solves the optimal transport problem using linear programming.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        n, m = C.shape
        c = C.flatten().cpu().numpy()  

        # Construct equality constraints
        A_eq = torch.zeros((n + m, n * m))
        for i in range(n):
            A_eq[i, i * m:(i + 1) * m] = 1  # Row constraints for source distribution
        for j in range(m):
            A_eq[n + j, j::m] = 1  # Column constraints for target distribution
        
        b_eq = torch.cat([mu, nu])  

        # Convert to NumPy for scipy.linprog
        A_eq_np = A_eq.numpy()
        b_eq_np = b_eq.cpu().numpy()

        # Bounds for each variable (non-negative transport plan)
        bounds = [(0, None) for _ in range(n * m)]

        result = linprog(c, A_eq=A_eq_np, b_eq=b_eq_np, bounds=bounds, method='highs')

        if not result.success:
            raise ValueError(f"Linear programming failed: {result.message}")

        P = torch.tensor(result.x, device=C.device).reshape(n, m)
        return P