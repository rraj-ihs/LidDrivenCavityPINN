"""
Initial conditions and initail mesh generations
"""

import torch

# generate random points for t=0
def InitialCondition(N_init):
    global X_init
    global Y_init
    X_init = torch.rand( size=(N_init,1), dtype=torch.float32, requires_grad=True )
    Y_init = torch.rand( size=(N_init,1), dtype=torch.float32, requires_grad=True )*(1.0-1e-6)
    
    return X_init, Y_init


# generate random points for boundary condition
def BoundaryCondition(N_boundary, T):
    global Y_boundary_left
    Y_boundary_left = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )

    global Y_boundary_right
    Y_boundary_right = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )
  
    global X_boundary_down
    X_boundary_down = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )

    global T_boundary
    T_boundary = torch.rand( size=(N_boundary*3,1), dtype=torch.float32, requires_grad=True )*T

    return Y_boundary_left, Y_boundary_right, X_boundary_down, T_boundary

# generate random points for upper boundary condition
def UpperBoundaryCondition(N_upper, T):
    global X_upper
    global T_upper
    X_upper = torch.rand( size=(N_upper,1), dtype=torch.float32, requires_grad=True )
    T_upper = torch.rand( size=(N_upper,1), dtype=torch.float32, requires_grad=True )*T

    return X_upper, T_upper

# generate random points for inside domain condition ( NS equation )
def DomainCondition(N_mesh, T):
    global X_domain
    global Y_domain
    global T_domain
    X_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )
    Y_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )
    T_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )*T

    return X_domain, Y_domain, T_domain
