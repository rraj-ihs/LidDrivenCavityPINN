"""
Main file for PINN Cavity problem
"""

import torch
import numpy as np 
import torch.nn as nn

from timeit import default_timer as timer

import matplotlib.pyplot as plt 

from utils import *
from pinn import *
from initial import *

from tqdm.auto import tqdm

if __name__== "__main__":

    # set seeds
    set_seeds()

    device = device_setup()
    print(f"Device is {device}")

    # set hyper parameters (later change it to a given terms from argparser)
    Re = 100  # Reynold's number
    # Simulation domain = [0, 1]x[0, 1]
    # training time = [0, T]
    # dirichlet BC on y=1 with u=1 and v=0
    # dirichlet BC on y=0, x=0 and x=1, u=0, v=0

    T = 5.0
    cavity = LidDrivenCavityPINN(nu=1.0/Re)

    try:
        cavity.net.load_state_dict(torch.load('cavity.pth'))
    except FileNotFoundError:
        pass


    # number of points for initail conditions
    N_init = 64

    # Number of points for upper boundary
    N_upper = 64

    # Number of points for each boundary
    N_boundary = 16

    # Number of points for inside domain
    N_mesh = 128

    T_init = torch.zeros(size=(N_init, 1), dtype=torch.float32, requires_grad=True)
    X_init = torch.rand(size=(N_init, 1), dtype=torch.float32, requires_grad=True)
    Y_init = torch.rand(size=(N_init, 1), dtype=torch.float32, requires_grad=True)*(1.0-1e-6)

    X_boundary_left = torch.zeros( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )
    Y_boundary_left = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )

    X_boundary_right = torch.ones( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )
    Y_boundary_right = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )

    X_boundary_down = torch.rand( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )
    Y_boundary_down = torch.zeros( size=(N_boundary,1), dtype=torch.float32, requires_grad=True )

    T_boundary = torch.rand( size=(N_boundary*3,1), dtype=torch.float32, requires_grad=True )*T

    
    X_upper = torch.rand( size=(N_upper,1), dtype=torch.float32, requires_grad=True )
    Y_upper = torch.ones( size=(N_upper,1), dtype=torch.float32, requires_grad=True )
    T_upper = torch.rand( size=(N_upper,1), dtype=torch.float32, requires_grad=True )*T

    X_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )
    Y_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )
    T_domain = torch.rand( size=(N_mesh,1), dtype=torch.float32, requires_grad=True )*T

    Epochs = 1000

    Loss = [0]*Epochs

    for i in tqdm(range(Epochs)):
        l = 0
        cavity.optimizer.zero_grad()

        # initial condition
        X_init, Y_init = InitialCondition(N_init)
        u, v, _, fx, fy, _ = cavity.function( X_init, Y_init, T_init )
        l = l + torch.mean( u**2 + v**2 + fx**2 + fy**2 )

        # left, right, down boundary
        Y_boundary_left, Y_boundary_right, X_boundary_down, T_boundary = BoundaryCondition(N_boundary, T)
        X_boundary = torch.vstack( (X_boundary_left, X_boundary_right, X_boundary_down) )
        Y_boundary = torch.vstack( (Y_boundary_left, Y_boundary_right, Y_boundary_down) )
        u, v, _, fx, fy, _ = cavity.function( X_boundary, Y_boundary, T_boundary )
        l = l + torch.mean( u**2 + v**2 + fx**2 + fy**2 )

        # up boundary
        X_upper, T_upper = UpperBoundaryCondition(N_upper, T)
        u, v, _, fx, fy, _ = cavity.function( X_upper, Y_upper, T_upper )
        l = l + torch.mean( (u-1.0)**2 + v**2 + fx**2 + fy**2 )

        # inside domain
        X_domain, Y_domain, T_domain = DomainCondition(N_mesh, T)
        u, v, _, fx, fy, _ = cavity.function( X_domain, Y_domain, T_domain )
        l = l + 2*torch.mean( fx**2 + fy**2 )

        print( i, l.item() )
        l.backward()
        cavity.optimizer.step()

        Loss[i] = l.item()

    torch.save( cavity.net.state_dict(), 'cavity.pt' )

    PlotN = 50
    PlotX = np.linspace(0, 1, PlotN)
    PlotY = np.linspace(0, 1, PlotN)
    PlotX, PlotY = np.meshgrid( PlotX, PlotY )
    PlotT = np.ones_like( PlotX )*0.8*T
    plotshape = PlotT.shape

    u, v, p, fx, fy, psi = cavity.function(
      torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
      torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True),
      torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32, requires_grad=True)
    )
    u = u.detach().numpy().reshape( plotshape )
    v = v.detach().numpy().reshape( plotshape )
    p = p.detach().numpy().reshape( plotshape )
    psi = psi.detach().numpy().reshape( plotshape )
    fx = fx.detach().numpy().reshape( plotshape )
    fy = fy.detach().numpy().reshape( plotshape )
    plt.quiver(
        PlotX, PlotY, u, v, np.sqrt(u**2+v**2),
    )

    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.title( 'u' )
    plt.show()

    plt.imshow( psi, origin='lower' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.colorbar()
    plt.title( 'psi' )
    plt.show()

    plt.imshow( p, origin='lower' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.colorbar()
    plt.title( 'pressure' )
    plt.show()

    plt.imshow( fx, origin='lower' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.colorbar()
    plt.title( 'fx' )
    plt.show()

    plt.imshow( fy, origin='lower' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.colorbar()
    plt.title( 'fy' )
    plt.show()

    Ys = np.linspace(0, 1, 100)
    Xs = np.ones_like(Ys)*0.5
    Ts = np.ones_like( Xs )*0.8*T

    u, v, p, fx, fy, psi = cavity.function(
        torch.tensor(Xs.reshape(-1,1), dtype=torch.float32, requires_grad=True),
        torch.tensor(Ys.reshape(-1,1), dtype=torch.float32, requires_grad=True),
        torch.tensor(Ts.reshape(-1,1), dtype=torch.float32, requires_grad=True)
    )
    u = u.detach().numpy().reshape( -1 )
    plt.plot( u, Ys, label='Neural Network' )


    # Ghia et al. (1982)
    Ys = [
    1.00000
    ,0.9766
    ,0.9688
    ,0.9609
    ,0.9531
    ,0.8516
    ,0.7344
    ,0.6172
    ,0.5000
    ,0.4531
    ,0.2813
    ,0.1719
    ,0.1016
    ,0.0703
    ,0.0625
    ,0.0547
    ,0.0000
    ]

    Us =[
    1.00000
    ,0.84123
    ,0.78871
    ,0.73722
    ,0.68717
    ,0.23151
    ,0.00332
    ,-0.13641
    ,-0.20581
    ,-0.21090
    ,-0.15662
    ,-0.10150
    ,-0.06434
    ,-0.04775
    ,-0.04192
    ,-0.03717
    ,0.00000
    ]

    plt.plot( Us, Ys, 'o-', label='Ghia et al.' )
    plt.xlabel( 'u' )
    plt.ylabel( 'y' )
    plt.legend()
    plt.show()

    Loss = np.array( Loss )

    plt.plot( Loss )
    plt.xlabel( 'Epochs' )
    plt.yscale( 'log' )
    plt.ylabel( 'Loss' )
    plt.show()
