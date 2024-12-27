"""
PINN for lid driven cavity code by ehwan: github.com/ehwan/PINN/blob/main/cavity
"""
#! /usr/bin/bash

import torch
import numpy as np

class LidDrivenCavityPINN:
    def __init__(self,
              nu):

        self.nu = nu 
        #activation = torch.nn.Tanh
        activation = torch.nn.SiLU()
        #activation = SinActivation()

        Inputs = 20
        NumLayers = 8

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            activation,
            torch.nn.Linear(64, Inputs),
        )

        for i in range(NumLayers):
            self.net.append(activation)
            self.net.append(torch.nn.Linear(Inputs, Inputs))

        self.net.append(activation)
        self.net.append(torch.nn.Linear(Inputs, 2))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)
        # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.003)
        
    def forward(self,
                x, 
                y,
                t):
        return self.net(torch.hstack((x, y, t)))

    def function(self, 
                 x, 
                 y,
                 t):
        res = self.net(torch.hstack((x, y, t)))
        p = res[:, 0:1]
        psi = res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        uy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]

        vx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        vy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

        ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        vt = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        px = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        py = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        fx = ut + u*ux + v*uy + px - self.nu*(uxx + uyy)
        fy = vt + u*vx + v*vy + py - self.nu*(vxx + vyy)

        return u, v, p, fx, fy, psi
