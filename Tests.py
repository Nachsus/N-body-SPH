# Enheder: kpc, 10**10 M_sol, Gyr, Kelvin

from numba.typed.typedlist import T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numpy.random import uniform
from time import time
from tqdm.auto import trange
import copy
import math
import random
from pygadgetreader import *
from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing

from threading import Thread

from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, prange, types, typeof
from numba.experimental import jitclass
from numba.typed import List

np.seterr(invalid='raise');           # Fail the code if any exception is raised 


sam=0

cpu_count = 12

density_test = False
Multi_position = False
Position_dm = False
Position_gas = False
Position_gas_and_dm = False
test2 = False
Energy1 = False

filename = 'gas100.g2'


G = 6.67428*10**(-11)*(3.0856776*10**(19))**(-3)*10**(10)*1.9891*10**(30)*(31557600*10**9)**2
N = 100
m = 1
k_B = 1.38064852*10**(-23)   *(3.24077929*10**(-20))**2   *(5.02739933*10**(-31)*10**(-10))   *(3.15576*10**(16))**2
m_u = 10**(-10) * (5.02739933 * 10**(-31)) * (1.6605402 * 10**(-27))
R_g = k_B/m_u
print(R_g)
print('--')
mu = 0.6 #Sagde Troels?


#Viscosity
alpha = 0.9
beta = 2*alpha
eta = np.sqrt(0.01)

tsteptest = 0

thetafaktor = 0.5
soft = 0.1
Cdt = 0.05

N_target = 5

t = 1
tsteps = 1600
tstep = t/tsteps

start_box_xmin = -100
start_box_xmax = 100
start_box_ymin = -100
start_box_ymax = 100
start_box_zmin = -100
start_box_zmax = 100

nBins = 100

histData = []
v = []


def find_W(body1, body2, h):

    L = 1
    a = body1.pos[0]
    b = body2.pos[0]
    dxdydz_list = [abs(a-b),abs(a+L-b),abs(a-L-b)]
    dxdydz = sorted(dxdydz_list)[0]
    dist = dxdydz

    #dist = np.sqrt(sum((body1.pos - body2.pos)**2))
    if 0 <= dist/h <= 1:
        W = 1 / (np.pi * h**3) * (1 - 3/2 * (dist/h)**2 + 3/4 * (dist/h)**3)
    elif 1 < dist/h <= 2:
        W = 1 / (4 * np.pi * h**3) * (2 - (dist/h))**3
    elif dist/h >= 2:
        W = 0
    else:
        print(body1.pos, body2.pos)
    return W

def nW(body1, body2, h):

    L = np.array([1,0,0])
    a = body1.pos
    b = body2.pos
    dxdydz_list = [a-b,a+L-b,a-L-b]
    dxdydz = sorted(dxdydz_list, key=lambda x: np.sqrt(sum(x)**2))[0]
    dist = np.sqrt(sum(dxdydz**2))

    #dist = np.sqrt(sum((body1.pos - body2.pos)**2))
    #dxdydz = body1.pos - body2.pos
    if 0 <= dist/h <= 1:
        nW =  1/(np.pi * h**3) * ((-3 * dist/h**2) + (9 * dist**2 /(4 * h**3)))  
    elif 1 < dist/h <= 2:
        nW = - 3/(4 * np.pi * h**4) * (2 - dist/h)**2                             
    elif dist/h >= 2:
        nW = 0
    n = nW*dxdydz/dist
    for bod in n:
        if math.isnan(bod):
            print(dist)
    return nW*dxdydz/dist

'''class test():
    def __init__(self,a):
        self.a = a
class_array = [test(1),test(2)]


stringtest = 'string'

spec_body = [
    ('mass',float64),
    ('pos',float64[:,:]),
    ('v',float64[:,:]),
    ('acc',float64[:,:]),
    ('accbrute',float64[:,:]),
    ('masseffect',float64),
    ('masseffect_brute',float64),
    ('dt_old',float64),
    ('dt',float64),
    ('tseen',float64),
    ('type',typeof(stringtest)),
    ('density',float64),
    ('P',float64),
    ('T',float64),
    ('h',float64),
    ('sound_speed',float64)
]'''
gamma = 1.001
#@jitclass(spec_body)
class Body():
    def __init__(self, mass, xpos, ypos, zpos, vx, vy, vz, type, Eint=0, h=0):
        self.mass = mass
        self.pos = np.array([xpos, ypos, zpos])
        self.v = np.array([vx, vy, vz])
        self.acc = np.array([0., 0., 0.])
        self.accbrute = np.array([0., 0., 0.])
        self.masseffect = 0
        self.masseffect_brute = 0
        self.dt_old = tstep
        self.dt = tstep
        self.tseen = 0
        self.type = type
        if type == 'gas':
            self.density_comps = []
            self.density = 0
            self.P = 0
            self.Eint = Eint
            self.T = (2 * self.Eint)/(3 * R_g) #temperature of gas
            self.h = h
            self.sound_speed = 0
            self.prediction_velocity = np.array([0.,0.,0.])
            self.a_old = np.array([0.,0.,0.])
            self.k = 0
            self.visc_tensors = []
            self.nW_h_i = []
            self.nW_h_j = []
            self.Pd_fac = []
            self.old_density_comps = []

    def update_T(self):
        self.T = (2 * self.Eint)/(3 * R_g)
        #self.T = self.P / self.density * #mu * m_p / k_B
        if self.T <0:
            print('Problem T',self.Eint)


    def update_k(self):
        self.k = 10 * (self.T/10**4)**(1/6) * (self.density * 10**(10)/10**6)**(-1/2)

    def add_force_brute(self, body):
        dxdydz = self.pos - body.pos
        d = np.sqrt(sum(dxdydz**2))
        if 0 <= d/soft < 1/2:
            softg = 4/soft**2 * (8/3 * (d/soft) - 48/5 * (d/soft)**3 + 8 * (d/soft)**4)
            self.accbrute += - G * self.mass * softg * (dxdydz/d)
        elif 1/2 <= d/soft < 1:
            softg = 4/soft**2 * (16/3 * (d/soft) - 12 * (d/soft)**2 + 48/5 * (d/soft)**3 - 8/3 * (d/soft)**4 - 1/60 * (d/soft)**(-2))
            self.accbrute += - G * self.mass * softg * (dxdydz/d)
        elif d/soft >= 1:
            softg = 1/d**2
            self.accbrute += - G * self.mass * softg * (dxdydz/d)
        self.masseffect_brute += body.mass

    def kick(self, dt):
        self.v += 0.5 * dt * self.acc

    def drift(self, dt):
        self.pos += dt * self.v


    def a_g(self, branch):
        branch.walk(self)

    def find_dt(self):
        if np.sqrt(sum(self.acc**2)) == 0:
            dt_body = tstep
        else:
            dt_body = Cdt*np.sqrt(self.h/(np.sqrt(sum(self.acc**2))))
        n = 0
        while tstep/2**n > dt_body:
            n += 1
        dt1 = tstep/2**n
        n1 = n

        if np.sqrt(sum(self.acc**2)) == 0:
            dt_body = tstep
        else:
            maxmu = 0
            for gas in self.density_comps:
                muher = my(self,gas)
                if muher > maxmu:
                    maxmu = muher

            dt_body = self.h/(self.sound_speed + 0.6*(alpha*self.sound_speed + beta*maxmu))
        n = 0
        while tstep/2**n > dt_body:
            n += 1
        n2 = n
        dt2 = tstep/2**n 
        dt = min([dt1,dt2])
        n = max([n1,n2])
        return dt, n

    def check_for_gas(self):
        self.density_comps.append(self)
        for body in self.density_comps:
            if body.type != 'gas':
                self.density_comps.remove(body)
            elif np.sqrt(sum((self.pos - body.pos)**2)) > 2 * self.h:
                self.density_comps.remove(body)
    
    def update_density(self,dt):
        old_dens = self.density
        self.density = 0

        #self.density_comps.append(self)
        old_dens_comps = self.density_comps
        #self.old_density_comps = old_dens_comps
        #self.density_comps = []

        #for body in old_dens_comps:
        #    if body.type == 'gas' and np.sqrt(sum((self.pos - body.pos)**2)) <= 2 * self.h and body not in self.density_comps:
        #        self.density_comps.append(body)

        #for bod in self.density_comps:
        #    if np.sqrt(sum((self.pos - bod.pos)**2)) > 2*self.h:
        #        print('Yep!!')

        for body2 in self.density_comps:
            self.density += body2.mass * ( find_W(self,body2,body2.h) + find_W(self,body2,self.h) )
        self.density += self.mass * ( find_W(self,self,self.h) + find_W(self,self,self.h))
        self.density = self.density/2


    def find_pressure(self):
        #self.P = R_g/mu * self.density * self.T

        self.P = (gamma-1)*self.density*self.Eint
        if self.P < 0:
            print(self.pos)
            print('Problem here', self.density, self.Eint, self.P)
        
    def pressure_gradient(self):
        if len(self.density_comps) > 1:   
            '''or gas in self.density_comps:
                if self != gas:
                    trykfac1 = np.sqrt(self.P*gas.P)/(self.density*gas.density)
                    #P_grad = gas.mass * (trykfac1) * (nW(self, gas, self.h) + nW(self, gas, gas.h))
                    P_grad = gas.mass * trykfac1 * ()
                    self.acc += -P_grad'''
            for i in range(len(self.density_comps)):
                P_grad = self.density_comps[i].mass * self.Pd_fac[i] * (self.nW_h_i[i] + self.nW_h_j[i])
                self.acc += -P_grad


    def E_int(self,dt):
        '''for gas in self.density_comps:
            dE_adb = gas.mass * (np.sqrt(self.P * gas.P)/(self.density * gas.density)) * sum((self.v - gas.v) * 1/2 * (nW(self, gas, self.h) + nW(self, gas, gas.h)))
            self.Eint += dE_adb * dt
            dE_visc = gas.mass/2 * self.visc_tensor(gas) * sum((self.v - gas.v) * 1/2 * (nW(self, gas, self.h) + nW(self, gas, gas.h)))
            self.Eint += dE_visc * dt

            #self.Eint += gas.mass/2 * self.visc_tensor(gas) * sum((self.v - gas.v) * 1/2 * (nW(self, gas, self.h) + nW(self, gas, gas.h)))'''

        for i in range(len(self.density_comps)):
            dE_adb = self.density_comps[i].mass * self.Pd_fac[i] * sum((self.v - self.density_comps[i].v) * 1/2 * (self.nW_h_i[i] + self.nW_h_j[i]))
            self.Eint += dE_adb * dt
            dE_visc = self.density_comps[i].mass/2 * self.visc_tensors[i] * sum((self.v - self.density_comps[i].v) * 1/2 * (self.nW_h_i[i] + self.nW_h_j[i]))
            self.Eint += dE_visc * dt
        

    def heat_transfer(self,dt):
        #k = 10 #?????????
        #for gas in self.density_comps:
            #if self != gas:
                #E = - gas.mass * (k + k) * (self.T - gas.T) * sum((gas.pos - self.pos) * nW(self, gas, (self.h + gas.h)/2)) / (self.density * gas.density * np.sqrt(sum(((self.pos - gas.pos)**2)**2)))
                #E = - gas.mass * (k + k) * (self.T - gas.T) * sum((gas.pos - self.pos) * 1/2 * (nW(self, gas, self.h) + nW(self,gas,gas.h)   )) / (self.density * gas.density * np.sqrt(sum(((self.pos - gas.pos)**2)**2)))
                #if self.Eint + E > 200:
                #self.Eint += E*dt
                #elif self.Eint + E <= 200:
                #    self.Eint = 200
        a = 0

    def find_sound_speed(self):
        '''
        Måske har jeg formlen:
        v_s = sqrt(gamma*P/rho)
        Jeg tror måske man kan bruge gamma = 3/5 men jeg har ingen god grund hertil.
        '''
        self.sound_speed = np.sqrt(gamma*self.P/self.density)
        #return np.sqrt(gamma*self.P/self.density)
        #print(self.sound_speed)

        # Anden formel for lydhastigheden
        #self.sound_speed = np.sqrt(R_g * self.T / 1)

    def visc_tensor(self,body):
        return (- 0.5 * alpha * my(self, body) * (self.sound_speed + body.sound_speed) + beta * my(self, body)**2)/(0.5 * (self.density + body.density))

    def a_visc(self):
        '''for gas in self.density_comps:
            if self != gas:
                self.acc += -1/2 * gas.mass * self.visc_tensor(gas) * (nW(self, gas, self.h) + nW(self, gas, gas.h))'''

        for i in range(len(self.density_comps)):
            self.acc += -1/2 * self.density_comps[i].mass * self.visc_tensors[i] * (self.nW_h_i[i] + self.nW_h_j[i])

    def find_prediction_velocity(self,dt):
        self.prediction_velocity = self.v + 0.5*dt*self.a_old

    def find_initial_h(self,gas_bodies):
        '''dists = []
        L = 1
        for gas in gas_bodies:
            if self != gas:
                a = self.pos[0]
                b = gas.pos[0]
                dxdydz_list = [abs(a-b),abs(a+L-b),abs(a-L-b)]
                dxdydz = sorted(dxdydz_list)[0]
                dist = dxdydz
                dists.append(dist)
        self.h = dists[N_target]'''

        self.h = 0.037

        #s = sorted(gas_bodies, key=lambda x: np.sqrt(sum((self.pos-x.pos)**2)))
        #self.h = 1 * np.sqrt(sum((self.pos - s[N_target].pos)**2))

    def test_find_comps(self,gas_bodies):
        L = 1
        self.density_comps = []
        for gas in gas_bodies:
            if self != gas:
                a = self.pos[0]
                b = gas.pos[0]
                dxdydz_list = [abs(a-b),abs(a+L-b),abs(a-L-b)]
                dxdydz = sorted(dxdydz_list)[0]
                dist = dxdydz
                if dist < 2* self.h:
                    self.density_comps.append(gas)




    def test_find_comps2(self):
        self.density_comps = self.old_density_comps

    def update_h(self):
        self.h = self.h * (1/2) * (1 + (N_target/len(self.density_comps))**(1/3))

    def find_visc_tensors(self):
        self.visc_tensors = []
        for gas in self.density_comps:
            if self != gas:
                self.visc_tensors.append(self.visc_tensor(gas))

    def find_nWs(self):
        self.nW_h_i = []
        self.nW_h_j = []
        for gas in self.density_comps:
            if self != gas:
                self.nW_h_i.append(nW(self, gas, self.h))
                self.nW_h_j.append(nW(self, gas, gas.h))

    def find_Pd_fac(self):
        self.Pd_fac = []
        for body in self.density_comps:
            self.Pd_fac.append(np.sqrt(self.P*body.P)/(self.density*body.density))


def my(body1, body2):
    L = np.array([1,0,0])
    a = body1.pos
    b = body2.pos
    dxdydz_list = [a-b,a+L-b,a-L-b]
    dxdydz = sorted(dxdydz_list, key=lambda x: np.sqrt(sum(x)**2))[0]
    dist = np.sqrt(sum(dxdydz**2))
    #print(body1.prediction_velocity,body2.prediction_velocity)
    dot_product = np.sum((body1.prediction_velocity - body2.prediction_velocity) * (dxdydz))
    if dot_product < 0:
        h_ij = 0.5 * (body1.h + body2.h)
        #r_ij = np.sqrt(np.sum((body1.pos - body2.pos)**2))
        r_ij = dist
        return dot_product/(h_ij * (r_ij**2/h_ij**2 + eta**2))
    else:
        return 0



class Branch():
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, btc):
        self.lims = np.array([[xmin, ymin, zmin],[xmax, ymax, zmax]])
        self.sons = []
        self.btc = btc
        #self.density_comps = []
        self.bodies_in_corners(btc)
        if self.Nbodies > 1:
            self.make_branches()
        
        self.COM_func()
    
    def between(self, body):
        if (self.lims[0] < body.pos).all() and (body.pos < self.lims[1]).all():
            return True
        else:
            return False
    
    def bodies_in_corners(self, btc):
        self.bodies = []
        for i in range(len(btc)):
            if self.between(btc[i]) == True:
                self.bodies.append(btc[i])
        self.Nbodies = len(self.bodies)

    def make_branches(self):
        self.mid = (self.lims[0] + self.lims[1]) / 2
        sons = []
        son1 = Branch(self.lims[0][0], self.mid[0], self.lims[0][1], self.mid[1], self.lims[0][2], self.mid[2], self.bodies)
        if son1.Nbodies > 0:
            sons.append(son1)
        son2 = Branch(self.mid[0], self.lims[1][0], self.lims[0][1], self.mid[1], self.lims[0][2], self.mid[2], self.bodies)
        if son2.Nbodies > 0:
            sons.append(son2)
        son3 = Branch(self.lims[0][0], self.mid[0], self.mid[1], self.lims[1][1], self.lims[0][2], self.mid[2], self.bodies)
        if son3.Nbodies > 0:
            sons.append(son3)
        son4 = Branch(self.mid[0], self.lims[1][0], self.mid[1], self.lims[1][1], self.lims[0][2], self.mid[2], self.bodies)
        if son4.Nbodies > 0:
            sons.append(son4)
        son5 = Branch(self.lims[0][0], self.mid[0], self.lims[0][1], self.mid[1], self.mid[2], self.lims[1][2], self.bodies)
        if son5.Nbodies > 0:
            sons.append(son5)
        son6 = Branch(self.mid[0], self.lims[1][0], self.lims[0][1], self.mid[1], self.mid[2], self.lims[1][2], self.bodies)
        if son6.Nbodies > 0:
            sons.append(son6)
        son7 = Branch(self.lims[0][0], self.mid[0], self.mid[1], self.lims[1][1], self.mid[2], self.lims[1][2], self.bodies)
        if son7.Nbodies > 0:
            sons.append(son7)
        son8 = Branch(self.mid[0], self.lims[1][0], self.mid[1], self.lims[1][1], self.mid[2], self.lims[1][2], self.bodies)
        if son8.Nbodies > 0:
            sons.append(son8)
        self.sons = sons
    
    # def make_branches(self):
    #     self.mid = (self.lims[0] + self.lims[1]) / 2
    #     son1 = Branch(self.lims[0][0], self.mid[0], self.lims[0][1], self.mid[1], self.lims[0][2], self.mid[2], self.bodies)
    #     son2 = Branch(self.mid[0], self.lims[1][0], self.lims[0][1], self.mid[1], self.lims[0][2], self.mid[2], self.bodies)
    #     son3 = Branch(self.lims[0][0], self.mid[0], self.mid[1], self.lims[1][1], self.lims[0][2], self.mid[2], self.bodies)
    #     son4 = Branch(self.mid[0], self.lims[1][0], self.mid[1], self.lims[1][1], self.lims[0][2], self.mid[2], self.bodies)
    #     son5 = Branch(self.lims[0][0], self.mid[0], self.lims[0][1], self.mid[1], self.mid[2], self.lims[1][2], self.bodies)
    #     son6 = Branch(self.mid[0], self.lims[1][0], self.lims[0][1], self.mid[1], self.mid[2], self.lims[1][2], self.bodies)
    #     son7 = Branch(self.lims[0][0], self.mid[0], self.mid[1], self.lims[1][1], self.mid[2], self.lims[1][2], self.bodies)
    #     son8 = Branch(self.mid[0], self.lims[1][0], self.mid[1], self.lims[1][1], self.mid[2], self.lims[1][2], self.bodies)
    #     self.sons = [son1, son2, son3, son4, son5, son6, son7, son8]
    
    def COM_func(self):
        if self.Nbodies == 0:
            self.mass = 0
        else:
            xyz = np.array([0., 0., 0.])
            Mass = 0
            for body in self.bodies:
                xyz += body.pos * body.mass
                Mass += body.mass
            
            self.COM = xyz / Mass
            self.mass = Mass
    
    def theta(self, body):
        if self.mass != 0:
            l = abs(self.lims[0][0] - self.lims[1][0])
            d = np.sqrt(sum((self.COM - body.pos)**2))
            if d == 0:
                return 0
            else:
                return l/d
        else:
            return 0

    def walk(self, body, sam):
        if self.theta(body) > thetafaktor and self.Nbodies > 1:
            for son in self.sons:
                son.walk(body, sam)
        else:
            #sam += len(self.bodies)
            self.add_force(body)
    
    def add_force(self, body):
        if self.Nbodies > 0 and body not in self.bodies:
            dxdydz = body.pos - self.COM
            d = np.sqrt(sum(dxdydz**2))
            if 0 <= d/soft < 1/2:
                softg = 4/soft**2 * (8/3 * (d/soft) - 48/5 * (d/soft)**3 + 8 * (d/soft)**4)
                body.acc += - G * self.mass * softg * (dxdydz/d)
            elif 1/2 <= d/soft < 1:
                softg = 4/soft**2 * (16/3 * (d/soft) - 12 * (d/soft)**2 + 48/5 * (d/soft)**3 - 8/3 * (d/soft)**4 - 1/60 * (d/soft)**(-2))
                body.acc += - G * self.mass * softg * (dxdydz/d)
            elif d/soft >= 1:
                softg = 1/d**2
                body.acc += - G * self.mass * softg * (dxdydz/d)
            body.masseffect += self.mass

            rel_pos = np.array([not self.lims[0][i]<body.pos[i]<self.lims[1][i] for i in range(3)])
            close_edge = np.array([self.lims[0][i] if body.pos[i]<self.lims[0][i] else self.lims[1][i] if body.pos[i]>self.lims[1][i] else 0 for i in range(3)])
            if np.sqrt(sum((rel_pos*(body.pos - close_edge))**2)) <= 2 * body.h:
                body.density_comps += self.bodies
    
def Kick(all_bodies, dt):
    for body in all_bodies:
        body.kick(dt)

def Drift(all_bodies, dt):
    for body in all_bodies:
        body.drift(dt)

def Gravity(branch, all_bodies):
    for body in all_bodies:
        body.density_comps = []
    for body in all_bodies:
        #print('før')
        branch.walk(body,sam)
        #print(sam)
        #print('efter')

def Pressure(gas_bodies,dt):
    for body in gas_bodies:
        body.update_density(dt)
    #for body in gas_bodies:
    #    body.E_int(dt)
    #for body in gas_bodies:
    #    body.update_T()
        #body.update_k()
    # for body in gas_bodies:
    #     body.heat_transfer(dt)
    #for body in gas_bodies: 
    #    body.update_T()
        #body.update_k()
    for body in gas_bodies:
        body.find_pressure()
    for body in gas_bodies:
        body.find_sound_speed()
    for body in gas_bodies:
        body.find_nWs()
    for body in gas_bodies:
        body.find_visc_tensors()
    for body in gas_bodies:
        body.find_Pd_fac()
    for body in gas_bodies:
        body.pressure_gradient()

def Internal_energy(gas_bodies,dt):
    for body in gas_bodies:
        body.E_int(dt)
    #for body in gas_bodies:
    #    body.heat_transfer(dt)
    #for body in gas_bodies:
    #    body.update_T()

def Reset_acceleration(all_bodies):
    for body in all_bodies:
        body.a_old = body.acc
        body.acc = np.array([0., 0., 0.])

def Viscosity(gas_bodies,dt):
    for body in gas_bodies:
        #print('hallo')
        #print(body.prediction_velocity)
        body.find_prediction_velocity(dt)
        #print(body.prediction_velocity)
    for body in gas_bodies:
        body.a_visc()

def Smallest_dt(all_bodies):
    dt = 0
    n = -1
    for body in all_bodies:
        dtn,nn = body.find_dt()
        if nn > n:
            n = nn
            dt = dtn
    return dt, n

def Find_hs(gas_bodies):
    for gas in gas_bodies:
        gas.find_initial_h(gas_bodies)

def Update_hs(gas_bodies):
    for gas in gas_bodies:
        gas.update_h()


def Evolution(all_bodies, gas_bodies):
    t = 0
    dt,_ = Smallest_dt(all_bodies)

    while t + dt < tstep:
        t += dt
        Kick(all_bodies, dt)
        Drift(all_bodies, dt)
        Reset_acceleration(all_bodies)
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        Update_hs(gas_bodies)
        Gravity(branch, all_bodies) #This also finds the bodies for which the pressure is calculated
        Pressure(gas_bodies,dt)
        Viscosity(gas_bodies,dt)
        Internal_energy(gas_bodies,dt)
        Kick(all_bodies, dt)
        dt,_ = Smallest_dt(all_bodies)


    dt = tstep - t
    Kick(all_bodies, dt)
    Drift(all_bodies, dt)
    Reset_acceleration(all_bodies)
    branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
    Update_hs(gas_bodies)
    Gravity(branch, all_bodies) #This also finds the bodies for which the pressure is calculated
    Pressure(gas_bodies,dt)
    Viscosity(gas_bodies,dt)
    Kick(all_bodies, dt)




def find_dt(body):
    if np.sqrt(sum(body.acc**2)) == 0:
        dt_body = tstep
    else:
        dt_body = Cdt*np.sqrt(soft/(np.sqrt(sum(body.acc**2))))
    n = 0
    while tstep/2**n > dt_body:
        n += 1
    dt = tstep/2**n 
    return dt,n

def one_tstep(all_bodies, gas_bodies):
    branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
    
    n_bodies = {}
    for body in all_bodies:
        body.acc = np.array([0., 0., 0.])
        branch.walk(body)
        dt,n = find_dt(body)
        if n in n_bodies:
            n_bodies[n][0].append(body)
        else:
            n_bodies[n] = [[body],dt,1/2**n,0]
    for n in n_bodies:
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, n_bodies[n][0])
        n_bodies[n].append(branch)
        
        
    N=0.
    while ((n_bodies[max(n_bodies)][3])*2**N).is_integer() == False:
        N += 1
            
    in_sync = [n for n in n_bodies if n >= N]
    not_in_sync = [n for n in n_bodies if n < N]

    
    while n_bodies[max(n_bodies)][3] < 1:
        #first kick
        for n in in_sync:
            for body in n_bodies[n][0]:
                for m in in_sync:
                    if m > n:
                        body.kick(n_bodies[m][1])
                    else:
                        body.kick(n_bodies[n][1])
    
        #drift
        N=0.
        while (((n_bodies[max(n_bodies)][3] + n_bodies[max(n_bodies)][2])*2**N)).is_integer() == False:
            N += 1

        in_sync = [n for n in n_bodies if n >= N]
        not_in_sync = [n for n in n_bodies if n < N]
        
    
        for n in in_sync:
            for body in n_bodies[n][0]:
                body.drift(n_bodies[n][1])

        for n in in_sync:
            branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, n_bodies[n][0])
            n_bodies[n][4] = branch

        #updates
        Reset_acceleration(all_bodies)
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        Gravity(branch, all_bodies) #This also finds the bodies for which the pressure is calculated
        Pressure(gas_bodies)
        Viscosity(gas_bodies)
        
        
        #kick
        for n in in_sync:
            for body in n_bodies[n][0]:
                for m in in_sync:
                    if m > n:
                        body.kick(n_bodies[m][1])
                    else:
                        body.kick(n_bodies[n][1])
        
        #new groups
        new_n_bodies = {}
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        
        for n in in_sync:
            for body in n_bodies[n][0]:
                body.acc = np.array([0., 0., 0.])
                branch.walk(body)
                new_dt,new_n = find_dt(body)
                if new_n >= N:
                    if new_n in new_n_bodies:
                        new_n_bodies[new_n][0].append(body)
                    else:
                        new_n_bodies[new_n] = [[body], new_dt, 1/2**new_n, n_bodies[n][3] + n_bodies[n][2]]
                else:
                    if n in new_n_bodies:
                        new_n_bodies[n][0].append(body)
                    else:
                        new_n_bodies[n] = [[body], n_bodies[n][1], n_bodies[n][2], n_bodies[n][3] + n_bodies[n][2]]
        for n in not_in_sync:
            new_n_bodies[n] = n_bodies[n]
                    
        n_bodies = new_n_bodies
        
        in_sync = [n for n in n_bodies if n >= N]
        not_in_sync = [n for n in n_bodies if n < N]
        
        for n in in_sync:
            branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, n_bodies[n][0])
            n_bodies[n].append(branch)



def dice_data_dm(filename):
    pos_dm = readsnap(filename, 'pos', 'dm')
    vel_dm = readsnap(filename, 'vel', 'dm')
    vel_dm = vel_dm * (3.0856776 * 10**(16))**(-1) * 31557600 * 10**9
    mass_dm = readsnap(filename, 'mass', 'dm')
    print(pos_dm.shape)
    #plt.plot(pos[:,0],pos[:,1],'.')
    #plt.imshow(np.histogram2d(pos_dm[:,0],pos_dm[:,1],bins=70,range=[[-60,60],[-60,60]])[0],origin='lower',vmax=4)

    all_bodies_dm = []
    for i in range(len(pos_dm)):
        all_bodies_dm.append(Body(mass_dm[i], pos_dm[i][0], pos_dm[i][1], pos_dm[i][2], vel_dm[i][0], vel_dm[i][1], vel_dm[i][2], type='dm'))
    return all_bodies_dm

def dice_data_gas(filename):
    pos_gas = readsnap(filename, 'pos', 'gas')
    vel_gas = readsnap(filename, 'vel', 'gas')
    vel_gas = vel_gas * (3.0856776 * 10**(16))**(-1) * 31557600 * 10**9
    mass_gas = readsnap(filename, 'mass', 'gas')
    h = readsnap(filename, 'hsml', 'gas')
    h = 2 * h

    u = readsnap(filename, 'u', 'gas') #specific internal energy in km^2/s^2
    u = u * ((3.0856776 * 10**(16))**(-1) * 31557600 * 10**9)**2

    print(pos_gas.shape)
    #plt.plot(pos[:,0],pos[:,1],'.')
    plt.figure()
    plt.imshow(np.histogram2d(pos_gas[:,0], pos_gas[:,1], bins=70, range=[[-10,10], [-10,10]])[0], origin='lower', vmax=4)

    all_bodies_gas = []
    for i in range(len(pos_gas)):
        all_bodies_gas.append(Body(mass_gas[i], pos_gas[i][0], pos_gas[i][1], pos_gas[i][2], vel_gas[i][0], vel_gas[i][1], vel_gas[i][2], type='gas', Eint=u[i], h=h[i]))
    
    #for body in all_bodies_gas:
    #    print(body.pos)
    
    return all_bodies_gas




if density_test == True:
    all_bodies_gas = dice_data_gas(filename)
    branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies_gas)
    x=[]
    y=[]
    density=[]
    for body in all_bodies_gas:
                branch.update_density(body, all_bodies_gas)
                x.append(body.pos[0])
                y.append(body.pos[1])
                density.append(body.density)
    plt.figure()
    plt.imshow(np.histogram2d(x, y, bins=70, range=[[-10,10],[-10,10]], weights=density)[0], origin='lower', vmax=4)




if Position_dm == True:

    #all_bodies = rand_bodies_no_vel(N, 10000, start_box_xmin +10, start_box_xmax -10, start_box_ymin +10, start_box_ymax -10, start_box_zmin +10, start_box_zmax -10)
    #all_bodies = two_bodies(10000000000,-1,-1,-1,0,0,0,1,50,1,1,0,2,0)
    #all_bodies = keplervel(N,m)
    all_bodies = dice_data_dm(filename)
    u = 80
    all_data = []
    #S = SPH()
    for i in trange(tsteps):
        start = time()
    
        x = []
        y = []
        z = []
        
        histData = np.zeros([nBins, nBins, nBins])
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        #one_tstep(all_bodies)

        for body in all_bodies:
            #S.find_neighbours(body,all_bodies)
            branch.kdk(body)
            #branch.gas_kdk(body,all_bodies)
            x.append(body.pos[0])
            y.append(body.pos[1])
        
        if i % u == 0 or i==0:
            all_data.append(np.histogram2d(x,y,bins=70,range=[[-20,20],[-20,20]])[0])

    
    
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)



    def ani_hist(x):
        #l = x*u
        fig.clear()
        h = all_data[x]
        plt.imshow(h,origin='lower',vmax=4)
        plt.colorbar()
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('ininovar.gif',writer='Pillow')
    #plt.show()


if Position_gas == True:

    #all_bodies = rand_bodies_no_vel(N, 10000, start_box_xmin +10, start_box_xmax -10, start_box_ymin +10, start_box_ymax -10, start_box_zmin +10, start_box_zmax -10)
    #all_bodies = two_bodies(10000000000,-1,-1,-1,0,0,0,1,50,1,1,0,2,0)
    #all_bodies = keplervel(N,m)
    all_bodies = dice_data_gas(filename)
    u = 80
    all_data = []
    #S = SPH()
    for i in trange(tsteps):
        start = time()
    
        x = []
        y = []
        z = []
        
        histData = np.zeros([nBins, nBins, nBins])
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        #one_tstep(all_bodies)

        for body in all_bodies:
            #S.find_neighbours(body,all_bodies)
            #branch.kdk(body)
            branch.gas_kdk(body,all_bodies)
            x.append(body.pos[0])
            y.append(body.pos[1])
        
        if i % u == 0 or i==0:
            all_data.append(np.histogram2d(x,y,bins=70,range=[[-20,20],[-20,20]])[0])

    
    
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)



    def ani_hist(x):
        #l = x*u
        fig.clear()
        h = all_data[x]
        plt.imshow(h,origin='lower',vmax=4)
        plt.colorbar()
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('ininovar.gif',writer='Pillow')
    #plt.show()


if Position_gas_and_dm == True:

    #all_bodies = rand_bodies_no_vel(N, 10000, start_box_xmin +10, start_box_xmax -10, start_box_ymin +10, start_box_ymax -10, start_box_zmin +10, start_box_zmax -10)
    #all_bodies = two_bodies(10000000000,-1,-1,-1,0,0,0,1,50,1,1,0,2,0)
    #all_bodies = keplervel(N,m)
    all_bodies_gas = dice_data_gas(filename)
    all_bodies_dm = dice_data_dm(filename)
    all_bodies = [all_bodies_gas, all_bodies_dm]
    u = 80
    all_data_gas = []
    all_data_dm = []
    #S = SPH()
    for i in trange(tsteps):
        start = time()
    
        x_gas = []
        y_gas = []
        z_gas = []
        x_dm = []
        y_dm = []
        z_dm = []
        
        histData = np.zeros([nBins, nBins, nBins])
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
        #one_tstep(all_bodies)

        for body in sum(all_bodies,[]):
            #S.find_neighbours(body,all_bodies)
            #branch.kdk(body)
            branch.gas_kdk(body,all_bodies)
            x.append(body.pos[0])
            y.append(body.pos[1])
        
        if i % u == 0 or i==0:
            all_data.append(np.histogram2d(x,y,bins=70,range=[[-20,20],[-20,20]])[0])

    
    
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)



    def ani_hist(x):
        #l = x*u
        fig.clear()
        h = all_data[x]
        plt.imshow(h,origin='lower',vmax=4)
        plt.colorbar()
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('ininovar.gif',writer='Pillow')
    #plt.show()


if Energy1 == True:
    all_bodies = dice_data_gas(filename)
    gas_bodies = all_bodies
    u = 80
    all_data = []
    all_energy = []

    dt,_ = Smallest_dt(all_bodies)
    branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
    Find_hs(gas_bodies)
    Gravity(branch, all_bodies) #This also finds the bodies for which the pressure is calculated
    Pressure(gas_bodies,dt)
    print('here')
    for i in trange(tsteps):
        start = time()
    
        x = []
        y = []
        z = []
        
        histData = np.zeros([nBins, nBins, nBins])
        
        Evolution(all_bodies,gas_bodies)
        #one_tstep(all_bodies, gas_bodies)

        for body in all_bodies:
            x.append(body.pos[0])
            y.append(body.pos[1])
        
        energy = 0

        for n in range(len(all_bodies)):
            b1 = all_bodies[n]
            v1 = np.sqrt(sum(b1.v**2))
            b1kin = 1/2 * b1.mass * v1**2
            b1pot = 0
            for m in range(n+1,len(all_bodies)):
                if m != n:
                    b2 = all_bodies[m]
                    d = np.sqrt(sum((b1.pos - b2.pos)**2))
                    b1pot += (G*b1.mass*b2.mass)/d

            energy += b1kin + b1pot + b1.Eint

        all_energy.append(energy)
    
    plt.figure()
    plt.plot(all_energy)
    plt.show()


def sims(dem_bodies,branch):
    arr = []
    for body in dem_bodies:
        branch.kdk(body)
        arr.append(body)
    dem_bodies = arr


def main():
    all_bodies = dice_data()
    u = int(tsteps/10)
    lens = len(all_bodies)//(cpu_count-1)
    body_lists = []
    for i in range(0,len(all_bodies),lens):
        body_lists.append(all_bodies[i:i+lens])
    all_data = []

    for i in trange(tsteps):
        x = []
        y = []
        branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)

        pool = Pool(cpu_count)
        for list in body_lists:
            pool.apply_async(sims,args=(list,branch))
        
        pool.close()
        pool.join()

if Multi_position == True:
    if __name__ == '__main__':
        main()


if test2 == True:
    all_bodies = dice_data_gas(filename)
    gas_bodies = all_bodies
    u = 80
    all_data = []

    branch = Branch(start_box_xmin, start_box_xmax, start_box_ymin, start_box_ymax, start_box_zmin, start_box_zmax, all_bodies)
    Gravity(branch, all_bodies) #This also finds the bodies for which the pressure is calculated
    Pressure(gas_bodies)
    for i in trange(tsteps):
        start = time()
    
        x = []
        y = []
        z = []
        
        histData = np.zeros([nBins, nBins, nBins])
        
        Evolution(all_bodies,gas_bodies)
        #one_tstep(all_bodies, gas_bodies)

        for body in all_bodies:
            x.append(body.pos[0])
            y.append(body.pos[1])
        
        if i % u == 0 or i==0:
            all_data.append(np.histogram2d(x,y,bins=70,range=[[-50,50],[-50,50]])[0])

    
    
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)



    def ani_hist(x):
        #l = x*u
        fig.clear()
        h = all_data[x]
        plt.imshow(h,origin='lower',vmax=4)
        plt.colorbar()
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('ininovar2.gif',writer='Pillow')
    #plt.show()


def Evolution_hydro(gas_bodies,tstep):
    t = 0
    dt,_ = Smallest_dt(gas_bodies)
    while t + dt < tstep:
        print('test')
        t += dt

        t1 = time()
        Kick(gas_bodies, dt)
        t2 = time()
        Drift(gas_bodies, dt)
        t3 = time()
        Reset_acceleration(gas_bodies)
        t4 = time()
        Update_hs(gas_bodies)
        t5 = time()
        for body in gas_bodies:
            body.test_find_comps(gas_bodies)
        t6 = time()
        Pressure(gas_bodies,dt)
        t7 = time()
        Viscosity(gas_bodies,dt)
        t8 = time()
        Internal_energy(gas_bodies,dt)
        t9 = time()
        Kick(gas_bodies, dt)
        t10 = time()
        dt,_ = Smallest_dt(gas_bodies)

        #print(" kick: " + str(t2-t1) + " drift: " + str(t3-t2) + " reset: " + str(t4-t3) + " updateh: " + str(t5-t4) + " findcomps: " + str(t6-t5) + " Pressure: " + str(t7-t6) + " visc: " + str(t8-t7) + " internal: " + str(t9-t8))




    dt = tstep - t
    Kick(gas_bodies, dt)
    Drift(gas_bodies, dt)
    Reset_acceleration(gas_bodies)
    Update_hs(gas_bodies)
    for body in gas_bodies:
        body.test_find_comps(gas_bodies)
    Pressure(gas_bodies,dt)
    Viscosity(gas_bodies,dt)
    Kick(gas_bodies, dt)


def analytical_shock_tube(rho_left,P_left,c_left,rho_right,P_right,c_right):
    gamma = 1.4
    u2limx = np.linspace(0.38,0.42,20)
    t = 0.1
    u2 = 2/(gamma+1) * (c_left + (u2limx - 0.5)/t)
    rho2 = rho_left * ( 1 - (gamma-1)/2 * u2/c_left )**(2/(gamma-1))
    beta2 = (gamma-1)/(2*gamma)
    Lambda = (gamma-1)/(gamma+1)
    
    P3_temp = np.linspace(0,1,100)
    u4 = (P3_temp - P_right) * np.sqrt( (1-Lambda)/(rho_right*(P3_temp + Lambda*P_right)) )
    u3 = (P_left**beta2 - P3_temp**beta2)* np.sqrt( ((1-Lambda**2)*P_left**(1/gamma))/(Lambda**2 * rho_left) )
    indexformin = np.argmin(abs(u3-u4))
    print(abs(u3-u4))
    P3 = P3_temp[indexformin]
    rho3 = rho_left*(P3/P_left)**(1/gamma)
    P4 = P3
    rho4 = rho_right*(P4 + Lambda*P_right)/(P_right + Lambda*P4)

    xleft = np.linspace(0,0.383,20)

    x3 = np.linspace(0.4,0.54,20)
    x4 = np.linspace(0.54,0.63,20)
    xright = np.linspace(0.63,1,20)

    rhop1 = np.ones(20)*rho_left
    rhop2 = rho2
    rhop3 = np.ones(20)*rho3
    rhop4 = rho4*np.ones(20)
    rhop5 = rho_right*np.ones(20)

    #plt.figure()
    plt.plot(xleft,rhop1)
    plt.plot(u2limx,rhop2)
    plt.plot(x3,rhop3)
    plt.plot(x4,rhop4)
    plt.plot(xright,rhop5)
    #plt.show()

N_target = 13
def schocktest():
    N = 200
    L = 1

    a = 1
    b = 0.033

    x1 = []
    for i in range(N):
        x1.append(i + a*np.sin(b*i))

    x1 = x1/(np.max(x1)+1)

    #x1 = np.linspace(0,0.4,4*N)
    #print(x1[1]-x1[0])
    #x2 = np.linspace(0.401,0.5,N)
    #print(x2[1]-x2[0])
    #x3 = np.linspace(0.501,1,5*N)
    #print(x3[1]-x3[0])
    #x1 = np.random.normal(0.5,0.4,350)
    #x1 = x1[x1<1]
    #x1 = x1[x1>0]
    #x1.sort()
    T = 0.9/(gamma-1)
    bodies = []
    mass = 0.000059
    fac = 1
    for i in range(len(x1)):
        body = Body(mass,x1[i],0.,0.,0.,0.,0.,type='gas',Eint=T,h=2*fac)
        bodies.append(body)
    '''for i in range(len(x2)):
        body = Body(mass,x2[i],0.,0.,0.,0.,0.,type='gas',Eint=5*T,h=2*fac)
        bodies.append(body)
    for i in range(len(x3)):
        body = Body(mass,x3[i],0.,0.,0.,0.,0.,type='gas',Eint=T,h=2*fac)
        bodies.append(body)'''
    '''for i in range(3*N):
        body = Body(1,x3[i],0.,0.,0.,0.,0.,type='gas',Eint=T,h=2*fac)
        bodies.append(body)'''
    #for i in range(len(x1)):
    #    body = Body(mass,x1[i],0.,0.,0.,0.,0.,type='gas',Eint=T,h=2*fac)
    #    bodies.append(body)
    
    t = 2
    tsteps = 200
    tstep = float(t/tsteps)

    Ps = []
    vs = []
    ds = []
    xs = []
    for body in bodies:
        body.find_initial_h(bodies)
        body.h = body.h/2
    for body in bodies:
        body.test_find_comps(bodies)
    dt,_ = Smallest_dt(bodies)
    Pressure(bodies,dt)
    '''plt.figure()
    for body in bodies:
        plt.scatter(body.pos[0],body.P)'''

    Viscosity(bodies,dt)

    u = 2


    #Analytical stuff
    rho_ana = []
    sound_speeds = []
    xs_ana = []
    for body in bodies:
        rho_ana.append(body.density)
        sound_speeds.append(body.sound_speed)
        xs_ana.append(body.pos[0])
    rho_ana = np.array(rho_ana)
    sound_speeds = np.array(sound_speeds)
    sound_speeds = np.mean(sound_speeds)
    xs_ana = np.array(xs_ana)

    

    


    '''maxrho = np.max(rho_ana)
    minrho = np.min(rho_ana)
    a = maxrho
    xlin = np.linspace(0,1,100)
    ys = np.sin(xlin*2*np.pi-np.pi/2)*0.48'''

    xlin = np.linspace(0,1,100)

    drho0 = 0.054
    k = 2*np.pi
    c_s = sound_speeds
    omega = k*c_s
    t = 0

    ys = (-drho0*k**2*c_s**2 * np.cos(k*xlin + omega*t - 0.1) -drho0*k**2*c_s**2 * np.cos(k*xlin - omega*t - 0.1))/2

    yss = []
    #for i in range(tsteps):
        #t = i*tstep

        #ys = (-drho0*k**2*c_s**2 * np.cos(k*xlin + omega*t) -drho0*k**2*c_s**2 * np.cos(k*xlin - omega*t))/2

        #yss.append(ys)

    '''fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)


    def ani_hist(x):
        #l = x*u
        fig.clear()
        #plt.plot(xs[x],Ps[x]/max(Ps[x]))
        #plt.plot(xs[x],vs[x]/max(vs[x]))
        plt.ylim(-0.7,0.7)
        plt.xlim(0,1)
        plt.plot(xlin,yss[x],'.')
        
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('soundwavetest.gif',writer='Pillow')
    #plt.show()'''




    '''plt.plot(xlin,ys+14.85)
    plt.show()'''



    for i in range(tsteps):
        #print(bodies[1].h)
        p = []
        v = []
        d = []
        x = []

        for body in bodies:
            p.append(body.P)
            v.append(abs(body.v[0]))
            d.append(body.density)
            x.append(body.pos[0])


        print(i)
        Evolution_hydro(bodies,tstep)


        for body in bodies:
            body.pos[1] = 0
            body.pos[2] = 0
            if body.pos[0] < 0:
                body.pos[0] += L
            if body.pos[0] > L:
                body.pos[0] += -L

        print(bodies[1].density)

        Ps.append(p)
        vs.append(v)
        ds.append(d)
        xs.append(x)

        '''plt.figure()
        plt.plot(x,d,'.')
        plt.show()'''



        #plt.figure()
        #analytical_shock_tube(shock_tube_initial[0]+0.05,shock_tube_initial[1]+0.05,shock_tube_initial[2],shock_tube_initial[3],shock_tube_initial[4],shock_tube_initial[5])
        #plt.plot(xs[-1],ds[-1],'.')
        #plt.ylabel('Densitet')
        #plt.xlabel('x')
        #plt.figure()
        #plt.plot(xs[-1],Ps[-1],'.')
        #plt.ylabel('tryk')
        #plt.figure()
        #plt.plot(xs[-1],vs[-1],'.')
        #plt.ylabel('hastighed')
        #plt.show()
        #plt.savefig('soundwave.png')

        t = i*tstep

        ys = (-drho0*k**2*c_s**2 * np.cos(k*xlin + omega*t + 0.1) -drho0*k**2*c_s**2 * np.cos(k*xlin - omega*t + 0.1))/2

        yss.append(ys)


        if i == 5:
            print(i)
            print(t)
            plt.figure()
            plt.plot(x,d,'.')
            plt.plot(xlin,ys+18.6)
            plt.ylim(10,25)
            plt.xlabel('x')
            plt.ylabel('Densitet')
            plt.savefig(str(t)+'i'+'5.png')

        if i == 75:
            print(i)
            print(t)
            plt.figure()
            plt.plot(x,d,'.')
            plt.ylim(10,25)
            plt.plot(xlin,ys+18.6)
            plt.xlabel('x')
            plt.ylabel('Densitet')
            plt.savefig(str(t)+'i'+'75.png')

        if i == 125:
            print(i)
            print(t)
            plt.figure()
            plt.plot(x,d,'.')
            plt.ylim(10,25)
            plt.plot(xlin,ys+18.6)
            plt.xlabel('x')
            plt.ylabel('Densitet')
            plt.savefig(str(t)+'i'+'125.png')

        if i == 170:
            print(i)
            print(t)
            plt.figure()
            plt.plot(x,d,'.')
            plt.ylim(10,25)
            plt.plot(xlin,ys+18.6)
            plt.xlabel('x')
            plt.ylabel('Densitet')
            plt.savefig(str(t)+'i'+'170.png')

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)


    def ani_hist(x):
        #l = x*u
        fig.clear()
        #plt.plot(xs[x],Ps[x]/max(Ps[x]))
        #plt.plot(xs[x],vs[x]/max(vs[x]))
        plt.ylim(10,25)
        plt.xlim(0,1)
        plt.plot(xs[x],ds[x],'.')
        plt.plot(xlin,yss[x]+18.6)
        
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(tsteps/u),interval=0.0001)

    anim.save('2soundwave.gif',writer='Pillow')
    #plt.show()


schocktest()