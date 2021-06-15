import numpy as np
from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, prange, types, typeof
from numba.experimental import jitclass
from numba.typed import List
from numba import jit
from pygadgetreader import *
import math
import time
import matplotlib.animation as ani

import matplotlib.pyplot as plt

np.seterr(invalid='raise')





'''from collections import Counter
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()'''













filename = "stars6000_arms.g2"


G = 6.67428*10**(-11)*(3.0856776*10**(19))**(-3)*10**(10)*1.9891*10**(30)*(31557600*10**9)**2
N = 100
m = 1
k_B = 1.38064852*10**(-23)   *(3.24077929*10**(-20))**2   *(5.02739933*10**(-31)*10**(-10))   *(3.15576*10**(16))**2
m_u = 10**(-10) * (5.02739933 * 10**(-31)) * (1.6605402 * 10**(-27))
R_g = k_B/m_u
mu = 0.6 #Sagde Troels?


#Viscosity
alpha = 0.9
beta = 2 * alpha
eta = np.sqrt(0.01)

tsteptest = 0

thetafactor = 0.3
soft = 1
Cdt = 0.2
gamma = 1.001 # 5/3
N_target = 30

t = 0.5
tsteps = 100
tstep = t/tsteps

start_box_xmin = -100
start_box_xmax = 100
start_box_ymin = -100
start_box_ymax = 100
start_box_zmin = -100
start_box_zmax = 100


def make_matrix(mass, pos, v, type, Eint):
    b=[[Eint[i]]for i in range(len(mass))]
    a=[[(2 * Eint[i])/(3 * R_g)]for i in range(len(mass))]
    matrix = [[mass[i], pos[i][0], pos[i][1], pos[i][2], v[i][0], v[i][1], v[i][2], 0., 0., 0., type, math.nan, math.nan, Eint[i], (2 * Eint[i])/(3 * R_g), math.nan, math.nan, 0., 0., 0., 0., 0., 0., math.nan, i] for i in range(len(mass))]
    return matrix

def dice_data_dm(filename):
    pos_dm = readsnap(filename, 'pos', 'dm')
    vel_dm = readsnap(filename, 'vel', 'dm')
    vel_dm = vel_dm * (3.0856776 * 10**(16))**(-1) * 31557600 * 10**9
    mass_dm = readsnap(filename, 'mass', 'dm')

    u = np.array([0. for _ in range(len(mass_dm))])

    matrix = make_matrix(mass_dm, pos_dm, vel_dm, 0., u)

    return matrix

def dice_data_stars(filename):
    pos_dm = readsnap(filename, 'pos', 'disk')
    vel_dm = readsnap(filename, 'vel', 'disk')
    vel_dm = vel_dm * (3.0856776 * 10**(16))**(-1) * 31557600 * 10**9
    mass_dm = readsnap(filename, 'mass', 'disk')

    u = np.array([0. for _ in range(len(mass_dm))])

    matrix = make_matrix(mass_dm, pos_dm, vel_dm, 0., u)

    return matrix

def dice_data_gas(filename):
    pos_gas = readsnap(filename, 'pos', 'gas')
    vel_gas = readsnap(filename, 'vel', 'gas')
    vel_gas = vel_gas * (3.0856776 * 10**(16))**(-1) * 31557600 * 10**9
    mass_gas = readsnap(filename, 'mass', 'gas')
    h = readsnap(filename, 'hsml', 'gas')
    h = 2 * h

    u = readsnap(filename, 'u', 'gas') #specific internal energy in km^2/s^2
    dens = readsnap(filename, "rho",'gas') # H/cm^3

    u = (5/3-1)/(gamma-1) * u
    u = u * ((3.0856776 * 10**(16))**(-1) * 31557600 * 10**9)**2
    

    matrix = make_matrix(np.float32(mass_gas), pos_gas, vel_gas, 1., u)

    return matrix

def make_lists(N):
    lists = [[[0. for _ in range(N)] for _ in range(6)] for _ in range(N)]
    return lists


gas_bodies = dice_data_gas(filename)
gas_m = gas_bodies[0][0]
gas_N = len(gas_bodies)
dm_bodies = dice_data_dm(filename)
dm_m = dm_bodies[0][0]
dm_n = len(dm_bodies)
#Sætter bare dm og stjerner sammen fordi de gør det samme.
star_bodies = dice_data_stars(filename)
star_N = len(star_bodies)
star_m = star_bodies[0][0]
dm_bodies = dm_bodies + star_bodies
dm_N = len(dm_bodies)
all_bodies = np.array(gas_bodies+dm_bodies)
all_N = len(all_bodies)
body_length = len(all_bodies[0])
all_lists = make_lists(len(all_bodies))


total_mass = 0

for i in range(len(all_bodies)):
    all_bodies[i][-1] = i
    total_mass += all_bodies[i][0]

spec_branch = [
    ('xmin',float64),
    ('xmax',float64),
    ('ymin',float64),
    ('ymax',float64),
    ('zmin',float64),
    ('zmax',float64),
    ('btc',float64[:,:]),
    ('N',int64),
    ('child_N',int64[:]),
    ('xmid',float64),
    ('ymid',float64),
    ('zmid',float64),
    ('child_bodies',float64[:,:]),
    ('bo0',float64[:,:]),
    ('bo1',float64[:,:]),
    ('bo2',float64[:,:]),
    ('bo3',float64[:,:]),
    ('bo4',float64[:,:]),
    ('bo5',float64[:,:]),
    ('bo6',float64[:,:]),
    ('bo7',float64[:,:]),
    ('nr_childs',int64),
    ('COM',float64[:]),
    ('Mass',float64),
    ('index',int64),
    ('child_indicis',int64[:]),
    ('l',float64)
]

@jitclass(spec_branch)
class Branch():
    def __init__(self,xmi,xma,ymi,yma,zmi,zma,btc):
        self.xmin = xmi
        self.xmax = xma
        self.ymin = ymi 
        self.ymax = yma 
        self.zmin = zmi 
        self.zmax = zma 
        self.xmid = (self.xmax+self.xmin)/2
        self.ymid = (self.ymax+self.ymin)/2
        self.zmid = (self.zmax+self.zmin)/2
        self.l = abs(xma-xmi)
        self.btc = btc
        self.N = len(btc)
        self.child_N = np.zeros(8,dtype=np.int64)
        self.bo0 = -np.ones((self.N,body_length))
        self.bo1 = -np.ones((self.N,body_length))
        self.bo2 = -np.ones((self.N,body_length))
        self.bo3 = -np.ones((self.N,body_length))
        self.bo4 = -np.ones((self.N,body_length))
        self.bo5 = -np.ones((self.N,body_length))
        self.bo6 = -np.ones((self.N,body_length))
        self.bo7 = -np.ones((self.N,body_length))
        nr_childs = 0


        # Calculating COM and Mass
        self.COM = np.zeros(3)
        xyz = np.zeros(3)
        self.Mass = 0
        if self.N > 0:
            for body in btc:
                xyz += body[1:4] * body[0]
                self.Mass += body[0]
            self.COM = xyz / self.Mass

        if self.N > 1:
            # Figure out what child to put bodies into
            i0 = 0
            i1 = 0
            i2 = 0
            i3 = 0
            i4 = 0
            i5 = 0
            i6 = 0
            i7 = 0

            for i in range(len(btc)):
                body = btc[i]

                # Find octant
                x1 = self.xmin < body[1] < self.xmid
                x2 = self.xmid < body[1] < self.xmax
                y1 = self.ymin < body[2] < self.ymid
                y2 = self.ymid < body[2] < self.ymax
                z1 = self.zmin < body[3] < self.zmid
                z2 = self.zmid < body[3] < self.zmax

                octant = -1
                if x1 and y1 and z1:
                    octant = 0
                    self.bo0[i0] = body
                    i0 += 1
                elif x1 and y1 and z2:
                    octant = 1
                    self.bo1[i1] = body
                    i1 += 1
                elif x1 and y2 and z1:
                    octant = 2
                    self.bo2[i2] = body
                    i2 += 1
                elif x1 and y2 and z2:
                    octant = 3
                    self.bo3[i3] = body
                    i3 += 1
                elif x2 and y1 and z1:
                    octant = 4
                    self.bo4[i4] = body
                    i4 += 1
                elif x2 and y1 and z2:
                    octant = 5
                    self.bo5[i5] = body
                    i5 += 1
                elif x2 and y2 and z1:
                    octant = 6
                    self.bo6[i6] = body
                    i6 += 1
                elif x2 and y2 and z2:
                    octant = 7
                    self.bo7[i7] = body
                    i7 += 1

                if octant != -1:
                    self.child_N[octant] += 1

            # Check which children exist
            for child in self.child_N:
                if child != 0:
                    self.nr_childs += 1

            # What bodies are in which octant
            self.bo0 = self.bo0[0:i0]
            self.bo1 = self.bo1[0:i1]
            self.bo2 = self.bo2[0:i2]
            self.bo3 = self.bo3[0:i3]
            self.bo4 = self.bo4[0:i4]
            self.bo5 = self.bo5[0:i5]
            self.bo6 = self.bo6[0:i6]
            self.bo7 = self.bo7[0:i7]

            
@njit
def Children(b):
    children = []
    if b.child_N[0] != 0:
        children.append(Branch(b.xmin,b.xmid,b.ymin,b.ymid,b.zmin,b.zmid,b.bo0))
    if b.child_N[1] != 0:
        children.append(Branch(b.xmin,b.xmid,b.ymin,b.ymid,b.zmid,b.zmax,b.bo1))
    if b.child_N[2] != 0:
        children.append(Branch(b.xmin,b.xmid,b.ymid,b.ymax,b.zmin,b.zmid,b.bo2))
    if b.child_N[3] != 0:
        children.append(Branch(b.xmin,b.xmid,b.ymid,b.ymax,b.zmid,b.zmax,b.bo3))
    if b.child_N[4] != 0:
        children.append(Branch(b.xmid,b.xmax,b.ymin,b.ymid,b.zmin,b.zmid,b.bo4))
    if b.child_N[5] != 0:
        children.append(Branch(b.xmid,b.xmax,b.ymin,b.ymid,b.zmid,b.zmax,b.bo5))
    if b.child_N[6] != 0:
        children.append(Branch(b.xmid,b.xmax,b.ymid,b.ymax,b.zmin,b.zmid,b.bo6))
    if b.child_N[7] != 0:
        children.append(Branch(b.xmid,b.xmax,b.ymid,b.ymax,b.zmid,b.zmax,b.bo7))

    return children


@njit
def MakeTree(bodies):
    # Now we make all of the branches that we need and throw them in an array
    root = Branch(-100,100,-100,100,-100,100,bodies)
    all_nodes = [root,]
    nr_branches = 1

    count = 0
    length = 1
    while count < length:
        parent = all_nodes[count]
        if parent.N > 1:
            new_branches = Children(parent)
            child_indicis = []
            for new in new_branches:
                #if new.N == 0:
                #    print('ALARM,')
                all_nodes.append(new)
                child_indicis.append(length)
                length += 1
            parent.child_indicis = np.array(child_indicis)
                
        count += 1

    # Find which nodes to calculate force
    body_force_nodes = []
    body_force_distances= []
    body_force_dxdydz = []
    for body in bodies:
        branches = [root,]
        force_nodes = []
        force_distances = []
        force_dxdydz = []
        count = 0
        length = 1
        while count < length:
            node = branches[count]
            dxdydz = body[1:4] - node.COM
            l = node.l
            d = np.sqrt(np.sum((dxdydz**2)))

            
            # DET HER ER VIRKELIGT LANGSOOOOOMT
            bodyinnode = False
            for i in range(len(node.btc)):
                if node.btc[i][-1] == body[-1]:
                    bodyinnode = True

            if bodyinnode:
                if node.N > 1:
                    for index in node.child_indicis:
                        branches.append(all_nodes[index])
                        length += 1
            else:
                theta = l/d
                if node.N > 1:
                    if theta > thetafactor:
                        for index in node.child_indicis:
                            branches.append(all_nodes[index])
                            length += 1
                    else:
                        force_nodes.append(node)
                        force_distances.append(d)
                        force_dxdydz.append(dxdydz)
                else:
                    force_nodes.append(node)
                    force_distances.append(d)
                    force_dxdydz.append(dxdydz)

            count += 1
        body_force_nodes.append(force_nodes)
        body_force_distances.append(force_distances)
        body_force_dxdydz.append(force_dxdydz)

    # Now we calculate the force
    #all_density_comps = []
    #all_body_comps = np.zeros(len(gas_bodies))
    all_density_comps = -1 * np.ones((gas_N,2*N_target),np.int64)
    #np.array([[-1 for _ in range(2*N_target)] for _ in range(gas_N)])
    #np.zeros((gas_N,gas_N))
    overflow = 0
    acc_boxes = np.zeros(all_N)
    for i in range(all_N):
        body = bodies[i]
        #density_comps = []
        acc = np.zeros(3)
        #body_comps = np.zeros((len(gas_bodies),len(gas_bodies)))
        total_mass_effect = 0
        no = 0
        for j in range(len(body_force_nodes[i])):
            node = body_force_nodes[i][j]
            total_mass_effect += node.Mass
            d = body_force_distances[i][j]
            dxdydz = body_force_dxdydz[i][j]

            if 0 <= d/soft < 1/2:
                softg = 4/soft**2 * (8/3 * (d/soft) - 48/5 * (d/soft)**3 + 8 * (d/soft)**4)
            elif 1/2 <= d/soft < 1:
                softg = 4/soft**2 * (16/3 * (d/soft) - 12 * (d/soft)**2 + 48/5 * (d/soft)**3 - 8/3 * (d/soft)**4 - 1/60 * (d/soft)**(-2))
            else:
                softg = 1/d**2
            acc += -G * node.Mass * softg * dxdydz/d
            
            # When we are here we might as well find the neighbours for SPH
            if body[10] == 1.:
                distance = np.zeros(2*N_target)
                max_distance = 0.
                lims = [[node.xmin, node.ymin, node.zmin], [node.xmax, node.ymax, node.zmax]]
                rel_pos = np.array([not lims[0][j]<body[1:4][j]<lims[1][j] for j in range(3)])

                close_edge = np.array([lims[0][j] if body[1:4][j]<lims[0][j] else lims[1][j] if body[1:4][j]>lims[1][j] else 0. for j in range(3)])

                if np.sqrt(np.sum((rel_pos*(body[1:4] - close_edge))**2)) <= 2 * body[15]:

                #if d <= 2*body[15]:
                    for comp in node.btc:
                        #density_comps.append(comp)
                        #body_comps[comp[-1]] = 1
                        g = 0
                        for n in all_density_comps[i]:
                            if int(n) == comp[24]:
                                g += 1
                        if g == 0:
                            d = np.sqrt(np.sum((body[1:4]-comp[1:4])**2))
                            if d <= 2*body[15] and comp[10] == 1.:
                                #print('dist',np.sqrt(np.sum((body[1:4]-comp[1:4])**2)),2*body[15])
                                if no <= 2*N_target:
                                    all_density_comps[i][no] = comp[24]
                                    distance[no] = d
                                    max_distance = max(d,max_distance)
                                    '''else:
                                        overflow += 1
                                        if no == 2*N_target +1:
                                            print(body[-1])
                                            print(body)'''
                                else:
                                    if d < max_distance:
                                        j = np.argmax(distance)
                                        distance[j] = d
                                        all_density_comps[i][j] = comp[24]
                                        max_distance = np.max(distance)
                                no += 1
        bodies[i][7:10] = acc
        acc_boxes[i] = np.sqrt(np.sum(acc**2))
        '''if no > 2*N_target:
            print(body[-1])
            print(body)
            print(no)'''
        '''print(total_mass_effect)
        print(total_mass - body[0])'''
        #all_density_comps.append(density_comps)
        #all_density_comps[i] = body_comps
    #if overflow > 0:
    #    print(overflow)

    '''acc_brutes = np.zeros(all_N)
    for i in range(all_N):
        acc_brute = np.zeros(3)
        body1 = bodies[i]
        for j in range(all_N):
            if i != j:
                body2 = bodies[j]
                dxdydz = body1[1:4] - body2[1:4]
                d = np.sqrt(np.sum((dxdydz)**2))

                if 0 <= d/soft < 1/2:
                    softg = 4/soft**2 * (8/3 * (d/soft) - 48/5 * (d/soft)**3 + 8 * (d/soft)**4)
                elif 1/2 <= d/soft < 1:
                    softg = 4/soft**2 * (16/3 * (d/soft) - 12 * (d/soft)**2 + 48/5 * (d/soft)**3 - 8/3 * (d/soft)**4 - 1/60 * (d/soft)**(-2))
                else:
                    softg = 1/d**2

                acc_brute += -G * body2[0] * softg * dxdydz/d
        acc_brutes[i] = np.sqrt(np.sum(acc_brute**2))

    y = (acc_boxes - acc_brutes)/acc_brutes
    print(y)
    x = acc_brutes'''





    return all_density_comps #bodies, 


# Now it is time for SPH and evolution in time
@njit
def Kick(bodies, dt):
    for i in range(all_N):
        bodies[i][4:7] = bodies[i][4:7] + 0.5 * dt * bodies[i][7:10]


@njit
def Drift(bodies,dt):
    for i in range(all_N):
        bodies[i][1:4] = bodies[i][1:4] + dt * bodies[i][4:7]


@njit
def find_dt(bodies):
    dt_body_use = tstep
    for body in bodies:
        a = np.sqrt(np.sum((body[7:10])**2))
        if a == 0.:
            dt_body = tstep
        else:
            dt_body = Cdt * np.sqrt( soft/a )
        if dt_body < dt_body_use:
            dt_body_use = dt_body

        if body[11] != math.nan:
            dt_sph = Cdt * 3 / np.sqrt(8 * np.pi * G * body[11])
        if dt_sph < dt_body_use:
            dt_body_use = dt_sph

    
    return dt_body_use


def find_initial_h(bodies):
    gas = []
    for i in range(all_N):
        body = bodies[i]
        if body[10] == 1.:
            gas.append(body)
    #dist = np.max(np.sqrt(np.sum(np.array(gas[:][1:4])**2,axis=1)))
    for i in range(all_N):
        if bodies[i][10] == 1.:
            #s = sorted(gas, key=lambda x: np.sqrt(sum((np.array(bodies[i][1:4]) - np.array(x[1:4]))**2)))
            #print('start')
            #for b in s:
            #    print(np.sqrt(sum((np.array(bodies[i][1:4]) - np.array(b[1:4]))**2)))
            #bodies[i][15] = 0.5 * np.sqrt(sum((np.array(bodies[i][1:4]) - np.array(s[N_target][1:4]))**2))
            #bodies[i][15] = dist/all_N**(1/3) *N_target**(1/3) 
            bodies[i][15] = soft
    #return bodies


@njit
def find_W(body1,body2, h):
    d = np.sqrt(np.sum((body1[1:4]-body2[1:4])**2))

    ratio = d/h
    if 0 <= ratio <= 1:
        W = 1 / (np.pi * h**3) * (1 - 3/2 * (ratio)**2 + 3/4 * (ratio)**3)
    elif 1 < ratio < 2:
        W = 1 / (4 * np.pi * h**3) * (2 - (ratio))**3
    else:
        W = 0
    return W


@njit
def update_density(bodies,comps):
    for i in range(gas_N):
        bodies[i][11] = 0

        for j in comps[i]:
            if j >= 0:
                bodies[i][11] += bodies[j][0] * (find_W(bodies[i],bodies[j],bodies[j][15])+find_W(bodies[i],bodies[j],bodies[i][15]))
        bodies[i][11] += bodies[i][0] * (find_W(bodies[i],bodies[i],bodies[i][15]) + find_W(bodies[i],bodies[i],bodies[i][15]))
        bodies[i][11] = bodies[i][11]/2

@njit
def find_pressure(bodies):
    for i in range(gas_N):
        bodies[i][12] = (gamma-1)*bodies[i][11]*bodies[i][13]


@njit
def find_Pd_fac(bodies,comps):
    all_Pd_fac = np.zeros((gas_N,gas_N))
    for i in range(gas_N):
        for j in comps[i]:
            if j >= 0:
                all_Pd_fac[i][j] = np.sqrt(bodies[i][12]*bodies[j][12])/(bodies[i][11]*bodies[j][11])
    return all_Pd_fac


@njit
def nW(body1,body2,h):
    d = np.sqrt(np.sum((body1[1:4]-body2[1:4])**2))
    dxdydz = body1[1:4] - body2[1:4]
    ratio = d/h 
    if 0 <= ratio <= 1:
        nW =  1/(np.pi * h**4) * ((-3 * ratio) + (9 * d**2 /(4 * h**2)))  
    elif 1 < ratio <= 2:
        nW = - 3/(4 * np.pi * h**4) * (2 - ratio)**2                             
    else:
        nW = 0
    return nW*dxdydz/d

@njit
def find_nWs(bodies, comps):
    nW_h_i = np.zeros((gas_N,gas_N,3))
    nW_h_j = np.zeros((gas_N,gas_N,3))
    for i in range(gas_N):
        for j in comps[i]:
            
            if j >= 0:
                nW_h_i[i][j] = nW(bodies[i],bodies[j],bodies[i][15])
                nW_h_j[i][j] = nW(bodies[i],bodies[j],bodies[j][15])
            # if i == 366:
            #     #if nW_h_i[i][j][0] != 0.0:
            #     '''if math.isnan(nW_h_i[i][j][0]):
            #         print(nW_h_i[i][j])
            #         a = nW(bodies[i],bodies[j],bodies[i][15],True)
            #         #print(bodies[366])'''
    return nW_h_i, nW_h_j


@njit
def pressure_gradient(bodies, nW_h_i, nW_h_j, Pd_fac, comps):
    for i in range(gas_N):
        acc = bodies[i][7:10]
        for j in comps[i]:
            
            if j >= 0:
                P_grad = bodies[j][0] * Pd_fac[i][j] * (nW_h_i[i][j] + nW_h_j[i][j])
                acc -= P_grad
        bodies[i][7:10] = acc
        


@njit
def find_prediction_velocity(bodies,dt):
    for i in range(gas_N):
        bodies[i][17:20] = bodies[i][4:7] + 0.5*dt*bodies[i][20:23]


@njit
def reset_acceleration(bodies):
    for i in range(all_N):
        bodies[i][20:23] = bodies[i][7:10]
        bodies[i][7:10] = np.zeros(3)


@njit
def find_sound_speed(bodies,comps):
    for i in range(gas_N):
        bodies[i][16] = np.sqrt(gamma*bodies[i][12]/bodies[i][11])

    
@njit
def my(body1,body2):
    dot_product = np.sum((body1[17:20] - body2[17:20]) * (body1[1:4] - body2[1:4]))
    if dot_product < 0.:
        h_ij = 0.5 * (body1[15] + body2[15])
        r_ij = np.sqrt(np.sum((body1[1:4] - body2[1:4])**2))
        return dot_product/(h_ij * (r_ij**2/h_ij**2 + eta**2))
    else:
        return 0


@njit
def visc_tensor(body1,body2):
    return (- 0.5 * alpha * my(body1, body2) * (body1[16] + body2[16]) + beta * my(body1, body2)**2)/(0.5 * (body1[11] + body2[11]))


@njit
def find_visc_tensors(bodies,comps):
    all_visc_tensors = np.zeros((gas_N,gas_N))
    for i in range(gas_N):
        for j in comps[i]:
            
            if j >= 0:
                all_visc_tensors[i][j] = visc_tensor(bodies[i],bodies[j])
    return all_visc_tensors

E_min = 100000
for body in all_bodies:
    if body[13] < E_min:
        E_min = body[13]
@njit
def E_int(bodies,dt,comps,Pd_fac,nW_h_i,nW_h_j,visc_tensors):
    for i in range(gas_N):
        for j in comps[i]:
            
            if j >= 0:
                dE_adb = bodies[j][0] * Pd_fac[i][j] * np.sum((bodies[i][4:7] - bodies[j][4:7]) * 1/2 * (nW_h_i[i][j] + nW_h_j[i][j]))
                bodies[i][13] = bodies[i][13] + dE_adb * dt
                dE_visc = bodies[j][0]/2. * visc_tensors[i][j] * np.sum(bodies[i][4:7] - bodies[j][4:7] * 1/2 * (nW_h_i[i][j] + nW_h_j[i][j]))
                bodies[i][13] = bodies[i][13] + dE_visc * dt
        if bodies[i][13] < E_min:
            bodies[i][13] = E_min


@njit
def a_visc(bodies,comps,visc_tensors,nW_h_i,nW_h_j):
    for i in range(gas_N):
        acc = bodies[i][7:10]
        for j in comps[i]:
            
            if j >= 0:
                acc += - 1/2 * bodies[j][0] * visc_tensors[i][j] * (nW_h_i[i][j] + nW_h_j[i][j])
        bodies[i][7:10] = acc


@njit
def update_h(bodies,comps):
    for i in range(gas_N):
        length = 0
        for comp in comps[i]:
            if comp >= 0:
                length += 1
        factor =  (1./2.) * (1. + (N_target/(length+1))**(1/3))
        if factor < 0.5:
            factor = 0.5
        if factor > 2:
            factor = 2
        bodies[i][15] = bodies[i][15]*factor


def plotdens(bodies,nr,t_passed):
    ds = []
    rhos = []
    for i in range(gas_N):
        body = bodies[i]
        d = np.sqrt(np.sum(body[1:4]**2))
        rho = body[11]/(10**(10))
        ds.append(d)
        rhos.append(rho)
    plt.figure()
    plt.plot(ds,rhos,'.')
    plt.xlabel(r'R [kpc]')
    plt.ylabel(r'Densitet $[M_{sol}/kpc^3]$')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'dens.png'
    plt.savefig(name,bbox_inches='tight')


def plotnumdens(bodies,nr,t_passed):
    opl = 15
    ds = []
    antal = []
    centers = []
    for i in range(gas_N):
        body = bodies[i]
        d = np.sqrt(np.sum(body[1:4]**2))
        ds.append(d)
    sorted_gas = sorted(ds)
    x = np.linspace(0,sorted_gas[-1],opl)
    dx = x[1]-x[0]
    for pos in x:
        antal.append(len(np.where(np.logical_and(sorted_gas > pos - dx/2, sorted_gas < pos +dx/2))[0]))
    plt.figure()
    plt.plot(x,antal)
    plt.xlabel('R [kpc]')
    plt.ylabel('Antals densitet')
    tstr = str(round(t_passed,5))
    nrstr = str(nr)
    name = nrstr + 'i' + tstr + 'antaldens.png'
    plt.savefig(name,bbox_inches='tight')


kodetilkms = 0.9777922275
def plotvel_dm(bodies,nr,t_passed):
    ds = []
    tanvel = []
    perpvel = []
    for i in range(dm_N):
        body = bodies[i+gas_N]
        d = np.sqrt(np.sum(body[1:4]**2))
        normal = body[1:4]/d 
        vperp = np.sum(body[4:7]*normal)*normal
        vtan = body[4:7]-vperp
        ds.append(d)
        tanvel.append(np.sqrt(np.sum(vtan**2))*kodetilkms)
        perpvel.append(np.sqrt(np.sum(vperp**2))*kodetilkms)
    plt.figure()
    plt.plot(ds,perpvel,'.')
    plt.xlabel('R [kpc]')
    plt.ylabel('Perpendikulær hastighed [km/s]')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'pdm.png'
    plt.savefig(name,bbox_inches='tight')
    plt.figure()
    plt.plot(ds,tanvel,'.')
    plt.xlabel('R [kpc]')
    plt.ylabel('Tangentiel hastighed [km/s]')
    name = nrstr + 'i' + tstr + 'tdm.png'
    plt.savefig(name,bbox_inches='tight')



def plotvel_gas(bodies,nr,t_passed):
    ds = []
    tanvel = []
    perpvel = []
    for i in range(gas_N):
        body = bodies[i]
        d = np.sqrt(np.sum(body[1:4]**2))
        normal = body[1:4]/d 
        vperp = np.sum(body[4:7]*normal)*normal
        vtan = body[4:7]-vperp
        ds.append(d)
        tanvel.append(np.sqrt(np.sum(vtan**2))*kodetilkms)
        perpvel.append(np.sqrt(np.sum(vperp**2))*kodetilkms)
    plt.figure()
    plt.plot(ds,perpvel,'.')
    plt.xlabel('R [kpc]')
    plt.ylabel('Perpendikulær hastighed [km/s]')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'pgas.png'
    plt.savefig(name,bbox_inches='tight')
    plt.figure()
    plt.plot(ds,tanvel,'.')
    plt.xlabel('R [kpc]')
    plt.ylabel('Tangentiel hastighed [km/s]')
    name = nrstr + 'i' + 'tgas.png'
    plt.savefig(name,bbox_inches='tight')


def plot_hists(bodies,nr,t_passed):
    gas_bodies = bodies[0:gas_N]
    dm_bodies = bodies[gas_N:gas_N+dm_n]
    star_bodies = bodies[gas_N+dm_n:]
    xs_gas = []
    ys_gas = []
    for body in gas_bodies:
        xs_gas.append(body[1])
        ys_gas.append(body[2])
    xs_dm = []
    ys_dm = []
    for body in dm_bodies:
        xs_dm.append(body[1])
        ys_dm.append(body[2])
    xs_star = []
    ys_star = []
    for body in star_bodies:
        xs_star.append(body[1])
        ys_star.append(body[2])

    xs_all = xs_gas + xs_dm + xs_star 
    ys_all = ys_gas + ys_dm + ys_star

    xs_gasstar = xs_gas + xs_star 
    ys_gasstar = ys_gas + ys_star

    rang = 40
    #gas_hist = np.histogram2d(xs_gas,ys_gas,bins=100,range=[[-rang,rang],[-rang,rang]])[0]
    gas_hist = np.histogram2d(xs_gas,ys_gas,bins=100,range=[[-rang,rang],[-rang,rang]])[0]+1
    star_hist = np.histogram2d(xs_star,ys_star,bins=100,range=[[-rang,rang],[-rang,rang]])[0]+1
    gasstar_hist = np.histogram2d(xs_gasstar,ys_gasstar,bins=100,range=[[-rang,rang],[-rang,rang]])[0]+1
    dm_hist = np.histogram2d(xs_dm,ys_dm,bins=100,range=[[-rang,rang],[-rang,rang]])[0]+1
    all_hist = np.histogram2d(xs_all,ys_all,bins=100,range=[[-rang,rang],[-rang,rang]])[0]+1
    gas_hist = np.log(gas_hist)
    star_hist = np.log(star_hist)
    gasstar_hist = np.log(gasstar_hist)
    dm_hist = np.log(dm_hist)
    all_hist = np.log(all_hist)

    #rang_surf = 5
    #gas_hist_lim = np.histogram2d(xs_gas,ys_gas,bins=100,range=[[-rang_surf,rang_surf],[-rang_surf,rang_surf]])[0]
    #gas_mass = gas_hist_lim*gas_m
    #avg_surf_dens = np.mean(gas_mass)
    cs = []
    rhos = []
    kappas = []
    for i in range(gas_N):
        cs.append(bodies[i][16])
        rhos.append(bodies[i][11])
        kappas.append(2*np.pi*np.sqrt(np.sum(bodies[i][1:4]**2))/(np.sqrt(np.sum(bodies[i][4:7]**2))))
    cs = np.array(cs)
    rhos = np.array(rhos)
    kappas = np.array(kappas)
    #avg_cs = np.mean(cs)
    #kappa = 0.5
    print('Q')
    #print(avg_cs*kappa/(np.pi*G*avg_surf_dens))
    print(cs*kappas/(np.pi*G*rhos))


    extent = -rang,rang,-rang,rang
    plt.figure()
    plt.imshow(gas_hist,origin='lower',extent=extent)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'gashist.png'
    plt.colorbar()
    plt.savefig(name,bbox_inches='tight')

    plt.figure()
    plt.imshow(star_hist,origin='lower',extent=extent)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'starhist.png'
    plt.colorbar()
    plt.savefig(name,bbox_inches='tight')

    plt.figure()
    plt.imshow(gasstar_hist,origin='lower',extent=extent)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    name = nrstr + 'i' + tstr + 'gasstarhist.png'
    plt.colorbar()
    plt.savefig(name,bbox_inches='tight')

    plt.figure()
    plt.imshow(dm_hist,origin='lower',extent=extent)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    name = nrstr + 'i' + tstr + 'dmhist.png'
    plt.colorbar()
    plt.savefig(name,bbox_inches='tight')

    plt.figure()
    plt.imshow(all_hist,origin='lower',extent=extent)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    name = nrstr + 'i' + tstr + 'allhist.png'
    plt.colorbar()
    plt.savefig(name,bbox_inches='tight')


kodetila = 3.098436597*10**(-17)
def plot_accelerations(bodies,nr,t_passed,dt):
    ds_gas = []
    accs_gas = []
    for i in range(gas_N):
        ds_gas.append(np.sqrt(np.sum(bodies[i][1:4]**2))*kodetila)
        accs_gas.append(np.sqrt(np.sum(bodies[i][7:10]**2))*kodetila)
    ds_dm = []
    accs_dm = []
    for i in range(dm_N):
        body = bodies[i+gas_N]
        ds_dm.append(np.sqrt(np.sum(body[1:4]**2))*kodetila)
        accs_dm.append(np.sqrt(np.sum(body[7:10]**2))*kodetila)

    plt.figure()
    plt.plot(ds_gas,accs_gas,'.')
    plt.xlabel('kpc')
    plt.ylabel(r'$a$ $[km/s^2]$')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    dtstr = str(round(dt,5))
    name = nrstr + 'i' + tstr + 'i' + dtstr + 'gasacc.png'
    plt.savefig(name,bbox_inches='tight')

    plt.figure()
    plt.plot(ds_dm,accs_dm,'.')
    plt.xlabel('kpc')
    plt.ylabel(r'$a$ $[km/s^2]$')
    nrstr = str(nr)
    tstr = str(round(t_passed,5))
    dtstr = str(round(dt,5))
    name = nrstr + 'i' + tstr + 'i' + dtstr + 'dmacc.png'
    plt.savefig(name,bbox_inches='tight')


def run():
    bodies = all_bodies
    find_initial_h(bodies)

    its = 4
    for i in range(its):
        comps = MakeTree(bodies)
        update_h(bodies,comps)

    for i in range(len(bodies)):
        bodies[i][7:10] = np.zeros(3)

    reset_acceleration(bodies)
    update_h(bodies,comps)
    comps = MakeTree(bodies)
    update_density(bodies,comps)
    find_pressure(bodies)
    find_sound_speed(bodies,comps)
    nW_h_i, nW_h_j = find_nWs(bodies,comps)
    visc_tensors = find_visc_tensors(bodies,comps)
    Pd_facs = find_Pd_fac(bodies,comps)
    pressure_gradient(bodies,nW_h_i,nW_h_j,Pd_facs,comps)
    dt = find_dt(bodies)
    find_prediction_velocity(bodies,dt)
    a_visc(bodies,comps,visc_tensors,nW_h_i,nW_h_j)
    dt = find_dt(bodies)
    


    xs0 = []
    ys0 = []

    hists = []
    t1 = time.time()
    u = 10
    #for i in range(500):
    dt = 0
    nr = 0
    t_passed = 0
    t_passed_plot = 0
    while t_passed < t and nr < 1001:
        print(nr)
        if nr%10 == 0:
            #plotdens(bodies,nr,t_passed)
            #plotvel_dm(bodies,nr,t_passed)
            #plotvel_gas(bodies,nr,t_passed)
            #plotnumdens(bodies,nr,t_passed)
            plot_hists(bodies,nr,t_passed)
            #plot_accelerations(bodies,nr,t_passed,dt)




        xs = []
        ys = []
        '''snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)'''
        dt = find_dt(bodies)
        print(dt)
        print(t_passed)
        t_passed += dt
        t_passed_plot += dt
        Kick(bodies,dt)
        Drift(bodies,dt)
        reset_acceleration(bodies)
        update_h(bodies,comps)
        comps = MakeTree(bodies)
        update_density(bodies,comps)
        find_pressure(bodies)
        find_sound_speed(bodies,comps)
        nW_h_i, nW_h_j = find_nWs(bodies,comps)
        visc_tensors = find_visc_tensors(bodies,comps)
        Pd_facs = find_Pd_fac(bodies,comps)
        pressure_gradient(bodies,nW_h_i,nW_h_j,Pd_facs,comps)
        find_prediction_velocity(bodies,dt)
        a_visc(bodies,comps,visc_tensors,nW_h_i,nW_h_j)
        #E_int(bodies,dt,comps,Pd_facs,nW_h_i,nW_h_j,visc_tensors)
        Kick(bodies,dt)
        #if t_passed_plot > t/100:
        if nr%10 == 0:
            t_passed_plot = 0
            for k in range(len(bodies)):
                xs.append(bodies[k][1])
                ys.append(bodies[k][2])
            hist = np.histogram2d(xs,ys,bins=70,range=[[-40,40],[-40,40]])[0]
            hist = np.log(hist, out=np.zeros_like(hist), where =(hist!=0))
            hists.append(hist)
            xs0.append(bodies[0][1])
            ys0.append(bodies[0][2])
        nr += 1
    t2 = time.time()
    print(t2-t1)

    plt.figure()
    plt.plot(xs0,ys0)
    #plt.savefig('one.png')

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)



    def ani_hist(x):
        fig.clear()
        h = hists[x]
        plt.imshow(h,origin='lower')
        plt.colorbar()
        


    anim = ani.FuncAnimation(fig,ani_hist,frames=int(len(hists)),interval=0.0001)

    anim.save('C2.gif',writer='Pillow')
    #plt.show()


run()