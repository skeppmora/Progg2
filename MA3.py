#!/usr/bin/env python3
""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from functools import reduce
from numba import njit
# Exc1
def approximate_pi(n):
    Nc = 0      # number of points in circle
    circle_lst = []
    square_lst = []
    
    fig, ax = plt.subplots()
    ax.set_aspect(1)


    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        if (x**2 + y**2) <= 1:
            Nc += 1
            circle_lst.append([x, y])
            # ax.scatter(x, y, s=5, color='r')

        else:
            square_lst.append((x, y))
            # ax.scatter(x, y, s= 5, color='b')

    ax.scatter([item[0] for item in circle_lst], [item[1] for item in circle_lst], s=5, color= 'r')     # plot all points inside circle
    ax.scatter([item[0] for item in square_lst], [item[1] for item in square_lst], s=5, color= 'b')     # plot all points outside of circle
    
    fig.savefig(f'Pi_approximation_for_{n}.png', dpi=300)
    plt.show()

    pi_approx = 4 * Nc / n
    print(f'pi~{pi_approx}')
    return pi_approx

# Exc2, approximation
def sphere_volume(n, d, r= 1):    # n=testing points  d=dimension
    coordinates = [[random.uniform(-r, r) for _ in range(d)] for _ in range(n)]

    def sphere_check(point):
        return reduce(lambda result, x: result + x**2, point, 0) <= r

    in_sphere = [point for point in coordinates if sphere_check(point) is True]
    volume_approx = ((2*r)**d) * (len(in_sphere) / n)

    return volume_approx

#Exc2, real value
def hypersphere_exact(n, d, r=1):
    return (r**d) * ((m.pi)**(d/2) / m.gamma((d/2) + 1))


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int, r:float =1)->float:
    coordinates = [[random.uniform(-r, r) for _ in range(d)] for _ in range(n)]
    
    def sphere_check(point):
        return reduce(lambda result, x: result + x**2, point, 0) <= r
    
    in_sphere = [point for point in coordinates if sphere_check(point) is True]
    volume_approx = ((2*r)**d) * (len(in_sphere) / n)

    #np is the number of processes
    return volume_approx

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_check(args):        
    n, d = args
    sphere_points = 0
    coordinates = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)] # gör om denna till att inte spara alla punkter i yttre loop, istället
    # bättre att skapa en punkt coordinate och kontrollera ifall den uppfyller kravet, isåfall lägga till 1 i sphere_points. Sparar massa minne
    for point in coordinates:
        if sum(map(lambda x: x**2, point)) <= 1:
            sphere_points += 1

    return sphere_points

def sphere_volume_parallel(n, d, np=10):
    base, rest = divmod(n, np)
    splits = [base + (1 if i < rest else 0) for i in range(np)]
    
    with future.ProcessPoolExecutor(max_workers= np) as ex:        
        counts = list(ex.map(sphere_check, [(split, d) for split in splits]))
        
    tot_inside = sum(counts)
    volume_approx = (2**d) * (tot_inside / n)

    return volume_approx
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    print(f'Approximated volume of {d}-dimensional sphere = {sphere_volume(n, d)}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n, d)}")

    n = 100000
    d = 11
    print(f'Approximated volume of {d}-dimensional sphere = {sphere_volume(n, d)}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n, d)}")

    # Exc3
    n = 1000000
    d = 11
    seq_time_lst = []   # for sequential calculation
    for _ in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        seq_time_lst.append(stop-start)
    print(f"Exc3: Sequential times of {d} and {n}: {seq_time_lst}")

    njit_time_lst = []  # for Numba calculation
    for _ in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        njit_time_lst.append(stop-start)
    print(f"Exc3: Numba times of {d} and {n}: {njit_time_lst}")
    Numba-version runs faster, first call is a bit slower, probably because of compiling
    

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n} is {stop-start}")
    
    start = pc() 
    sphere_volume_parallel(n,d, 16)
    stop = pc()
    print(f'Exc4: Parallell time of {d} and {n} is {stop - start}')

    
    

if __name__ == '__main__':
	main()
