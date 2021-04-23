#!/usr/bin/env python
# coding: utf-8


#import import_ipynb
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
from functools import reduce
import matplotlib.pyplot as plt
import math, random


def sum_of_squares(v):
    '''Computa a soma dos elementos ao quadrado em v'''
    
    return sum(v_i ** 2 for v_i in v)



def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h



def plot_estimated_derivative():
    
    def square(x):
        return x * x
    
    def derivative(x):
        return 2 * x
    
    derivative_estimate = lambda x: difference_quotient(square, x, h=0.00001)
    
    # planeja mostrar que são basicamente o mesmo
    x = range(-10,10)
    plt.plot(x, list(map(derivative, x)), 'rx', label='actual')            # red x
    plt.plot(x, list(map(derivative_estimate, x)), 'b+', label='estimate') # blue +
    plt.title('Actual Derivatives vs. Estimates')
    plt.legend(loc=9)
    plt.show()
    
plot_estimated_derivative()


def partial_difference_quotient(f, v, i, h):
    
    '''computa o i-ésimo quociente diferencial parcial de f em v'''
    w = [v_j + (h if j == i else 0) # adiciona h ao elemento i-ésimo de v
         for j, v_j in enumerate(v)]
    
    return (f(w) - f(v)) / h


def estimate_gradiente(f,v,h=0.00001):
    
    return [partial_difference_quotient(f,v,i,h)
            for i, _ in enumerate(v)]


def step(v, direction, step_size):
    
    '''move step_size na direção a partir de v'''
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]



def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]



def safe(f):
    '''retorna uma nova função que é igual a f, exceto que ele exibe
    infinito como saída toda vez que f produz um erro'''
    
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf') # isso significa 'infinito' em Python
    return safe_f



#
#
# minimize / maximize batch
#
#


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    '''usa o gradiente descendente para encontrar theta que minimize a função alvo'''
    
    step_sizes = [100,10,1,0.1,0.01,0.001,0.0001,0.00001]
    
    theta = theta_0             # ajusta theta para o valor inicial
    target_fn = safe(target_fn) # versão segura de target_fn
    value =  target_fn(theta)   # valor que estamos minimizando
    
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]
        
        # escolhe aquele que minimiza a função de erro
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        
        # para se estivermos 'convergindo'
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


def negate(f):
    '''retorna uma função que, para qualquer entrada, x retorna -f(x)'''
    
    return lambda *args, **kwargs: -f(*args, **kwargs)



def negate_all(f):
    '''O mesmo quando f retorna uma lista de numeros'''
    
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


#
# minimize / maximize stochastic
#


def in_random_order(data):
    '''gerador retorna os elementos do dado em ordem aleatória'''
    
    indexes = [i for i, _ in enumerate(data)] # cria uma lista de indices
    random.shuffle(indexes)                   # os embaralha
    for i in indexes:                         # retorna os dados naquela ordem
        yield data[i]


def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    
    data = list(zip(x,y))
    theta = theta_0                           # palpite inicial
    alpha = alpha_0                           # tamanho do passo inicial
    min_theta, min_value = None, float('inf') # o minimo até agora
    iterations_with_no_improvement = 0
    
    # Se formos até 100 iterações sem melhorias, paramos
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        
        if value < min_value:
            
            # Se achou um novo minimo, lembre-se
            # e volte para o tamanho do passo original
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            
            # Do contrario, não estamos melhorando, portanto tente
            # diminuir o tamanho do passo
            iterations_with_no_improvement += 1
            alpha *= 0.9
            
        # E ande um passo gradiente para todos os pontos de dados
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x,y,theta_0, alpha_0)


if __name__ == '__main__':
    
    print('using the gradient')
    
    v = [random.randint(-10,10) for i in range(3)]
    
    tolerance = 0.0000001
    
    while True:
        gradient = sum_of_squares_gradient(v) # computa o gradiente em v
        next_v = step(v, gradient, -0.01)     # pega um passo gradiente negativo
        if distance(next_v, v) < tolerance:   # para se estivermos convergindo
            break
        v = next_v                            # continua se não estivermos
        
    print(f'Minimum v: {v}')
    print(f'Minimum value: {sum_of_squares(v)}')
    print('------------------------------')
    
    print('Using minimize_batch')
    
    v = [random.randint(-10,10) for i in range(3)]
    
    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)
    
    print(f'minimum v: {v}')
    print(f'minimum value: {sum_of_squares(v)}')



