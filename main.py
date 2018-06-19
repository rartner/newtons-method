u"""Método de Newton para sistemas não lineares."""
import argparse
import math
import numpy as np


# Funções do sistema dado
def fst_function(p, x1, x2):
    return np.exp(-x1) + np.exp(x2) - (x1 ** 2) - (2 * (x2 ** 3))


def snd_function(p, x1, x2):
    return np.cos(x1 + p * x2) + (p * x1) - x2 - 1


def get_Fx(p, x1, x2):
    """Gera a matrix F(x1, x2)."""
    return np.array([fst_function(p, x1, x2), snd_function(p, x1, x2)])


# Funções da matriz J
def delf1_x1(p, x1, x2):
    return -np.exp(-x1) - (2 * x1)


def delf1_x2(p, x1, x2):
    return np.exp(x2) - (6 * (x2 ** 2))


def delf2_x1(p, x1, x2):
    return p - np.sin(x1 + (p * x2))


def delf2_x2(p, x1, x2):
    return -p * np.sin(x1 + (p * x2)) - 1


def get_Jx(p, x1, x2):
    u"""Gera a matriz de derivadas parciais para as funções."""
    return np.array([
                [delf1_x1(p, x1, x2), delf1_x2(p, x1, x2)],
                [delf2_x1(p, x1, x2), delf2_x2(p, x1, x2)]
            ])


# Método de Newton
def get_error(old_X, new_X):
    """Calcula o erro absoluto."""
    return max([abs(new_X[0] - old_X[0]), abs(new_X[1] - old_X[1])])


def solve_system(p, X):
    u"""Função que resolve o sistema linear com a matrix Jacobiana."""
    Jx = get_Jx(p, X[0], X[1])
    Fx = get_Fx(p, X[0], X[1])
    w = np.linalg.solve(Jx, -Fx)
    return np.add(X, w)


def get_result(p, X, verbose_mode):
    u"""Calcula os valores de X para que o sistema seja igual a 0."""
    error = 1
    iteration = 0
    print ('X0: ({x}, {y})'.format(x=X[0], y=X[1]))
    print ('Primeiras 5 iterações:')
    print ('k: \tx1: \t\tx2: \t\tEabs')
    while (error > 0.00001 and iteration <= 1000):
        iteration += 1
        oldX = X
        X = solve_system(p, X)
        if (math.isnan(X[0]) or math.isnan(X[1])):
            print ('\nFalha na iteração:', iteration)
            print ('x1: %.6g \tx2: %.6g \tEabs: %.6g' % (X[0], X[1], error))
            print ('Execute com a flag -v para acompanhar o processo.')
            return
        error = get_error(oldX, X)
        if (iteration <= 5 or verbose_mode):
            print ('%g \t%.6g \t%.6g \t%.6g' % (iteration, X[0], X[1], error))
    str_result('Resultado após {} iterações:'.format(iteration), X, error, p)


def str_result(string, result, error, p):
    u"""Mostra o resultado final da execução."""
    print ('\n{}'.format(string))
    print ('- x1:  ', result[0])
    print ('- x2:  ', result[1])
    print ('- Erro:', error)
    print ('- f1:  ', fst_function(p, result[0], result[1]))
    print ('- f2:  ', snd_function(p, result[0], result[1]))


def main():
    u"""Método de Newton para sistemas não lineares."""
    parser = argparse.ArgumentParser(
                description='Metodo de Newton para sistemas nao lineares')
    parser.add_argument(
        '-p', choices=[22, 29], help='valor de p para o sistema', type=int)
    parser.add_argument(
        '-v', help='use verbose mode', action='store_true')

    args = parser.parse_args()
    if (args.p == 22):
        print ('=============================> primeiro caso:')
        get_result(args.p, [0.5, 1.5], args.v)
        print ('=============================> segundo caso:')
        get_result(args.p, [4, 6], args.v)
        print ('=============================> terceiro caso:')
        # TODO: outro teste => get_result(args.p, [19, 1], args.v)
        get_result(args.p, [4.559999999999947, 4.559999999999947], args.v)
    else:
        print ('=============================> primeiro caso:')
        get_result(args.p, [0.5, 1.7], args.v)
        print ('=============================> segundo caso:')
        get_result(args.p, [4, 6], args.v)
        print ('=============================> terceiro caso:')
        get_result(args.p, [5.09, 5.09], args.v)


if __name__ == '__main__':
    main()
