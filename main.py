u"""Método de Newton para sistemas não lineares."""
import argparse
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


def fst_test(p, X):
    u"""
    Uma sequência cujo termo X(5) aproxima a outra solução com erro < 0.00001
    em exatamente 5 iterações.
    """
    print ('==> primeira situação:')
    for i in range(5):
        oldX = X
        X = solve_system(p, X)
        error = get_error(oldX, X)
        print ('x1: %.6g \tx2: %.6g \tEabs: %.6g' % (X[0], X[1], error))
    str_result('Resultado após 5 iterações:', X, get_error(oldX, X), p)


def snd_test(p, X):
    u"""
    Uma sequência cujo termo X(5) aproxima a outra solução com erro < 0.00001.
    """
    print ('==========================================\n==> segunda situação:')
    for i in range(5):
        oldX = X
        X = solve_system(p, X)
        error = get_error(oldX, X)
        print ('x1: %.6g \tx2: %.6g \tEabs: %.6g' % (X[0], X[1], error))
        if (error < 0.00001):
            str_result('Resultado após {} iterações:'.format(i+1), X, error, p)
            return
    str_result('Resultado após 5 iterações:', X, get_error(oldX, X), p)


def trd_test(p, X):
    u"""
    Uma sequência que (por algum motivo) não resulte em uma aproximação de
    nenhuma das soluções com erro absoluto menor do que 0.00001, mesmo se
    tentar executar 1000 iterações.
    """
    print ('=========================================\n==> terceira situação:')

    for i in range(1000):
        oldX = X
        X = solve_system(p, X)
        if (i < 5):
            error = get_error(oldX, X)
            print ('x1: %.6g \tx2: %.6g \tEabs: %.6g' % (X[0], X[1], error))
    str_result('Resultado após 1000 iterações:', X, get_error(oldX, X), p)


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

    args = parser.parse_args()
    if (args.p == 22):
        fst_test(args.p, [0.5, 1.5])
        snd_test(args.p, [0.3, 1.3])
        trd_test(args.p, [123, 100])
    else:
        fst_test(args.p, [0.5, 1.7])
        snd_test(args.p, [0.3, 1.3])
        trd_test(args.p, [100, 100])


if __name__ == '__main__':
    main()
