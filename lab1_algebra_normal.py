import random
from itertools import product
from typing import Dict, List, Tuple, Union
from sympy import gcd
from galois import GF, Poly
from sympy import factorial
from sympy.combinatorics import SymmetricGroup, Permutation
from sage.all import *
import random

def subgroups_of_Sm(N):
    """
        Возвращает словарь вида:
        {'Кол_во подгрупп': int,
        'Случайная подгруппа': [Permutation, Permutation, ...],
        'Индекс [Permutation, Permutation, ...]: int,
        'Нормальная': bool}
        (Так как мой компьютер не позволяет найти все подгруппы Sm,
        значение количества подгрупп подчситано на сайте CoCalc(файл в репозитории count_subgrups.ipynb),
        а случайная подгруппа и подгруппа для вычисления индекса и ее нормальности взяты из циклических)
        """
    m = 4 + (N % 5)
    S = SymmetricGroup(m)
    subs = S.subgroups()

    H_rand = random.choice(subs)
    H = subs[N % len(subs)]

    info = {
        "m": m,
        "order_S": S.order(),
        "subgroups_count": len(subs),
        "random_H_order": H_rand.order(),
        "random_H_gens": H_rand.gens(),
        "target_index": N % len(subs),
        "H_order": H.order(),
        "H_index": S.order() // H.order(),
        "H_gens": H.gens(),
        "is_normal": H.is_normal(S),
        "num_left_cosets": len(S.cosets(H, side='left')),
        "num_right_cosets": len(S.cosets(H, side='right')),
    }
    return info


ISU = 502727
N = ISU % 20
res = subgroups_of_Sm(N)

print(f"S_{res['m']} (порядок {res['order_S']})")
print(f"Всего подгрупп: {res['subgroups_count']}")
print(f"Случайная подгруппа: порядок {res['random_H_order']}, генераторы {res['random_H_gens']}")
print(f"Выбранная подгруппа №{res['target_index']}: порядок {res['H_order']}, индекс {res['H_index']}")
print(f"Нормальна: {res['is_normal']}")
print(f"Левых классов: {res['num_left_cosets']}, правых: {res['num_right_cosets']}")

def element_powers_in_Sm(N: int) -> Dict:
    """
    Анализ степеней элементов и порожденных ими подгрупп в симметрической группе.

    Для заданного N вычисляет m = 4 + N % 5 и выбирает элемент из S_m.
    Исследует степени этого элемента и циклические подгруппы, порожденные этими степенями.

    Parameters:
    N (int): Входной параметр для вариативности вычислений

    Returns:
    Dict: Словарь с информацией о степенях элемента и порожденных подгруппах
    """
    m = 4 + N % 5
    n1 = N % 6
    n2 = (N + 1) % 6
    n3 = (N + 2) % 6

    group = SymmetricGroup(m)
    group_elements = list(group.generate())

    element = group_elements[N % factorial(m)]
    element_n1 = element ** n1
    element_n2 = element ** n2
    element_n3 = element ** n3

    group_n1 = group.subgroup([element_n1])
    group_n2 = group.subgroup([element_n2])
    group_n3 = group.subgroup([element_n3])

    orders = dict()
    orders['g'] = element
    orders['o(g_n1)'] = [element_n1, element_n1.order()]
    orders['o(g_n2)'] = [element_n2, element_n2.order()]
    orders['o(g_n3)'] = [element_n3, element_n3.order()]
    orders['|<g_n1>|'] = [str(group_n1), group_n1.order()]
    orders['|<g_n2>|'] = [str(group_n2), group_n2.order()]
    orders['|<g_n3>|'] = [str(group_n3), group_n3.order()]
    return orders


def solve_sigma_power_eq(N: int) -> Dict:
    """
    Решение уравнения вида σ^n = τ в симметрической группе.

    Находит все перестановки σ в S_m такие, что некоторая степень σ равна
    фиксированной перестановке τ. Возвращает количество решений и примеры.

    Parameters:
    N (int): Входной параметр для вариативности вычислений

    Returns:
    Dict: Словарь с количеством решений и примерами
    """
    m = 4 + N % 5

    group = SymmetricGroup(m)
    group_elements = list(group.generate())

    solutions = set()
    answer = list(range(1, m))
    answer.append(0)
    for element in group_elements:
        for n in range(element.order()):
            if element ** n == Permutation(answer):
                solutions.add(element)

    result = dict()
    result['Кол-во решений'] = len(solutions)
    result['3 случайных решения'] = [random.choice(list(solutions)) for _ in range(3)]
    return result


def elements_of_order_k_in_cyclic_group(N: int) -> Dict:
    """
    Поиск элементов заданного порядка в симметрической группе.

    Для заданного N вычисляет m = 4 + N % 5 и k = 1 + N % 7.
    Находит все элементы, удовлетворяющие условию g^k = e, и все элементы порядка k.

    Parameters:
    N (int): Входной параметр для вариативности вычислений

    Returns:
    Dict: Словарь с двумя списками элементов
    """
    m = 4 + N % 5
    k = 1 + N % 7

    group = SymmetricGroup(m)
    group_elements = list(group.generate())

    list_power = list()
    list_orders = list()
    for element in group_elements:
        if element ** k == Permutation(range(m)):
            list_power.append(element)
        if element.order() == k:
            list_orders.append(element)

    result = dict()
    result['Список g ** k = e'] = list_power
    result['Список o(g) = k'] = list_orders
    return result


def subgroups_of_Zm_star(N: int) -> list:
    """Находит все подгруппы мультипликативной группы Zₘ*
    Входной параметр для вычисления m = 4 + (N % 5)

    Returns:
    Список подгрупп, отсортированных по размеру и элементам.
        Каждая подгруппа - список своих элементов."""
    m = 4 + N % 5
    units = [a for a in range(1, m) if gcd(a, m) == 1]
    subgroups_set = set()
    subgroups_set.add(tuple([1]))
    for g in units:
        subgroup = []
        x = g
        while x not in subgroup:
            subgroup.append(x)
            x = (x * g) % m
        subgroup_tuple = tuple(sorted(subgroup))
        subgroups_set.add(subgroup_tuple)
    result = [list(t) for t in subgroups_set]
    result.sort(key=lambda x: (len(x), x))
    return result


ISU = 502727
subgroups = subgroups_of_Zm_star(ISU % 20)
for sg in subgroups:
    print(sg)




def order_of_sr(N: int) -> int:
    """
    Вычисляет порядок элемента s^r в мультипликативной группе F_p* для p = 31.

    Args:
        N: Входной параметр для определения конкретного элемента

    Returns:
        Порядок элемента s^r в группе F_31*
    """
    p = 0
    if N == 7:
        p = 37
    s = 3  # тк N = 1
    r = 38  # N = 1
    temp = s % p
    m = 1
    while temp != 1:
        temp = (temp * s) % p
        m += 1

    gcd_ord = gcd(m, r)
    return m // gcd_ord


ISU = 502727
order = order_of_sr(ISU % 20)
print(order)




def order_and_primitivity_of_t(N: int) -> dict:
    """
    Проверяет, является ли элемент примитивным в мультипликативной группе F_31*.

    Args:
        N: Входной параметр для определения элемента

    Returns:
        Словарь с порядком элемента и результатом проверки (YES/NO)
    """
    p = 0
    if N == 7:
        p = 37
    t = 7
    m = 1
    temp = t % p
    while temp != 1:
        temp = (temp * t) % p
        m += 1
    if m == p - 1:
        return {m: 'YES'}
    else:
        return {m: 'NO'}


ISU = 502727
print(order_and_primitivity_of_t(ISU % 20))


def generators_of_Zm_star(N: int) -> list:
    """
    Находит все примитивные элементы мультипликативной группы Z_m*.

    Args:
        N: Входной параметр для вычисления m = 4 + N % 5

    Returns:
        Список примитивных элементов группы Z_m*
    """
    m = 4 + N % 5
    group_order = sum(1 for i in range(1, m) if gcd(i, m) == 1)

    generators = []
    for elem in range(1, m):
        if gcd(elem, m) != 1:
            continue
        order = 1
        temp = elem % m
        while temp != 1:
            temp = (temp * elem) % m
            order += 1
        if order == group_order:
            generators.append(elem)
    return generators


ISU = 502727
print(generators_of_Zm_star(ISU % 20))


def cyclic_subgroup_in_Zm_additive(N: int) -> dict:
    """
    Анализирует циклическую подгруппу в аддитивной группе Z_m.

    Args:
        N: Входной параметр для вычисления m = 4 + N % 5

    Returns:
        Словарь с порядком подгруппы и списком ее примитивных элементов
    """
    m = 4 + N % 5
    t_base = 7
    t = t_base % m
    subgroup = set()
    for i in range(m):
        subgroup.add((t * i) % m)
    subgroup_order = len(subgroup)
    primitive_elements = []
    for elem in subgroup:
        if elem == 0:
            continue
        if m // gcd(m, elem) == subgroup_order:
            primitive_elements.append(elem)
    return {subgroup_order: primitive_elements}
    
    
ISU = 502727
print(cyclic_subgroup_in_Zm_additive(ISU % 20))
    
    
    
    
def isomorphism_of_cyclic_subgroup_Zm_star(N: int):
    """
    Устанавливает изоморфизм между циклической подгруппой и подгруппой симметрической группы.

    Args:
        N: Входной параметр для вычисления m = 4 + N % 5

    Returns:
        Словарь с порядком подгруппы и информацией об изоморфизме
    """
    m = 4 + N % 5
    t_old = 7
    t = t_old % m
    subgroup = set()
    for i in range(1, m):
        subgroup.add((i * t) % m)
    d = len(subgroup)
    cyclic_group = f"<(1 2 ... {d})> при S{d}"
    return {d: (subgroup, cyclic_group)}
    
    
ISU = 502727
print(isomorphism_of_cyclic_subgroup_Zm_star(ISU % 20))

def ai(N: int, i: int) -> int:
    """Коэффициент a_i для построения полиномов над F4"""
    return (i + N) % 4


def bj(N: int, j: int) -> int:
    """Коэффициент b_j для построения полиномов над F7"""
    return (j + N) % 7


def ck(N: int, k: int) -> int:
    """Коэффициент c_k для построения полиномов над F5"""
    return (k + N) % 5


def dl(N: int, l: int) -> int:
    """Коэффициент d_l для построения полиномов над F9"""
    return (l + N) % 9


def rm(N: int, m: int) -> int:
    """Коэффициент r_m для построения полиномов над F11"""
    return (m + N) % 11


def st(N: int, t: int) -> int:
    """Коэффициент s_t для построения полиномов над F11 и F13"""
    return (t + N) % 11


def roots_F4(N: int) -> List[str]:
    """
    Находит корни полинома степени 8 в поле F4.

    Args:
        N: Входной параметр для определения коэффициентов

    Returns:
        Список корней в строковом представлении
    """
    field = GF(4)
    coefficients = [field(1)] + [field(ai(N, i)) for i in range(8, -1, -1)]
    func = Poly(coefficients, field=field)
    roots = [str(root) for root in func.roots()]
    return roots


def roots_F7(N: int) -> List[str]:
    """
    Находит корни полинома степени 6 в поле F7.

    Args:
        N: Входной параметр для определения коэффициентов

    Returns:
        Список корней в строковом представлении
    """
    field = GF(7)
    coefficients = [field(bj(N, i)) for i in range(6, -1, -1)]
    func = Poly(coefficients, field=field)
    roots = [str(root) for root in func.roots()]
    return roots


def reducibility_F5(N: int) -> str:
    """
    Разлагает полином степени 5 на неприводимые множители в F5.

    Args:
        N: Входной параметр для определения коэффициентов

    Returns:
        Строка с разложением на множители
    """
    field = GF(5)
    coefficients = [field(1)] + [field((ck(N, i))) for i in range(4, -1, -1)]
    func = Poly(coefficients, field=field)
    factors = func.factors()

    result = 'f(x) ='
    for factor, power in zip(factors[0], factors[1]):
        result += f' ({factor})^{power} *'
    return result[:-2]

def reducibility_F9(N: int) -> str:
    """
    Разлагает полином степени 4 на неприводимые множители в F9.

    Args:
        N: Входной параметр для определения коэффициентов

    Returns:
        Строка с разложением на множители
    """
    field = GF(9)
    coefficients = [field(1)] + [field((dl(N, i))) for i in range(3, -1, -1)]
    func = Poly(coefficients, field=field)
    factors = func.factors()

    result = 'f(x) ='
    for factor, power in zip(factors[0], factors[1]):
        result += f' ({factor})^{power} +'
    return result[:-2]


def gcd_F11(N: int) -> str:
    """
    Находит НОД двух полиномов и его линейное представление в F11.

    Args:
        N: Входной параметр для определения полиномов

    Returns:
        Строка с линейным представлением НОД
    """
    field = GF(11)
    coefficients_f = [field((rm(N, i))) for i in range(7, -1, -1)]
    coefficients_g = [field((st(N, i))) for i in range(3, -1, -1)]
    func_f = Poly(coefficients_f, field=field)
    func_g = Poly(coefficients_g, field=field)

    table1 = [func_f, field(1), field(0)]
    table2 = [func_g, field(0), field(1)]
    while table1[0] != field(0) and table2[0] != field(0):
        if table1[0].degree > table2[0].degree:
            coef_to_multiply = [1] + [0 for _ in range(table1[0].degree - table2[0].degree)]
            table1[0] -= table2[0] * Poly(coef_to_multiply, field=field)
            table1[1] -= table2[1] * Poly(coef_to_multiply, field=field)
            table1[2] -= table2[2] * Poly(coef_to_multiply, field=field)
        elif table1[0].degree < table2[0].degree:
            coef_to_multiply = [1] + [0 for _ in range(table2[0].degree - table1[0].degree)]
            table2[0] -= table1[0] * Poly(coef_to_multiply, field=field)
            table2[1] -= table1[1] * Poly(coef_to_multiply, field=field)
            table2[2] -= table1[2] * Poly(coef_to_multiply, field=field)
        else:
            if table1[0].coeffs[0] >= table2[0].coeffs[0]:
                table1[0] -= table2[0]
                table1[1] -= table2[1]
                table1[2] -= table2[2]
            else:
                table2[0] -= table1[0]
                table2[1] -= table1[1]
                table2[2] -= table1[2]

    if table1[0] == field(0):
        return f'{table2[0]} = ({func_f}) * ({table2[1]}) + ({func_g}) * ({table2[2]})'
    else:
        return f'{table1[0]} = ({func_f}) * ({table1[1]}) + ({func_g} * {table1[2]})'


def inverse_F13(N: int) -> Union[str, Tuple[str, str]]:
    """
    Находит обратный полином по модулю в F13.

    Args:
        N: Входной параметр для определения полиномов

    Returns:
        Обратный полином или сообщение о необратимости
    """
    field = GF(13)
    coefficients_f = [field((st(N, i))) for i in range(3, -1, -1)]
    coefficients_g = [1, 0, 0, 0, 1, 1, 0, 6, 2]
    func_f = Poly(coefficients_f, field=field)
    func_g = Poly(coefficients_g, field=field)

    table1 = [func_f, field(1), field(0)]
    table2 = [func_g, field(0), field(1)]
    while table1[0] != field(0) and table2[0] != field(0):
        if table1[0].degree > table2[0].degree:
            coef_to_multiply = [1] + [0 for _ in range(table1[0].degree - table2[0].degree)]
            table1[0] -= table2[0] * Poly(coef_to_multiply, field=field)
            table1[1] -= table2[1] * Poly(coef_to_multiply, field=field)
            table1[2] -= table2[2] * Poly(coef_to_multiply, field=field)
        elif table1[0].degree < table2[0].degree:
            coef_to_multiply = [1] + [0 for _ in range(table2[0].degree - table1[0].degree)]
            table2[0] -= table1[0] * Poly(coef_to_multiply, field=field)
            table2[1] -= table1[1] * Poly(coef_to_multiply, field=field)
            table2[2] -= table1[2] * Poly(coef_to_multiply, field=field)
        else:
            if table1[0].coeffs[0] >= table2[0].coeffs[0]:
                table1[0] -= table2[0]
                table1[1] -= table2[1]
                table1[2] -= table2[2]
            else:
                table2[0] -= table1[0]
                table2[1] -= table1[1]
                table2[2] -= table1[2]

    if table1[0] == field(0):
        if table2[0].degree != 0:
            return 'необратим'
        func_h = table2[1] // table2[0]
    else:
        if table1[0].degree != 0:
            return 'необратим'
        func_h = table1[1] // table1[0]

    return f'({func_h}) * ({func_f}) ≡ 1 mod ({func_g})', f'h(x) = {func_h}'


def generate_irreducible_polynomials(q: int, d: int) -> List[str]:
    """
    Генерирует все неприводимые полиномы степени d над полем F_q.

    Args:
        q: Характеристика поля
        d: Степень полиномов

    Returns:
        Список неприводимых полиномов
    """
    field = GF(q)
    irreducibles = []

    for part_coefs in product([field(i) for i in range(q)], repeat=d):
        coefficients = [field(1)] + list(part_coefs)
        func_f = Poly(coefficients, field=field)
        reducible = False
        for deg_g in range(1, d // 2 + 1):
            for coefs_g in product([field(i) for i in range(q)], repeat=deg_g):
                g = Poly([field(1)] + list(coefs_g), field=field)
                if func_f % g == 0:
                    reducible = True
                    break
            if reducible:
                break

        if not reducible:
            irreducibles.append(str(func_f))

    return irreducibles


ISU = 502727
N = ISU % 20

