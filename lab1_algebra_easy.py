from typing import List, Dict, Tuple
from sympy import isprime, primefactors, gcd, totient
from math import factorial
import time

def is_palindrome(n: int) -> bool:
    """Проверяет, является ли число палиндромом."""
    return str(n) == str(n)[::-1]

def palindromic_squares_and_circular_primes() -> Tuple[List[int], List[int]]:
    """
    Возвращает:
    tuple:
    - список палиндромов a < 100000, где a^2 тоже палиндром
    - список простых p < 1000000, у которых все циклические перестановки цифр просты
    """
    palindromic_squares: List[int] = []
    for a in range(1, 10 ** 5):
        if is_palindrome(a) and is_palindrome(a ** 2):
            palindromic_squares.append(a)


    def get_cyclic_rotations(n: int) -> List[int]:
        """Генерирует все циклические перестановки числа."""
        s = str(n)
        rotations = set()
        for i in range(len(s)):
            rotated_s = s[i:] + s[:i]
            rotations.add(int(rotated_s))
        return list(rotations)

    def is_circular_prime(p: int) -> bool:
        """Проверяет, является ли число круговым простым, используя sympy.isprime."""
        if p >= 10:
            s = str(p)
            if any(c in '02468' for c in s) or ('5' in s and p != 5):
                return False

        rotations = get_cyclic_rotations(p)
        for r in rotations:
            if not isprime(r):
                return False
        return True

    circular_primes: List[int] = []

    for p in range(2, 10**6):
        if isprime(p):
            if is_circular_prime(p):
                circular_primes.append(p)
    return palindromic_squares, circular_primes

def palindromic_cubs_and_palindromic_primes() -> Tuple[List[int], List[int]]:
    """
    Возвращает:
    tuple:
    - список палиндромов a < 100000, где a^3 тоже палиндром
    - список простых палиндромов p ≤ 10000
    """
    palindromic_cubs: List[int] = []
    for a in range(1,10**5):
        if is_palindrome(a) and is_palindrome(a**3):
            palindromic_cubs.append(a)

    palindromic_primes: List[int] = []
    for p in range(1,10001):
        if is_palindrome(p) and isprime(p):
            palindromic_primes.append(p)
    return palindromic_cubs, palindromic_primes

def primes_with_two_digits() -> Dict[str, List[int]]:
    """
    Возвращает словарь:
    {
    '13': [первые 100 простых из цифр {1,3}],
    '15': [первые 100 простых из цифр {1,5}],
    '17': [первые 100 простых из цифр {1,7}],
    '19': [первые 100 простых из цифр {1,9}]
    }
    """
    def generate_numbers(digits, limit=100):
        primes = []
        max_len = 20
        queue = [str(digits[0]), str(digits[1])]
        while queue and len(primes) < limit:
            num_str = queue.pop(0)
            num = int(num_str)
            if num > 1 and isprime(num):
                primes.append(num)
                if len(primes) == limit:
                    break
            if len(num_str) < max_len:
                for d in digits:
                    new_num_str = num_str + str(d)
                    queue.append(new_num_str)
        return primes

    digit_pairs = [(1, 3), (1, 5), (1, 7), (1, 9)]
    results = {}
    for pair in digit_pairs:
        results[f"1{str(pair[1])}"] = generate_numbers(pair, 100)
    print(results)
    return results


def twin_primes_analysis(limit_pairs: int = 1000) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Возвращает:
    - список первых `limit_pairs` пар простых близнецов (p, p+2)
    - список значений отношения pi_2(n) / pi(n) для n соответствующих последним элементам пар
    """
    num = 0
    pi = 0
    pi_2 = 0
    div_pi2pi = []
    list_limit_pairs = []
    while len(list_limit_pairs) < limit_pairs:
        num += 1
        if isprime(num):
            pi += 1
        if isprime(num) and isprime(num + 2):
            list_limit_pairs.append((num, num + 2))
            div_pi2pi.append(pi_2 / pi)
            pi_2 += 1
    print(list_limit_pairs)
    print(div_pi2pi)
    return list_limit_pairs, div_pi2pi

def factorial_plus_one_factors() -> Dict[int, Dict[int, int]]:
    """
    Возвращает словарь:
    {n: {простой_делитель: степень, ...}, ...}
    для n от 2 до 50, где значение - разложение n! + 1 на простые множители
    """
    num_big_prime_digit = {}
    dict_for_numbers = {}
    for num in range(2, 51):
        dict_for_number = {}
        number = factorial(num) + 1
        list_prime = primefactors(number)
        for i in list_prime:
            dict_for_number[i] = list_prime.count(i)
        dict_for_numbers[num] = dict_for_number
    print(dict_for_numbers)
    print(num_big_prime_digit)
    return dict_for_numbers

def euler_phi_direct(n: int) -> int:
    """Вычисляет (n) прямым перебором."""
    count = 0
    if n <= 0:
        return 0
    for num in range(1, n + 1):
        if gcd(num, n) == 1:
            count += 1
    print(count)
    return count


def euler_phi_factor(n: int) -> int:
    """Вычисляет (n) через разложение на простые множители."""
    primefactors_n = primefactors(n)
    phi = n
    for prime in primefactors_n:
        phi = int(phi * (1 - 1 / prime))
    print(phi)
    return phi


def compare_euler_phi_methods(test_values: List[int]) -> dict:
    """
    Сравнивает время работы трёх методов на заданных значениях.
    Возвращает словарь с тремя списками времён (в секундах).
    """
    time_results = {}
    for test_value in test_values:
        method_times = {}

        start_time1 = time.perf_counter()
        res1 = euler_phi_direct(test_value)
        time1 = time.perf_counter() - start_time1
        method_times['direct'] = time1

        start_time2 = time.perf_counter()
        res2 = euler_phi_factor(test_value)
        time2 = time.perf_counter() - start_time2
        method_times['factor'] = time2

        start_time3 = time.perf_counter()
        res3 = totient(test_value)
        time3 = time.perf_counter() - start_time3
        method_times['sympy'] = time3

        time_results[test_value] = {'times': method_times}
    print(time_results)
    return time_results


compare_euler_phi_methods([1684, 98, 77])













