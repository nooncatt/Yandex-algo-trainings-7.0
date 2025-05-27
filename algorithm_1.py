# Оценка сложности и времени работы алгоритмов


# Поиск минимума и максимума в массиве
def gen_min(a):
    ans = a[0]
    for i in range(1, len(a)):
        if ans > a[i]:
            ans = a[i]
    return ans


print(gen_min([2, 2, 3, 4, 6, 2, 3, 1]))


class Pair:
    def __init__(self):
        self.min = None
        self.max = None


def getMinMax(arr: list, n: int) -> Pair:
    minmax = Pair()
    # If there is only one element then return it as min and max both
    if n == 1:
        minmax.max = arr[0]
        minmax.min = arr[0]
    # If there are more than one elements, then initialize min
    # and max
    if arr[0] > arr[1]:
        minmax.max = arr[0]
        minmax.min = arr[1]
    else:
        minmax.max = arr[1]
        minmax.min = arr[0]
    for i in range(2, n):
        if arr[i] > minmax.max:
            minmax.max = arr[i]
        elif arr[i] < minmax.min:
            minmax.min = arr[i]
    return minmax


arr = [1, 2, 3, 4, 5, 6, 7, 3, 4, 3, 6, 7, 10]
answer = getMinMax(arr, len(arr))
print(answer.min)
print(answer.max)


def getMinMax(matrix: list) -> int:
    ans = matrix[0][0]
    for row in matrix:
        for element in row:
            if element < ans:
                ans = element
    return ans


matr = [[5, 2, 3], [1, 5, 3]]
print(getMinMax(matr))
# O(n^p) - p вложенных циклов


# SORT

# Сортировка пузырьковая

from random import randint

N = 10
a = []

for i in range(N):
    a.append(randint(1, 99))
print(a)

for i in range(N - 1):
    for j in range(N - i - 1):
        if a[j] > a[j + 1]:
            a[j], a[j + 1] = a[j + 1], a[j]
print(a)


# selection sort Сортировка выбором


def selection_sort(alist):
    for i in range(1, len(alist) - 1):
        smallest = i
        for j in range(i + 1, len(alist)):
            if alist[j] < alist[smallest]:
                smallest = j
                alist[i], alist[smallest] = alist[smallest], alist[i]


alist = input("Enter the list of numbers").split()
alist = [int(x) for x in alist]
print(selection_sort(alist))


# Incertion sort Сортировка вставков
def insertion_sort(array):
    n = len(array)
    for i in range(1, n):
        x = array[i]
        j = i

        while j > 0 and array[j - 1] > x:
            array[j] = array[j - 1]
            j -= 1

        array[j] = x

    return array


insertion_sort([6, 5, 3, 8, 9, 1])


# Merge sort
# n * log n по основанию 2
# крутая сортировка, меньше времени


def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    # create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)

    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    # Merge the temp arrays back into arr[l..r]
    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = l  # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], if there are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1


def mergeSort(arr, l, r):
    if l < r:
        # Same as (l+r)//2, but avoids overflow for large l and r
        m = l + (r - l) // 2

        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)


# Driver code to test above
arr = [12, 11, 13, 5, 6, 7]
n = len(arr)
print("Given array is")
for i in range(n):
    print("%d" % arr[i], end=" ")

mergeSort(arr, 0, n - 1)

print("\n\nSorted array is")
for i in range(n):
    print("%d" % arr[i], end=" ")


# Quick sort

import random


def quickSort(array):  #
    if len(array) <= 1:
        return array

    pivot_index = random.randint(0, len(array) - 1)
    pivot = array[pivot_index]
    lower = []
    equal = []
    greater = []

    for x in array:
        if x < pivot:
            lower.append(x)
        elif x == pivot:
            equal.append(x)
        else:
            greater.append(x)

    lower = quickSort(lower)
    greater = quickSort(greater)
    return lower + equal + greater


if __name__ == "__main__":
    n = int(input())
    data = [int(s) for s in input().split()]
    # data = [i for i in range(n)]
    data = quickSort(data)
    print(*data)


# покороче
def quickSort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
        s_nums = []
        m_nums = []
        e_nums = []
        for n in nums:
            if n < q:
                s_nums.append(n)
            elif n > q:
                m_nums.append(n)
            else:
                e_nums.append(n)
        return quickSort(s_nums) + e_nums + quickSort(m_nums)


if __name__ == "__main__":
    n = int(input())
    # data = [int(s) for s in input().split()]
    data = [random.randint(1, 1000) for i in range(n)]
    print(*data)
    data = quickSort(data)
    print(*data)


# используя Quicksort using list comprehensions
def qsort1(nums):
    if len(nums) <= 1:
        return nums
    else:
        pivot = list[0]
        lesser = qsort1([x for x in list[1:] if x < pivot])
        greater = qsort1([x for x in list[1:] if x >= pivot])
        return lesser + [pivot] + greater


# COUNT SORT работает за O(n)
def count_sort(arr):
    counts = [0] * 101  # max(arr) максимальный элемент , это гениально
    result = []
    for elem in arr:
        counts[elem] += 1

    for index, c in enumerate(counts):
        if c == 0:
            continue
        elif c == 1:
            result.append(index)
        else:
            b = [index for _ in range(1, c + 1)]
            result += b

    return result



# еще один вариант COUNT SORT
def count_sort1(arr):
    n = int(input())
    arr = [int(s) for s in input().split()]
    counts = [0] * 101  # max(arr) максимальный элемент , это гениально

    for elem in arr:
        counts[elem] += 1

    for i in range(101):
        for c in range(counts[c]):
            print(i, end=" ")




# тут платим памятью
# можно платить временем


# QUICK SORT более быстрая, меньше создаем
import random


def partition(array, low, high):
    if low >= high:
        return

    # Случайный выбор опорного элемента
    pivot_index = random.randint(low, high)
    pivot = array[pivot_index]
    array[pivot_index], array[low] = array[low], array[pivot_index]

    p1 = low - 1
    p2 = low

    # Проходим по массиву от low до high включительно
    for j in range(low + 1, high + 1):
        if array[j] < pivot:
            p1 += 1
            p2 += 1
            array[p1], array[p2] = array[p2], array[p1]
            if j != p2:
                (array[p1], array[j]) = (array[j], array[p1])
        elif array[j] == pivot:
            p2 += 1
            (array[p2], array[j]) = (array[j], array[p2])
    return (p1, p2 + 1)


def quickSort(array, low, high):
    if low < high:
        (pi, nxt) = partition(array, low, high)

        quickSort(array, low, pi)
        quickSort(array, nxt, high)


def test_qsort():
    for _ in range(100000):
        n = 1000
        a = [random.randint(1, 1000) for _ in range(n)]
        b = a.copy()
        # Сортируем встроенным методом для проверки
        b.sort()
        # Сортируем нашим quickSort
        quickSort(a, 0, n - 1)
        # Проверяем, совпадает ли результат
        assert a == b


if __name__ == "__main__":
    # Пример чтения входных данных из одной строки
    # Например, пользователь вводит: 3 1 2 5 4
    a = [int(s) for s in input().split()]
    quickSort(a, 0, len(a) - 1)
    print(*a)




# то же самое другими словами
# QUICK SORT

import random


def partition_3way(arr, low, high):
    # Выбираем случайный опорный элемент и ставим его в начало
    pivot_index = random.randint(low, high)
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    pivot = arr[low]

    lt = low  # arr[low..lt-1] < pivot
    i = low + 1  # текущий индекс
    gt = high  # arr[gt+1..high] > pivot

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1
    return lt, gt


def quickSort_3way(arr, low, high):
    if low < high:
        lt, gt = partition_3way(arr, low, high)
        quickSort_3way(arr, low, lt - 1)
        quickSort_3way(arr, gt + 1, high)


# Пример использования:
if __name__ == "__main__":
    a = [int(s) for s in input().split()]
    quickSort_3way(a, 0, len(a) - 1)
    print(*a)
# 5 4 3 1 2




# Сортировка для цветов 0, 1, 2
def sort_012(arr):
    low = 0  # Конец зоны с 0 (первые элементы массива)
    mid = 0  # Текущий элемент для обработки
    high = len(arr) - 1  # Начало зоны с 2 (конец массива)

    while mid <= high:
        if arr[mid] == 0:
            # Меняем элемент с arr[mid] с элементом в начале незаполненной зоны 0
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            # Если элемент равен 1, он уже на своём месте
            mid += 1
        else:  # arr[mid] == 2
            # Меняем элемент с arr[mid] с элементом в конце массива (зона >1)
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr