# Task A: https://contest.yandex.ru/contest/74966/problems/
n = int(input())
arr = list(map(int, input().split()))

new_length = 0

for i in range(18):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [[arr[i - n + 1], 1] if i >= n - 1 else [0, 0] for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 1 - 1, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    maxval = max(left[0], right[0])
    if left[0] == right[0]:
        tree[i] = [maxval, left[1] + right[1]]
    elif left[0] > right[0]:  # left son > right son
        tree[i] = [maxval, left[1]]
    else:  # left son < right son
        tree[i] = [maxval, right[1]]


INF = -1


def rmq(numb, seg_l, seg_r, L, R):
    """
    numb   — индекс узла в tree
    [seg_l..seg_r] — границы, за которые отвечает этот узел
    [L..R]         — запрошенный отрезок (включительно)
    """
    # 1) покрываем полностью
    if L <= seg_l and seg_r <= R:
        return tree[numb][0], tree[numb][1]

    # 2) не подходит, нет пересечения
    elif seg_r < L or seg_l > R:
        return (INF, 0)

    # 3) частичное — идём в детей
    else:
        mid = (seg_l + seg_r) // 2
        # index of children 2i+1 and 2i+2
        left_max, left_count = rmq(2 * numb + 1, seg_l, mid, L, R)
        right_max, right_count = rmq(2 * numb + 2, mid + 1, seg_r, L, R)

        # слияние двух ответов
        if left_max > right_max:
            return left_max, left_count
        if left_max < right_max:
            return right_max, right_count
        return left_max, left_count + right_count


# отвечаем на запросы
K = int(input())
for _ in range(K):
    L, R = map(int, input().split())
    mx, cnt = rmq(0, 0, n - 1, L - 1, R - 1)
    print(mx, cnt)


# Task B: https://contest.yandex.ru/contest/74966/problems/B/
n = int(input())
arr = list(map(int, input().split()))

new_length = 0

for i in range(18):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [[arr[i - n + 1], i - n + 1] if i >= n - 1 else [0, 0] for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 1 - 1, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    maxval = max(left[0], right[0])

    if left[0] >= right[0]:  # left son >= right son
        tree[i] = [maxval, left[1]]
    else:  # left son < right son
        tree[i] = [maxval, right[1]]


INF = -1


def rmq(numb, seg_l, seg_r, L, R):
    """
    numb   — индекс узла в tree
    [seg_l..seg_r] — границы, за которые отвечает этот узел
    [L..R]         — запрошенный отрезок (включительно)
    """
    # 1) покрываем полностью
    if L <= seg_l and seg_r <= R:
        return tree[numb][0], tree[numb][1]

    # 2) не подходит, нет пересечения
    elif seg_r < L or seg_l > R:
        return (INF, 0)

    # 3) частичное — идём в детей
    else:
        mid = (seg_l + seg_r) // 2
        # index of children 2i+1 and 2i+2
        left_max, left_idx = rmq(2 * numb + 1, seg_l, mid, L, R)
        right_max, right_idx = rmq(2 * numb + 2, mid + 1, seg_r, L, R)

        # слияние двух ответов
        if left_max >= right_max:
            return left_max, left_idx
        else:  # left_max < right_max:
            return right_max, right_idx


# отвечаем на запросы
K = int(input())
for _ in range(K):
    L, R = map(int, input().split())
    mx, idx = rmq(0, 0, n - 1, L - 1, R - 1)
    print(idx + 1)  # сдвиг т к индексы с 1


# Task C: https://contest.yandex.ru/contest/74966/problems/C/
n = int(input())
arr = list(map(int, input().split()))

new_length = 0

for i in range(18):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [[arr[i - n + 1], i - n + 1] if i >= n - 1 else [0, 0] for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 1 - 1, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    maxval = max(left[0], right[0])

    if left[0] >= right[0]:  # left son >= right son
        tree[i] = [maxval, left[1]]
    else:  # left son < right son
        tree[i] = [maxval, right[1]]


INF = -1


def rmq(numb, seg_l, seg_r, L, R):
    """
    numb   — индекс узла в tree
    [seg_l..seg_r] — границы, за которые отвечает этот узел
    [L..R]         — запрошенный отрезок (включительно)
    """
    # 1) покрываем полностью
    if L <= seg_l and seg_r <= R:
        return tree[numb][0], tree[numb][1]

    # 2) не подходит, нет пересечения
    elif seg_r < L or seg_l > R:
        return (INF, 0)

    # 3) частичное — идём в детей
    else:
        mid = (seg_l + seg_r) // 2
        # index of children 2i+1 and 2i+2
        left_max, left_idx = rmq(2 * numb + 1, seg_l, mid, L, R)
        right_max, right_idx = rmq(2 * numb + 2, mid + 1, seg_r, L, R)

        # слияние двух ответов
        if left_max >= right_max:
            return left_max, left_idx
        else:  # left_max < right_max:
            return right_max, right_idx


# отвечаем на запросы
K = int(input())
for _ in range(K):
    L, R = map(int, input().split())
    mx, idx = rmq(0, 0, n - 1, L - 1, R - 1)
    print(mx, idx + 1)  # сдвиг т к индексы с 1


# Task D: https://contest.yandex.ru/contest/74966/problems/D/
n = int(input())
arr = list(map(int, input().split()))

new_length = 0

for i in range(18):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        level = i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [arr[i - n + 1] if i >= n - 1 else 0 for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 1 - 1, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]

    tree[i] = max(left, right)


INF = -1


def rmq(numb, seg_l, seg_r, L, R):
    """
    numb   — индекс узла в tree
    [seg_l..seg_r] — границы, за которые отвечает этот узел
    [L..R]         — запрошенный отрезок (включительно)
    """
    # 1) покрываем полностью
    if L <= seg_l and seg_r <= R:
        return tree[numb]

    # 2) не подходит, нет пересечения
    elif seg_r < L or seg_l > R:
        return INF

    # 3) частичное — идём в детей
    else:
        mid = (seg_l + seg_r) // 2
        # index of children 2i+1 and 2i+2
        left_max = rmq(2 * numb + 1, seg_l, mid, L, R)
        right_max = rmq(2 * numb + 2, mid + 1, seg_r, L, R)

        # слияние двух ответов
        if left_max >= right_max:
            return left_max
        else:  # left_max < right_max:
            return right_max


def change_elem(idx, new_val):
    # change tree
    tree_idx = n - 1 + idx
    tree[tree_idx] = new_val

    while tree_idx > 0:
        tree_idx = (tree_idx - 1) // 2
        tree[tree_idx] = max(tree[2 * tree_idx + 1], tree[2 * tree_idx + 2])


# отвечаем на запросы
M = int(input())
answer = ""
for _ in range(M):
    req = input().split()
    if req[0] == "s":
        L, R = int(req[1]), int(req[2])
        mx = rmq(0, 0, n - 1, L - 1, R - 1)
        answer += f"{mx} "

    if req[0] == "u":
        elem_idx, new_val = int(req[1]) - 1, int(req[2])  # фиксируем сдвиг
        change_elem(elem_idx, new_val)
print(answer)


# Task E: https://contest.yandex.ru/contest/74966/problems/E/
import sys

input = sys.stdin.readline

n = int(input())
arr = list(map(int, input().split()))

new_length = 0

for i in range(19):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        level = i
        break

# extend our array to 2**i
arr = arr + [10**6 for i in range(new_length - n)]

# update n
n = new_length

tree = [[arr[i - n + 1], 1] if i >= n - 1 else [0, 0] for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 1 - 1, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    minval = min(left[0], right[0])

    if left[0] == right[0]:
        tree[i] = [minval, left[1] + right[1]]
    elif left[0] > right[0]:  # left son > right son
        tree[i] = [minval, right[1]]
    else:  # left son < right son
        tree[i] = [minval, left[1]]


INF = 10**6


# Функция для подсчета количества нулей на отрезке [0, L-1]


def count_zeros(numb, seg_l, seg_r, L, R):
    """
    Считает, сколько нулей на отрезке [L..R] в исходном массиве,
    используя дерево tree, где в каждом узле хранится [min, count].
    """
    # 1) Нет пересечения - ноль нулей
    if seg_r < L or seg_l > R:
        return 0

    # 2) Полное покрытие
    if L <= seg_l and seg_r <= R:
        node = tree[numb]
        return node[1] if node[0] == 0 else 0

    # 3) Частичное покрытие — спускаемся по детям
    mid = (seg_l + seg_r) // 2
    left = count_zeros(2 * numb + 1, seg_l, mid, L, R)
    right = count_zeros(2 * numb + 2, mid + 1, seg_r, L, R)
    return left + right


# Функция для поиска k-го нуля на отрезке
def find_kth_zero(l, r, k, left=0, right=n - 1, idx=0):
    # Если k больше общего числа нулей в дереве
    total_zeros = tree[idx][1] if tree[idx][0] == 0 else 0
    if k > total_zeros:
        return -1

    if left == right and tree[idx][0] == 0:
        return left

    mid = (left + right) // 2

    if k <= tree[idx * 2 + 1][1] and tree[idx * 2 + 1][0] == 0:
        return find_kth_zero(l, r, k, left, mid, idx * 2 + 1)
    else:
        if tree[idx * 2 + 1][0] == 0:
            k -= tree[idx * 2 + 1][1]
        return find_kth_zero(l, r, k, mid + 1, right, idx * 2 + 2)


def change_elem(idx, new_val):
    # change tree
    tree_idx = n - 1 + idx
    tree[tree_idx][0] = new_val
    tree[tree_idx][1] = 1

    while tree_idx > 0:
        tree_idx = (tree_idx - 1) // 2

        left = tree[2 * tree_idx + 1]
        right = tree[2 * tree_idx + 2]

        minval = min(left[0], right[0])

        if left[0] == right[0]:
            tree[tree_idx] = [minval, left[1] + right[1]]
        elif left[0] > right[0]:  # left son > right son
            tree[tree_idx] = [minval, right[1]]
        else:  # left son < right son
            tree[tree_idx] = [minval, left[1]]

        # update arr up-to-date
        # arr[idx] = new_val


# отвечаем на запросы
M = int(input())
ans = []
for _ in range(M):
    req = input().split()
    if req[0] == "s":
        if tree[0][0] != 0:  # we don't have zeroes
            ans.append(-1)
        else:
            L, R, k = int(req[1]), int(req[2]), int(req[3])
            if k > tree[0][1]:
                ans.append(-1)
            else:
                # прибавляем количество нулей на отрезке от 0 до L-1
                zeros_before_L = count_zeros(0, 0, n - 1, 0, L - 2) if L > 1 else 0
                k += zeros_before_L
                result = find_kth_zero(L - 1, R - 1, k, 0, n - 1, 0) + 1  # т к счит с 0
                if L <= result <= R:  # т к был сдвиг
                    ans.append(result)
                else:
                    ans.append(-1)

    if req[0] == "u":
        elem_idx, new_val = int(req[1]) - 1, int(req[2])  # фиксируем сдвиг
        change_elem(elem_idx, new_val)
print(*ans)


# Task F: https://contest.yandex.ru/contest/74966/problems/F/
import sys

input = sys.stdin.readline

n, m = map(int, input().split())
a = list(map(int, input().split()))

new_length = 0

for i in range(19):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

old_length = n

# extend our array to 2**i
arr = a + [-1 for i in range(new_length - n)]


# update n
n = new_length

tree = [arr[i - n + 1] if i >= n - 1 else 0 for i in range(2 * n - 1)]

# fulfill the tree
for i in range(n - 2, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    tree[i] = max(left, right)


# Функция для поиска первого числа >= x
def find_number_better_x(l, r, x, left=0, right=new_length - 1, idx=1):
    # Если текущий отрезок не пересекается с [l, r]
    if right < l or left > r:
        return -1

    # Если весь отрезок находится внутри [l, r]
    if left >= l and right <= r:
        # Если максимальное значение на отрезке меньше x, то возвращаем -1
        if tree[idx] < x:
            return -1
        # Если весь отрезок соответствует запросу, ищем в левом и правом поддеревьях
        if left == right:
            if tree[idx] >= x:
                return left
            else:
                return -1

    # Делаем рекурсию
    mid = (left + right) // 2
    left_result = find_number_better_x(l, r, x, left, mid, 2 * idx)
    if left_result != -1:
        return left_result

    return find_number_better_x(l, r, x, mid + 1, right, 2 * idx + 1)


# def find_number_better_x(l, r, x, left=0, right=n - 1, idx=0):
#
#     if left == right:
#         if tree[idx] >= x and left >= l:
#             return left
#         return -1
#
#     mid = (left + right) // 2
#
#     # если левый сын >= x, ищем в левом поддереве
#     if tree[idx * 2 + 1] >= x and mid >= l:  # т е правая граница левого сына должна >=l
#         return find_number_better_x(l, r, x, left, mid, idx * 2 + 1)
#     else:
#         return find_number_better_x(l, r, x, mid + 1, right, idx * 2 + 2)


def change_elem(idx, new_val):
    # change tree
    # tree_idx = n - 1 + idx
    tree_idx = new_length + idx - 1
    tree[tree_idx] = new_val

    while tree_idx > 0:
        tree_idx = (tree_idx - 1) // 2

        left = tree[2 * tree_idx + 1]
        right = tree[2 * tree_idx + 2]

        tree[tree_idx] = max(left, right)


# отвечаем на запросы
for _ in range(m):
    req = input().split()
    if req[0] == "1":
        i, x = int(req[1]), int(req[2])
        if tree[0] < x or old_length < i:  # max < x или кол-во элементов меньше i-го
            print(-1)
        else:
            res = find_number_better_x(i - 1, n - 1, x, 0, n - 1, 0) + 1  # т к счит с 0
            print(res)

    if req[0] == "0":
        i, x = int(req[1]) - 1, int(req[2])  # фиксируем сдвиг
        change_elem(i, x)


# Task H: https://contest.yandex.ru/contest/74966/problems/H/
import sys

input = sys.stdin.readline

n = int(input())
arr = list(map(int, input().split()))
m = int(input())

new_length = 0

for i in range(19):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [0] * (2 * n - 1)
for i in range(n):
    tree[n - 1 + i] = arr[i]

# fulfill the tree
for i in range(n - 2, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    tree[i] = max(left, right)

lazy = [0] * (2 * n - 1)


# Функция для обновления сегмента
def update_range(node, start, end, L, R, value):
    if lazy[node] != 0:
        # выполняем обещания
        tree[node] += lazy[node]
        if start != end:
            lazy[2 * node + 1] += lazy[node]
            lazy[2 * node + 2] += lazy[node]
        lazy[node] = 0

    if start > end or start > R or end < L:
        return

    if start >= L and end <= R:
        tree[node] += value
        if start != end:
            lazy[2 * node + 1] += value
            lazy[2 * node + 2] += value
        return

    mid = (start + end) // 2

    update_range(2 * node + 1, start, mid, L, R, value)
    update_range(2 * node + 2, mid + 1, end, L, R, value)
    tree[node] = max(tree[2 * node + 1], tree[2 * node + 2])


# Функция для запроса текущего значения на отрезке
def query(node, start, end, idx):
    if lazy[node] != 0:
        # выполняем обещание
        tree[node] += lazy[node]
        if start != end:
            lazy[2 * node + 1] += lazy[node]
            lazy[2 * node + 2] += lazy[node]
        lazy[node] = 0

    if start == end:
        return tree[node]

    mid = (start + end) // 2
    if start <= idx <= mid:
        return query(2 * node + 1, start, mid, idx)
    else:
        return query(2 * node + 2, mid + 1, end, idx)


# Обработка запросов
ans = []
for _ in range(m):
    req = input().split()
    if req[0] == "g":
        # get element
        idx = int(req[1]) - 1
        ans.append(query(0, 0, n - 1, idx))
    elif req[0] == "a":
        # update elements
        L, R, add = int(req[1]) - 1, int(req[2]) - 1, int(req[3])
        update_range(0, 0, n - 1, L, R, add)

print(*ans)


# Task I:
import sys

input = sys.stdin.readline

n = int(input())
arr = list(map(int, input().split()))
m = int(input())

new_length = 0

for i in range(19):
    if n > 2**i:
        continue
    else:  # <=
        new_length = 2**i
        break

# extend our array to 2**i
arr = arr + [-1 for i in range(new_length - n)]

# update n
n = new_length

tree = [0] * (2 * n - 1)
for i in range(n):
    tree[n - 1 + i] = arr[i]

# fulfill the tree
for i in range(n - 2, -1, -1):
    left = tree[2 * i + 1]
    right = tree[2 * i + 2]
    tree[i] = max(left, right)

lazy = [0] * (2 * n - 1)


# Функция для обновления сегмента
def update_range(node, start, end, L, R, value):
    if lazy[node] != 0:
        # выполняем обещания
        tree[node] += lazy[node]
        if start != end:
            lazy[2 * node + 1] += lazy[node]
            lazy[2 * node + 2] += lazy[node]
        lazy[node] = 0

    if start > end or start > R or end < L:
        return

    if start >= L and end <= R:
        tree[node] += value
        if start != end:
            lazy[2 * node + 1] += value
            lazy[2 * node + 2] += value
        return

    mid = (start + end) // 2

    update_range(2 * node + 1, start, mid, L, R, value)
    update_range(2 * node + 2, mid + 1, end, L, R, value)
    tree[node] = max(tree[2 * node + 1], tree[2 * node + 2])


# Функция для запроса текущего значения на отрезке
def query(node, start, end, L, R):
    if lazy[node] != 0:
        # выполняем обещания
        tree[node] += lazy[node]
        if start != end:
            lazy[2 * node + 1] += lazy[node]
            lazy[2 * node + 2] += lazy[node]
        lazy[node] = 0

    if start > end or start > R or end < L:
        return -1  # если нет пересечения

    if start >= L and end <= R:
        return tree[node]

    mid = (start + end) // 2

    left = query(2 * node + 1, start, mid, L, R)
    right = query(2 * node + 2, mid + 1, end, L, R)
    return max(left, right)


# Обработка запросов
ans = []
for _ in range(m):
    req = input().split()
    if req[0] == "m":
        # find max
        L, R = int(req[1]) - 1, int(req[2]) - 1
        ans.append(query(0, 0, n - 1, L, R))
    elif req[0] == "a":
        # update elements
        L, R, add = int(req[1]) - 1, int(req[2]) - 1, int(req[3])
        update_range(0, 0, n - 1, L, R, add)

print(*ans)
