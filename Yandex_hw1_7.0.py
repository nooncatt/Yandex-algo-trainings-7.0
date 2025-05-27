# Task A: https://contest.yandex.ru/contest/74964/problems/A/
def count_sort_arr(arr):
    # find max key for the first element in pair
    max_val = max(item[0] for item in arr)
    counts = [[] for _ in range(max_val + 1)]

    new_arr = []
    for i in range(len(arr)):
        counts[arr[i][0]].append(arr[i][1])
    for i in range(len(counts)):
        if counts[i]:
            for j in range(len(counts[i])):
                new_arr.append([i, counts[i][j]])
    return new_arr


s = list(map(int, input().split()))
N = s[0]
M = s[1]

students = list(map(int, input().split()))
computers = list(map(int, input().split()))

computers_n = [[computers[i], i + 1] for i in range(len(computers))]

suit_classroom = [0] * (N + 1)
counter = 0

computers = count_sort_arr(computers_n)

for i in range(len(students)):
    for j in range(len(computers)):
        if computers[j] != 0:
            if students[i] + 1 <= computers[j][0]:
                counter += 1
                suit_classroom[i + 1] = computers[j][1]
                computers[j] = 0
                break  # break the cycle when found suitable j
print(counter)
print(" ".join(map(str, suit_classroom[1:])))


# Task B: https://contest.yandex.ru/contest/74964/problems/B/
t = int(input())
for _ in range(t):
    n = int(input())
    nums = list(map(int, input().split()))

    k = 0
    current_length = 0
    current_min = float("inf")

    answer = ""

    for i in range(len(nums)):
        current_min = min(current_min, nums[i])

        if current_length < current_min:
            current_length += 1

        else:  # ==
            k += 1
            answer += f"{current_length} "
            current_length = 1
            current_min = nums[i]

        if i == n - 1:
            k += 1
            answer += f"{current_length} "

    print(k)
    print(answer)


# Task C:
M = int(input().strip())
a = list(map(int, input().split()))

# Preprocessing: for i from 1 to 30, ensure that a[i] >= 2 * a[i - 1],
# otherwise, buying two cheaper cards might be more profitable than buying one type I card.
for i in range(1, 31):
    a[i] = max(a[i], 2 * a[i - 1])

INF = 10**18
ans = INF
cost = 0
remain = M

# A greedy passage through the types of cards from the most expensive to the cheapest.
# The price of card i is 2^i, and a[i] is the number of seconds it gives.
for i in range(30, -1, -1):
    # We buy as many cards of this type as possible, which completely cover the remaining seconds.
    count = remain // a[i]
    cost += count * (1 << i)
    remain %= a[i]
    # At the same time, we evaluate a possible solution:
    # If there is a non-zero balance, you can buy another card of this type.,
    # to cover it (even if it takes more seconds than it takes).
    ans = min(ans, cost + (remain > 0) * (1 << i))

print(ans)


# Task D: https://contest.yandex.ru/contest/74964/problems/D/
# s = list(map(int, input().split()))
# N = s[0]
# M = s[1]
#
# s1 = list(map(int, input().split())
# arr = [-1] * (M + 1)
#
# arr[0] = 0
#
# for i in range(len(s1)):
#     for j in range(len(arr) - s1[i] - 1, -1, -1):
#         if arr[j] != -1:
#             arr[j + s1[i]] = "yes"
# for i in range(len(arr) - 1, -1, -1):
#     if arr[i] == "yes":
#         print(i)
#         break


# Task E: https://contest.yandex.ru/contest/74964/problems/E/
s = list(map(int, input().split()))
N = s[0]
M = s[1]

weights = list(map(int, input().split()))
costs = list(map(int, input().split()))

answer = [-1] * (M + 1)
answer[0] = 0


for i in range(len(weights)):
    for j in range(len(answer) - weights[i] - 1, -1, -1):
        if answer[j] != -1:
            c = answer[j] + costs[i]  # new cost in the same weight
            answer[j + weights[i]] = max(answer[j + weights[i]], c)  # before or new

print(max(answer))


# Task F: https://contest.yandex.ru/contest/74964/problems/F/
import copy

data = list(map(int, input().split()))
if not data:
    exit(0)

N = int(data[0])
M = int(data[1])
weights = list(map(int, input().split()))
costs = list(map(int, input().split()))

# if we have no items which smaller than rucksack capacity
if all(w > M for w in weights):
    exit(0)

rucksack = [[-1 for _ in range(M + 1)] for _ in range(N + 1)]

# base
for row in range(N + 1):
    rucksack[row][0] = [0, 0]
for col in range(M + 1):
    rucksack[0][col] = [0, 0]

for row in range(1, N + 1):
    for col in range(M - weights[row - 1], -1, -1):
        if rucksack[row][col] != -1:
            nxt = col + weights[row - 1]
            if nxt <= M:
                if rucksack[row][nxt] == -1:
                    rucksack[row][nxt] = [
                        rucksack[row][col][0] + costs[row - 1],
                        row,
                    ]
                else:
                    cost_val = rucksack[row][col][0] + costs[row - 1]
                    if cost_val > rucksack[row][nxt][0]:  # new > old
                        rucksack[row][nxt] = [cost_val, row]
                    else:
                        continue
    if row < N:  # copy prev row to new only if we have the next row
        rucksack[row + 1] = copy.deepcopy(rucksack[row])

# get numbers of items
# firstly find the max cost's weight and its corresponding "last_item"
best = [-1, -1, 0]  # [cost, weight, last_item]
for col in range(M, 0, -1):
    cell = rucksack[N][col]
    if cell != -1 and cell[0] > best[0]:
        best = [cell[0], col, cell[1]]  # cost, weight, number

# return answer using minimal changes:
nums = []
current_weight = best[1]  # current total weight (index in DP)
current_item = best[2]  # last chosen item (номер строки, начиная с 1)

# Restoring the path: if the last item in the optimal state is i,
# then the previous state should be in row i-1 and weight in current_weight - weights[i-1].
while current_weight > 0 and current_item > 0:
    nums.append(current_item)
    new_weight = current_weight - weights[current_item - 1]
    if new_weight <= 0:
        break
    # We check that the DP table has a state for the previous item at new_weight
    if rucksack[current_item - 1][new_weight] == -1:
        break
    current_item = rucksack[current_item - 1][new_weight][1]
    current_weight = new_weight

for item in nums[::-1]:
    print(item)
