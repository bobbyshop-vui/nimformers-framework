"""Generate Python exercises, complete solutions and executable examples."""

from dataclasses import dataclass
from random import Random
from textwrap import dedent
from typing import Callable


def numbers(rng, low=-30, high=30):
    return [rng.randint(low, high) for _ in range(rng.randint(3, 11))]


def word(rng):
    return "".join(rng.choices("aabcdeilmnoprst", k=rng.randint(2, 10)))


def intervals(rng):
    return sorted((start, start + rng.randrange(8)) for start in numbers(rng, 0, 54))


def graph(rng, dag=False, weighted=False):
    size = rng.randint(4, 9)
    adjacency = [[] for _ in range(size)]
    for _ in range(rng.randint(size, size * 2)):
        left, right = rng.sample(range(size), 2)
        if dag and left > right:
            left, right = right, left
        edge = (right, rng.randint(1, 25)) if weighted else right
        if edge not in adjacency[left]:
            adjacency[left].append(edge)
    return adjacency


@dataclass(frozen=True)
class Exercise:
    name: str
    prompt: str
    explanation: str
    args: Callable
    edge: tuple
    answer: object
    sources: tuple[str, ...]
    solve: Callable


def exercise(name, prompt, explanation, args, edge, answer, *sources):
    sources = tuple(dedent(source).strip() for source in sources)
    functions = []
    for source in sources:
        scope = {}
        exec(compile(source, f"<{name}>", "exec"), scope)
        solve = scope["solve"]
        if solve(*edge) != answer:
            raise ValueError(f"Incorrect edge case for {name}")
        functions.append(solve)
    rng = Random(42)
    for _ in range(24):
        sample = args(rng)
        expected = functions[0](*sample)
        if any(solve(*sample) != expected for solve in functions[1:]):
            raise ValueError(f"Implementations disagree for {name}")
    return Exercise(name, prompt, explanation, args, edge, answer, sources, functions[0])


BASIC = (
    exercise("count_in_range", "Count values between lo and hi, including both endpoints.",
             "The two comparisons exclude values outside the interval and preserve duplicates.",
             lambda r: (numbers(r), -r.randrange(20), r.randrange(20)), ([], 0, 3), 0, """
        def solve(values, lo, hi):
            total = 0
            for value in values:
                if lo <= value <= hi:
                    total += 1
            return total
        """, """
        def solve(values, lo, hi):
            return sum(lo <= value <= hi for value in values)
        """),
    exercise("unique_in_order", "Keep the first occurrence of every value in an integer list.",
             "A set tests membership while the output list keeps the original order.",
             lambda r: (numbers(r, -4, 5),), ([2, 1, 2, 1],), [2, 1], """
        def solve(values):
            seen = set()
            result = []
            for value in values:
                if value not in seen:
                    seen.add(value)
                    result.append(value)
            return result
        """, """
        def solve(values):
            return list(dict.fromkeys(values))
        """),
    exercise("rotate_left", "Rotate a list k positions left. Keep an empty list empty.",
             "Reducing k modulo the length handles both large and negative offsets.",
             lambda r: (numbers(r), r.randint(-50, 50)), ([1, 2, 3], -1), [3, 1, 2], """
        def solve(values, k):
            if not values:
                return []
            k %= len(values)
            return values[k:] + values[:k]
        """),
    exercise("count_letters", "Count ASCII letters by lowercase spelling and ignore other characters.",
             "The range check excludes digits and punctuation before updating the histogram.",
             lambda r: (f"{word(r).upper()}, {word(r)} {r.randrange(100)} {word(r)}.",),
             ("Aa-1B",), {"a": 2, "b": 1}, """
        def solve(text):
            result = {}
            for char in text.lower():
                if 'a' <= char <= 'z':
                    result[char] = result.get(char, 0) + 1
            return result
        """, """
        from collections import Counter

        def solve(text):
            return dict(Counter(char for char in text.lower() if 'a' <= char <= 'z'))
        """),
    exercise("add_columns", "Add corresponding entries in two lists of equal length.",
             "Pairing values by index associates each row with the same row in the other list.",
             lambda r: ((left := numbers(r)), [r.randint(-100, 100) for _ in left]),
             ([2, -1], [6, 9]), [8, 8], """
        def solve(left, right):
            return [a + b for a, b in zip(left, right)]
        """, """
        def solve(left, right):
            result = []
            for i in range(len(left)):
                result.append(left[i] + right[i])
            return result
        """),
    exercise("sum_multiples", "Sum the positive multiples of step at most limit. Assume step > 0.",
             "There are limit // step multiples. Their coefficients form an arithmetic series.",
             lambda r: (r.randint(0, 2500), r.randint(1, 30)), (15, 5), 30, """
        def solve(limit, step):
            count = limit // step
            return step * count * (count + 1) // 2
        """, """
        def solve(limit, step):
            return sum(range(step, limit + 1, step))
        """),
    exercise("chunk_list", "Split an integer list into chunks of width k, with a shorter final chunk allowed.",
             "Move the slice start by k each time. The final slice stops at the end of the list.",
             lambda r: (numbers(r), r.randint(1, 6)), ([3, 8, 1], 2), [[3, 8], [1]], """
        def solve(values, k):
            return [values[i:i + k] for i in range(0, len(values), k)]
        """),
    exercise("column_maxima", "Find the maximum of each column in a nonempty rectangular matrix.",
             "Transposing with zip groups values that belong to the same column.",
             lambda r: ([[r.randint(-15, 20) for _ in range(4)] for _ in range(r.randint(2, 5))],),
             ([[-6, -9], [-2, -8]],), [-2, -8], """
        def solve(rows):
            return [max(column) for column in zip(*rows)]
        """),
    exercise("first_matching_index", "Return the index of the first value at least limit, or -1 if absent.",
             "Returning as soon as the condition holds preserves first-match semantics.",
             lambda r: (numbers(r), r.randint(-5, 32)), ([3, 4, 8], 9), -1, """
        def solve(values, limit):
            for index, value in enumerate(values):
                if value >= limit:
                    return index
            return -1
        """, """
        def solve(values, limit):
            return next((i for i, value in enumerate(values) if value >= limit), -1)
        """),
    exercise("convert_base", "Write a nonnegative integer in base b using digits 0 through 9 and a through z.",
             "Each remainder is the next digit from the right. Reverse the digits at the end.",
             lambda r: (r.randint(0, 200_000), r.randint(2, 36)), (31, 16), "1f", """
        def solve(value, base):
            digits = '0123456789abcdefghijklmnopqrstuvwxyz'
            if value == 0:
                return '0'
            output = []
            while value:
                value, rest = divmod(value, base)
                output.append(digits[rest])
            return ''.join(reversed(output))
        """),
    exercise("run_length_encode", "Return pairs of a character and its consecutive run length.",
             "A run ends when the character changes. Equal letters in different runs stay separate.",
             lambda r: ("".join(r.choice("abcdn") * r.randint(1, 5) for _ in range(r.randint(3, 8))),),
             ("aaabaa",), [("a", 3), ("b", 1), ("a", 2)], """
        from itertools import groupby

        def solve(text):
            return [(key, sum(1 for _ in group)) for key, group in groupby(text)]
        """),
    exercise("merge_counts", "Add two dictionaries of counts. Missing keys contribute zero.",
             "The copy gives each key an independent output slot and leaves the inputs unchanged.",
             lambda r: ({r.choice("abcdef"): r.randrange(15) for _ in range(5)},
                        {r.choice("abcdef"): r.randrange(15) for _ in range(5)}),
             ({"x": 2}, {"x": 3, "q": 1}), {"x": 5, "q": 1}, """
        def solve(left, right):
            merged = left.copy()
            for key, value in right.items():
                merged[key] = merged.get(key, 0) + value
            return merged
        """),
)

INTERMEDIATE = (
    exercise("window_sums", "Return the sums of contiguous windows of length k, with 1 <= k <= len(values).",
             "Moving one position replaces only the outgoing and incoming values.",
             lambda r: ((values := numbers(r)), r.randint(1, len(values))), ([2, -3, 6, 1], 2), [-1, 3, 7], """
        def solve(values, k):
            total = sum(values[:k])
            result = [total]
            for i in range(k, len(values)):
                total += values[i] - values[i - k]
                result.append(total)
            return result
        """, """
        from itertools import accumulate

        def solve(values, k):
            sums = list(accumulate(values, initial=0))
            return [sums[i + k] - sums[i] for i in range(len(values) - k + 1)]
        """),
    exercise("merge_intervals", "Merge overlapping closed intervals and return them ordered by start.",
             "After sorting, only the final merged interval can overlap the next interval.",
             lambda r: (intervals(r),), ([(8, 9), (2, 4), (4, 5)],), [(2, 5), (8, 9)], """
        def solve(intervals):
            merged = []
            for start, stop in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    old_start, old_stop = merged[-1]
                    merged[-1] = (old_start, max(stop, old_stop))
                else:
                    merged.append((start, stop))
            return merged
        """),
    exercise("upper_bound", "Return the insertion index after all copies of key in a sorted list.",
             "At each step the search keeps the first element greater than key inside the remaining range.",
             lambda r: (sorted(numbers(r, -5, 15)), r.randrange(14)), ([2, 2, 2, 3], 2), 3, """
        def solve(values, key):
            low, high = 0, len(values)
            while low < high:
                mid = (low + high) // 2
                if values[mid] <= key:
                    low = mid + 1
                else:
                    high = mid
            return low
        """, """
        from bisect import bisect_right

        def solve(values, key):
            return bisect_right(values, key)
        """),
    exercise("balanced_brackets", "Report whether all brackets are balanced in a string of ()[]{}.",
             "The most recent opening bracket determines which character may close it.",
             lambda r: (r.choice(("([]{})", "([)]", "[]", "{}")) +
                        "".join(r.choices("()[]{}", k=r.randint(1, 22))),), ("([)]",), False, """
        def solve(text):
            closing = {'(': ')', '[': ']', '{': '}'}
            pending = []
            for char in text:
                if char in closing:
                    pending.append(closing[char])
                elif not pending or pending.pop() != char:
                    return False
            return not pending
        """),
    exercise("range_sums", "Answer the sums on zero-based half-open ranges [left, right).",
             "Two prefix sums isolate the requested interval in constant time per query.",
             lambda r: ((values := numbers(r)), [tuple(sorted(r.choices(range(len(values) + 1), k=2)))
                                                for _ in range(r.randint(2, 6))]),
             ([2, -4, 7], [(0, 0), (0, 3), (1, 2)]), [0, 5, -4], """
        from itertools import accumulate

        def solve(values, queries):
            prefix = list(accumulate(values, initial=0))
            return [prefix[right] - prefix[left] for left, right in queries]
        """),
    exercise("top_frequencies", "Find the k most common words. Break frequency ties alphabetically.",
             "Sorting by the pair of negative count and word applies both ordering rules.",
             lambda r: (r.choices([word(r) for _ in range(5)], k=r.randint(7, 18)), r.randint(1, 5)),
             (["beta", "alpha", "beta", "alpha"], 1), ["alpha"], """
        from collections import Counter

        def solve(words, k):
            counts = Counter(words)
            return sorted(counts, key=lambda word: (-counts[word], word))[:k]
        """),
    exercise("max_subarray", "Return the maximum sum over nonempty contiguous subarrays.",
             "The best sum ending here either starts here or extends the prior best suffix.",
             lambda r: (numbers(r),), ([-5, -1, -3],), -1, """
        def solve(values):
            ending = best = values[0]
            for value in values[1:]:
                ending = max(value, ending + value)
                best = max(best, ending)
            return best
        """),
    exercise("product_except_self", "Return the product of all other entries at every index.",
             "Multiplying a prefix by a suffix avoids division and works even when entries are zero.",
             lambda r: (numbers(r, -7, 8),), ([3, 0, 2],), [0, 6, 0], """
        def solve(values):
            result = [1] * len(values)
            prefix = 1
            for i, value in enumerate(values):
                result[i] = prefix
                prefix *= value
            suffix = 1
            for i in range(len(values) - 1, -1, -1):
                result[i] *= suffix
                suffix *= values[i]
            return result
        """),
    exercise("merge_sorted", "Merge two sorted lists into one sorted list while keeping duplicates.",
             "Each pointer advances only after emitting its next smallest remaining value.",
             lambda r: (sorted(numbers(r)), sorted(numbers(r))), ([1, 3], [1, 2]), [1, 1, 2, 3], """
        def solve(left, right):
            i = j = 0
            out = []
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    out.append(left[i])
                    i += 1
                else:
                    out.append(right[j])
                    j += 1
            return out + left[i:] + right[j:]
        """, """
        from heapq import merge

        def solve(left, right):
            return list(merge(left, right))
        """),
    exercise("schedule_intervals", "Count the maximum number of nonoverlapping half-open intervals of positive length.",
             "Choosing the interval with earliest end leaves the most room for the rest.",
             lambda r: ([(a, max(a + 1, b)) for a, b in intervals(r)],),
             ([(0, 3), (1, 2), (2, 4)],), 2, """
        def solve(intervals):
            end = None
            count = 0
            for left, right in sorted(intervals, key=lambda pair: pair[1]):
                if end is None or left >= end:
                    count += 1
                    end = right
            return count
        """),
)

ADVANCED = (
    exercise("shortest_hops", "Find the number of edges on a shortest directed path from source to target. Return -1 if unreachable.",
             "Breadth-first search visits reachable vertices in increasing distance from the source.",
             lambda r: ((adj := graph(r)), *r.sample(range(len(adj)), 2)), ([[1], [], []], 0, 2), -1, """
        from collections import deque

        def solve(adj, source, target):
            queue = deque([(source, 0)])
            visited = {source}
            while queue:
                vertex, distance = queue.popleft()
                if vertex == target:
                    return distance
                for neighbor in adj[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
            return -1
        """),
    exercise("cheapest_route", "Find the shortest path weight in a directed graph with nonnegative edge weights. Return -1 if disconnected.",
             "A priority queue processes the cheapest remaining distance. Ignore entries superseded by a better path.",
             lambda r: ((adj := graph(r, weighted=True)), *r.sample(range(len(adj)), 2)),
             ([[(1, 5), (2, 1)], [], [(1, 2)]], 0, 1), 3, """
        from heapq import heappop, heappush

        def solve(adj, source, target):
            distance = {source: 0}
            pending = [(0, source)]
            while pending:
                cost, vertex = heappop(pending)
                if cost != distance[vertex]:
                    continue
                if vertex == target:
                    return cost
                for neighbor, weight in adj[vertex]:
                    candidate = cost + weight
                    if candidate < distance.get(neighbor, float('inf')):
                        distance[neighbor] = candidate
                        heappush(pending, (candidate, neighbor))
            return -1
        """),
    exercise("order_dependencies", "Return the lexicographically first topological order, or an empty list for a graph with a cycle.",
             "Track indegrees and use the smallest available zero-indegree vertex at every step.",
             lambda r: (graph(r, dag=r.randrange(4) != 0),), ([[1], [0]],), [], """
        from heapq import heapify, heappop, heappush

        def solve(adj):
            incoming = [0] * len(adj)
            for edges in adj:
                for vertex in edges:
                    incoming[vertex] += 1
            ready = [i for i, degree in enumerate(incoming) if degree == 0]
            heapify(ready)
            order = []
            while ready:
                vertex = heappop(ready)
                order.append(vertex)
                for neighbor in adj[vertex]:
                    incoming[neighbor] -= 1
                    if incoming[neighbor] == 0:
                        heappush(ready, neighbor)
            return order if len(order) == len(adj) else []
        """),
    exercise("minimum_coins", "Find the fewest coins needed to reach amount. Each positive denomination can be used repeatedly.",
             "Compute smaller amounts first so every reachable predecessor has its minimum count.",
             lambda r: (sorted(set(numbers(r, 1, 15))), r.randint(4, 65)), ([4, 6], 5), -1, """
        def solve(coins, amount):
            missing = amount + 1
            best = [0] + [missing] * amount
            for value in range(1, amount + 1):
                for coin in coins:
                    if coin <= value:
                        best[value] = min(best[value], best[value - coin] + 1)
            return best[-1] if best[-1] != missing else -1
        """),
    exercise("knapsack_value", "Maximize the total value under a weight limit. Each item can be taken at most once.",
             "Update capacities from right to left so the current item cannot be reused.",
             lambda r: (list(zip(numbers(r, 1, 12), numbers(r, 1, 50))), r.randint(1, 29)),
             ([(2, 4), (3, 5), (4, 8)], 5), 9, """
        def solve(items, limit):
            best = [0] * (limit + 1)
            for weight, value in items:
                for capacity in range(limit, weight - 1, -1):
                    best[capacity] = max(best[capacity], best[capacity - weight] + value)
            return best[limit]
        """),
    exercise("edit_distance", "Compute Levenshtein distance with unit-cost insertions, deletions and substitutions.",
             "A dynamic-programming cell checks the three ways to align the final character of each prefix.",
             lambda r: (word(r), word(r)), ("kitten", "sitting"), 3, """
        def solve(left, right):
            previous = list(range(len(right) + 1))
            for i, a in enumerate(left, 1):
                current = [i]
                for j, b in enumerate(right, 1):
                    current.append(min(previous[j] + 1, current[j - 1] + 1,
                                       previous[j - 1] + (a != b)))
                previous = current
            return previous[-1]
        """),
    exercise("grid_paths", "Count paths from the top-left to bottom-right corner of a rectangular grid. Ones are blocked. Move only right or down.",
             "At an open cell, add the counts from above and from the left. Reset counts at obstacles.",
             lambda r: ([[int(r.random() < 0.26) for _ in range(5)] for _ in range(r.randint(2, 6))],),
             ([[0, 0, 0], [0, 1, 0], [0, 0, 0]],), 2, """
        def solve(grid):
            ways = [1] + [0] * (len(grid[0]) - 1)
            for row in grid:
                for col, blocked in enumerate(row):
                    if blocked:
                        ways[col] = 0
                    elif col > 0:
                        ways[col] += ways[col - 1]
            return ways[-1]
        """),
    exercise("longest_rising_run", "Find the length of the longest strictly increasing subsequence, with gaps allowed.",
             "The tails array keeps the smallest last value for every subsequence length.",
             lambda r: (numbers(r, -8, 20),), ([2, 2, 8, 1, 9],), 3, """
        from bisect import bisect_left

        def solve(values):
            tails = []
            for value in values:
                pos = bisect_left(tails, value)
                if pos == len(tails):
                    tails.append(value)
                else:
                    tails[pos] = value
            return len(tails)
        """, """
        def solve(values):
            length = [1] * len(values)
            for i in range(len(values)):
                for j in range(i):
                    if values[j] < values[i]:
                        length[i] = max(length[i], 1 + length[j])
            return max(length, default=0)
        """),
    exercise("island_count", "Count components of ones in a rectangular board using four-direction connectivity.",
             "Flood fill marks all cells reached from a new land cell before continuing the scan.",
             lambda r: ([[r.randrange(2) for _ in range(5)] for _ in range(r.randint(3, 6))],),
             ([[1, 0, 1], [1, 0, 1], [0, 1, 0]],), 3, """
        def solve(board):
            if not board or not board[0]:
                return 0
            rows, cols = len(board), len(board[0])
            todo = {(r, c) for r in range(rows) for c in range(cols) if board[r][c]}
            count = 0
            while todo:
                count += 1
                pending = [todo.pop()]
                while pending:
                    r, c = pending.pop()
                    for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if neighbor in todo:
                            todo.remove(neighbor)
                            pending.append(neighbor)
            return count
        """),
    exercise("merge_many_sorted_lists", "Merge any number of sorted lists, retaining duplicate values.",
             "Keep one value per nonempty iterator on a heap and fetch its successor after emission.",
             lambda r: ([sorted(numbers(r)) for _ in range(r.randint(2, 5))],),
             ([[], [2, 4], [1, 2]],), [1, 2, 2, 4], """
        from heapq import heappop, heappush

        def solve(lists):
            pending = []
            for row, values in enumerate(lists):
                if values:
                    heappush(pending, (values[0], row, 0))
            result = []
            while pending:
                value, row, col = heappop(pending)
                result.append(value)
                if col + 1 < len(lists[row]):
                    heappush(pending, (lists[row][col + 1], row, col + 1))
            return result
        """, """
        from heapq import merge

        def solve(lists):
            return list(merge(*lists))
        """),
)

EXERCISES = dict(basic=BASIC, intermediate=INTERMEDIATE, advanced=ADVANCED)


def examples(level, seed):
    rng = Random(seed)
    while True:
        for task in rng.sample(EXERCISES[level], len(EXERCISES[level])):
            name = rng.choice(("", "compute_", "get_")) + task.name
            solution = rng.choice(task.sources).replace("def solve(", f"def {name}(")
            calls = []
            for _ in range(rng.choice((1, 2, 2, 3))):
                args = task.args(rng)
                calls.append((args, task.solve(*args)))
            if rng.randrange(3) == 0:
                calls.append((task.edge, task.answer))
                rng.shuffle(calls)
            checks = "\n".join(
                f"assert {name}({', '.join(map(repr, args))}) == {value!r}"
                for args, value in calls
            )
            if rng.randrange(3) == 0:
                lines = (f"# {task.prompt}", f"# {task.explanation}", solution, checks)
            elif rng.randrange(2) == 0:
                lines = (f"Task: {task.prompt}", "Implementation:", f"```python\n{solution}\n{checks}\n```",
                         f"Explanation: {task.explanation}")
            else:
                lines = (f"Problem: {task.prompt}", f"Approach: {task.explanation}",
                         f"```python\n{solution}\n\n{checks}\n```")
            yield "\n\n".join(lines) + "\n\n"
