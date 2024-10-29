""" WEEK 4 MORE DATA STRUCTURES
/ Add and Search Word
/ Implement Trie (Prefix Tree)
/ Subtree of Another Tree
/ Kth Smallest Element in a BST
/ Lowest Common Ancestor of BST
/ Merge K Sorted Lists
/ Find Median from Data Stream
/ Insert Interval
/ Longest Consecutive Sequence
Word Search II

Meeting Rooms
Meeting Rooms II
Alien Dictionary
Graph Valid Tree
Number of Connected Components in an Undirected Graph
"""

from cgitb import lookup
from email.policy import default
from typing import List, Any, Optional
import collections
import heapq
import string
from helper import ListNode, TreeNode, Node
from queue import PriorityQueue

class Solution:
    def __init__(self):
        pass
    # 211
    class WordNode:
        def __init__(self):
            self.children = collections.defaultdict(Solution.WordNode)
            self.is_word = False
        
    class WordDictionary:
        def __init__(self):
            self.root = Solution.WordNode()        
        
        def addWord(self, word: str) -> None:
            if not word: return None
            tmp = self.root
            for c in word:
                if c not in tmp.children:
                    tmp.children[c] = Solution.WordNode()
                tmp = tmp.children[c]
            tmp.is_word = True
            return None
            
        def search(self, word: str) -> bool:
            if not word: return False
            return self.dfs_search(word, self.root)

        def dfs_search(self, word, node):
            for idx, c in enumerate(word):
                if c not in node.children:
                    if c == ".":
                        for n in node.children:
                            if node.children[n].is_word != "True" and self.dfs_search(word[idx+1:], node.children[n]):
                                return True
                    return False
                node = node.children[c]            
            return node.is_word

    # 572 O(M*N) M=no of nodes in root N=no of nodes in subRoot Space: O(H)
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def check(r,s):
            if not r or not s:
                return r == s
            return r.val == s.val and check(r.left, s.left) and check(r.right, s.right)
        
        def dfs(r,s):
            if not r: return False
            return check(r,s) or dfs(r.left, s) or dfs(r.right, s)
        
        return dfs(root, subRoot)

    # 230
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def flatten(node):
            return flatten(node.left) + [node.val] + flatten(node.right) if node else []
        return flatten(root)[k-1]

        # stack = []
        # while root or stack:
        #     while root:
        #         stack.append(root)
        #         root = root.left
        #     root = stack.pop()
        #     k -= 1
        #     if not k:
        #         return root.val
        #     root = root.right 

    # 146
    class LRUCache(collections.OrderedDict):

        def __init__(self, capacity: int):
            self.capacity = capacity
            self.cache = collections.OrderedDict()

        def get(self, key: int) -> int:
            if key not in self.cache:
                return -1
            
            self.cache.move_to_end(key)
            return self.cache[key]

            
        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value

            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    # 235
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if p.val < root.val > q.val:
                root = root.left
            elif p.val > root.val < q.val:
                root = root.right
            else:
                return root
    
    # 23
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        curr = head = ListNode(0)
        queue = []
        count = 0
        for l in lists:
            if l:
                count += 1
                heapq.heappush(queue, (l.val, count, l))
        while queue:
            _, _, curr.next = heapq.heappop(queue)
            curr = curr.next
            if curr.next:
                count += 1
                heapq.heappush(queue, (curr.next.val, count, curr.next))
        return head.next    

    # 295
    class MedianFinder:

        def __init__(self):
            self.heaps = [], []
            
        def addNum(self, num: int) -> None:
            small, large = self.heaps
            heapq.heappush(small, -heapq.heappushpop(large, -num))

            if len(large) < len(small):
                heapq.heappush(large, -heapq.heappop(small))

        def findMedian(self) -> float:
            small, large = self.heaps
            if len(large) > len(small):
                return -float(large[0])
            return (-large[0] + small[0]) / 2.0

    # 57
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        s, e = newInterval[0], newInterval[1]
        left, right = [], []
        for i in intervals:
            if i[1] < s:
                left += i,
            elif i[0] > e:
                right += i,
            else:
                s = min(s, i[0])
                e = max(e, i[1])
        return left + [[s, e]] + right
    
    # 128
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        best = 0
        for x in nums:
            if x - 1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                best = max(best, y - x)
        return best
    
    # 79
    def exist(self, board: List[List[str]], word: str) -> bool:
        height, width = len(board), len(board[0])
        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        def backtrack(row, col, suffix):
            if len(suffix) == 0:
                return True
            if row < 0 or row >= height or col < 0 or col >= width or suffix[0] != word[row][col]:
                return False
            board[row][col] = "#"
            for dr, dy in directions:
                if backtrack(row+dr, col+dy, suffix[1:]):
                    return True
            board[row][col] = suffix[0]
            return False
            
        for h in range(height):
            for w in range(width):
                if backtrack(h,w,word):
                    return True
        
        return False

    # 252
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort(key=lambda x: x[0])
        return all(intervals[x-1][1] <= intervals[x][0] for x in range(1, len(intervals)))
    
    # 253
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[0])
        rooms = []
        heapq.heappush(rooms, intervals[0][1])

        for i in intervals[1:]:
            if rooms[0] <= i[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, i[1])
        
        return len(rooms)

        
    # 261
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # if len(edges) != n-1: return False
        
        # graph = {x:[] for x in range(n)}

        # for start,end in edges:
        #     graph[start].append(end)
        #     graph[end].append(start)
        
        # queue = collections.deque([0])
        # visited = set()
        # parent = [-1] * n

        # while queue:
        #     cur = queue.popleft()
        #     visited.add(cur)
        #     for adj in graph[cur]:
        #         if adj not in visited:
        #             visited.add(adj)
        #             parent[adj] = cur
        #             queue.append(adj)
        #         elif adj != parent[cur]:
        #             return False
                
        
        # return len(visited) == n

        if len(edges) != n-1: return False

        parents = list(range(n))

        def find(x):
            if x != parents[x]:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(xy):
            px, py = map(find, xy)
            parents[px] = py
            return px != py
        
        for edge in edges:
            union(edge)
        
        return all(map(union, edges))


    # 323
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # cnt = 0
        # parent = list(range(n))

        # def find(x):
        #     if parent[x] != x:
        #         parent[x] = find(parent[x])
        #     return parent[x]
        
        # for edge in edges:
        #     x,y = map(find, edge)
        #     if x != y:
        #         parent[x] = y
        #         cnt += 1
        
        # return cnt

        graph = collections.defaultdict(list)
        visited = set()
        cnt = 0

        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for adj in graph[node]:
                    dfs(adj)

        for node in range(n):
            if node not in visited:
                dfs(node)
                cnt += 1

        return cnt