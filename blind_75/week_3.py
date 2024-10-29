""" WEEK 3 NON LINEAR DATA STRUCTURES
/ Validate Binary Search Tree
/ Invert/Flip Binary Tree
TODO Non-overlapping Intervals
/ Serialize and Deserialize Binary Tree
/ Construct Binary Tree from Preorder and Inorder Traversal
/ Top K Frequent Elements
/ Clone Graph
/ Course Schedule
/ Binary Tree Maximum Path Sum
/ Maximum Depth of Binary Tree
/ Same Tree
/ Binary Tree Level Order Traversal
/ Encode and Decode Strings (LeetCode Premium)
"""

from typing import List, Any, Optional
import collections
import heapq
import string
from helper import ListNode, TreeNode, Node

class Solution:
    def __init__(self):
        pass

    # 98
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def validate(node, floor=float("-inf"), ceiling=float("inf")):
            if not node: return True
            if node.val <= floor or node.val >= ceiling:
                return False
            return validate(node.left, floor, node.val) and validate(node.right, node.val, ceiling)

        return validate(root)

    # 226
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    # 435 TODO
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        return 1
    
    # 297
    class Codec:
        queue = collections.deque()

        def serialize(self, root):
            if not root: return "x"
            return ",".join([root.val + self.serialize(root.left) + self.serialize(root.right)])
        
        def des_helper(self):
            if self.queue[0] == "x":
                self.queue.popleft()
                return None
            node = TreeNode(self.queue.popleft())
            node.left = self.des_helper()
            node.right = self.des_helper()


        def deserialize(self, data):
            def des_helper(q):
                if q[0] == "x":
                    q.popleft()
                    return None
                node = TreeNode(q.popleft())
                node.left, node.right = des_helper(q), des_helper(q)
                return node
                
            queue = collections.deque(data.split(","))
            root = des_helper(queue)
            return root

    # 105
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
        root_idx = 0

        def helper(left_idx, right_idx):
            nonlocal root_idx
            if left_idx > right_idx: return None

            root_val = preorder[root_idx]
            node = TreeNode(root_val)
            root_idx += 1

            node.left = helper(left_idx, val_to_idx[root_val] - 1)
            node.right = helper(val_to_idx[root_val] + 1, right_idx)
            return node

        val_to_idx = {val:idx for idx,val in enumerate(inorder)}
        return helper(0, len(inorder) - 1)

    # 347
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = collections.Counter(nums)
        return [k for k,v in counts.most_common(k)]
        
        # counts = collections.Counter(nums)
        # return heapq.nlargest(k, counts.keys(), key=counts.get)
        
        # buckets = [[] for _ in range(len(nums)+1)]
        # counts = collections.Counter(nums)

        # for num, freq in counts.items():
        #     buckets[freq].append(num)
        
        # flatten = [num for bucket in buckets for num in bucket] # itertools.chain(*buckets)
        # return flatten[::-1][:k]

    # 133
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return Node(0)

        queue = collections.deque([node])
        clones = {node:Node(node.val)}
        
        while queue:
            cur = queue.popleft()
            for nei in cur.neighbors:
                if nei not in clones:
                    queue.append(nei)
                    clones[nei] = Node(nei.val)
                clones[cur].neighbors.append(clones[nei])

        return clones[node]

    # 207
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        topo_sort = []
        graph = collections.defaultdict(set)
        incoming = {n:0 for n in range(numCourses)}
        
        for end, start in prerequisites:
            if end not in graph[start]:
                graph[start].add(end)
                incoming[end] += 1
        
        queue = collections.deque([node for node in range(numCourses) if incoming[node] == 0])
        
        while queue:
            cur = queue.popleft()
            if cur not in topo_sort:
                topo_sort.append(cur)
            for adj in graph[cur]:
                incoming[adj] -= 1
                if incoming[adj] == 0:
                    queue.append(adj)
        
        return not sum(incoming.values()) # if still has edges then there is a cycle

    # 124
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        total_sum = 0

        def dfs(node):
            nonlocal total_sum
            if not node: return 0
            left = dfs(node.left)
            right = dfs(node.right)
            price_newpath = left + node.val + right
            total_sum = max(total_sum, price_newpath)
            return node.val + max(left, right)

        dfs(root)
        return total_sum

    # 104
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0

        # def dfs(node, depth):
        #     if not node: return depth
        #     return max(dfs(node.left, depth + 1), dfs(node.right, depth + 1))
        
        # return dfs(root, 0)

    # 100
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:         
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # 102
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res = []
        queue = collections.deque([root])

        while queue:
            level = []
            for _ in range(len(queue)):
                cur = queue.popleft()
                level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(level)

        return res

    # 217
    class Codec:    
        def encode(self, strs: list[str]) -> str:
            """Encodes a list of strings to a single string.
            """        
            return "".join('%d:' % len(s) + s for s in strs)
            

        def decode(self, s: str) -> list[str]:
            """Decodes a single string to a list of strings.
            """
            strs = []
            i = 0
            while i < len(s):
                j = s.find(":", i) # return first index of substring
                i = j + 1 + int(s[i:j]) 
                strs.append(s[j+1:i])
            return strs


solution = Solution()
print(solution.canFinish(numCourses = 2, prerequisites = [[1,0],[0,1]])) # False
print(solution.topKFrequent(nums = [1,1,1,2,2,3], k = 2)) # [1,2]
print(solution.eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]])) # 1
