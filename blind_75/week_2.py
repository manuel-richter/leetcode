""" WEEK 2 DATA STRUCTURES
/ Reverse a Linked List
/ Detect Cycle in a Linked List
/ Container With Most Water
/ Find Minimum in Rotated Sorted Array
/ Longest Repeating Character Replacement
/ Longest Substring Without Repeating Characters
/ Minimum Window Substring
/ Number of Islands
/ Remove Nth Node From End Of List
TODO Palindromic Substrings
/ Pacific Atlantic Water Flow
"""
from typing import List, Optional
import collections
from helper import ListNode


class Solution:
    def __init__(self):
        pass
    # 206
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # prev = None
        # while head:
        #     tmp = head.next
        #     head.next = prev
        #     prev = head
        #     head = tmp
        # return prev
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev

    # 141
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # head = [3,2,0,-4], pos = 1

        # hashset
        # seen = set()
        # while head:
        #     if head in seen:
        #         return True
        #     seen.add(head)
        #     head = head.next
        # return False
        if not head: return False
        slow, fast = head, head.next

        while fast != slow:
            if fast is None or slow is None: return False
            slow = slow.next
            fast = fast.next.next
        return True

    # 11
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_area = 0

        while left < right:
            max_area = max(max_area, (right - left) * min(height[left], height[right]))

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

    # 153
    def findMin(self, nums: List[int]):
        # if nums[0] < nums[-1]: return nums[0]
        # left, right = 0 , len(nums) - 1 
        # while left < right:
        #     mid = left + (right - left) // 2
        #     if nums[right] < nums[mid]:
        #         left = mid + 1
        #     else:
        #         right = mid
        # return nums[left]

        left, right = 0, len(nums)-1
        if len(nums) == 1 or nums[left] < nums[right]:
            return nums[left]

        while left < right:
            mid = left + (right - left) // 2
            # pivot point left > right
            if nums[mid] > nums[mid+1]:
                return nums[mid+1]
            # pivot let > right
            if nums[mid-1] > nums[mid]:
                return nums[mid]
            # somewhere right
            if nums[mid] > nums[0]:
                left = mid + 1
            else:
                # somewhere left
                right = mid - 1

    # 424
    def characterReplacement(self, s: str, k: int) -> int:
        # s = "ABAB", k = 2
        cnts = collections.Counter()
        max_char, win_start, res = 0, 0, 0

        for win_end in range(len(s)):
            cnts[s[win_end]] += 1
            max_char = max(max_char, cnts[s[win_end]])

            if win_end - win_start + 1 - max_char > k:
                cnts[s[win_start]] -= 1
                win_start += 1
            res = max(res, win_end - win_start + 1)
        return res

    # 3
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = {}
        max_length, window_start = 0, 0

        for window_end, character in enumerate(s):    
            if character in seen and window_start <= seen[character]:
                window_start = seen[character] + 1
            else:
                max_length = max(max_length, window_end - window_start + 1)
            seen[character] = window_end

        return max_length

    # 76
    def minWindow(self, s: str, t: str) -> str:
        target_cnts = collections.Counter(t)
        missing = len(t)
        start_win, res = 0, ""

        for end_win, char in enumerate(s):
            if target_cnts[char] > 0:
                missing -= 1

            target_cnts[char] -= 1

            while missing == 0:
                if not res or (end_win - start_win + 1) < len(res):
                    res = s[start_win:end_win+1]
                
                # moving window raise target cnt
                target_cnts[s[start_win]] += 1
				# Increase missing if cnts > 0 and break loop
                if target_cnts[s[start_win]] > 0:
                    missing += 1
                start_win += 1
        return res

    # 200
    def numIslands(self, grid: List[List[str]]) -> int:
        islands, width, height = 0, len(grid[0]), len(grid)

        def dfs(h, w):
            if h < 0 or h >= height or w < 0 or w >= width or grid[h][w] == "0":
                return
            grid[h][w] = "0" # mark visited
            dfs(h,w-1)
            dfs(h,w+1)
            dfs(h-1,w)
            dfs(h+1,w)

        for h in range(height):
            for w in range(width):
                if grid[h][w] == "1":
                    islands += 1
                    dfs(h,w)
        return islands

    # 19
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # Input: head = [1,2,3,4,5], n = 2
        # Output: [1,2,3,5]
        slow, fast = head, head

        for _ in range(n):
            fast = fast.next
        
        if not fast:
            return head.next

        while fast.next:
            fast, slow = fast.next, slow.next
        
        slow.next = slow.next.next
        return head

    # 647 TODO
    def countSubstrings(self, s: str) -> int:
        # "abc" # 3 "a" "b"  "c"
        # "aaa" # 6 "a", "a", "a", "aa", "aa", "aaa"
        return 1

    # 417
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        height, width = len(heights), len(heights[0])
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        pacific_cells, atlantic_cells = set(), set()

        def dfs(x,y, reachable):
            reachable.add((x,y))
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy                
                if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height or (new_x, new_y) in reachable:
                    continue 
                if heights[new_x][new_y] >= heights[x][y]:
                    dfs(new_x, new_y, reachable)

        for i in range(height):
            dfs(i, 0, pacific_cells)
            dfs(i, width-1, atlantic_cells)
        
        for i in range(width):
            dfs(0, i, pacific_cells)
            dfs(height-1, i, atlantic_cells)

        return list(atlantic_cells.intersection(pacific_cells))


    # 5
    def longestPalindrome(self, s: str) -> str:
        result = ""
        def get_substring(left, right):
            while 0 <= left and right < len(s) and s[left] == s[right]:
                left -= 1
                right +=1
            return s[left+1:right]

        for idx in range(len(s)):
            # idx, idx    (odd case, like "aba")
            # idx, idx+ 1 (even case, like "abba")
            result = max(result, get_substring(idx, idx), get_substring(idx, idx+1), key=len)

        return result


solution = Solution()
print(solution.pacificAtlantic(heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]))
print(solution.countSubstrings("abc"))
print(solution.numIslands(grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]))
print(solution.minWindow(s="ADOBECODEBANC", t="ABC")) # "BANC"
print(solution.characterReplacement("ABAB", 2))
print(solution.findMin([4,5,6,7,0,1,2])) # 1
ln = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
print(solution.reverseList(ln))
print(solution.maxArea(height=[1,8,6,2,5,4,8,3,7])) # 49
print(solution.longestPalindrome("abba")) # "bab"
print(solution.lengthOfLongestSubstring("abcabcbb")) # 3