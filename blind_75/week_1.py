""" WEEK 1 SEQUENCES
/ Two Sum
/ Contains Duplicate
/ Best Time to Buy and Sell Stock
/ Valid Anagram
/ Valid Parentheses
/ Product of Array Except Self
/ Maximum Subarray
/ 3Sum
/ Merge Intervals
/ Group Anagrams
/ Maximum Product Subarray
/ Search in Rotated Sorted Array
"""

from typing import List
import collections
import string



class Solution:
    # 1
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for idx, num in enumerate(nums):
            match = target - num
            if match in seen:
                return [seen[match], idx]
            seen[num] = idx
        return [-1]
    
    # 217
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) > len(set(nums))

    # 121 
    def maxProfit(self, prices: List[int]) -> int:
        max_profit, min_buy = 0, prices[0]
        for buy in range(1, len(prices)):
            min_buy = min(min_buy, prices[buy])
            if min_buy < prices[buy]:
                max_profit = max(max_profit, prices[buy] - min_buy)
        return max_profit

    # 242
    def isAnagram(self, s: str, t: str) -> bool:
        # return collections.Counter(s) == collections.Counter(t)
        return all([s.count(x) == t.count(x) for x in string.ascii_lowercase])
        # cnts_s = collections.Counter(s)
        # for char in t:
        #     if char not in cnts_s:
        #         return False
        #     cnts_s[char] -= 1
        #     if cnts_s[char] == 0:
        #         del cnts_s[char]
        # return len(cnts_s) == 0

    # 20
    def isValid(self, s: str) -> bool:
        bracket_map = {"(": ")", "[": "]",  "{": "}"}
        open_par = set(["(", "[", "{"])
        stack = []
        for i in s:
            if i in open_par:
                stack.append(i)
            elif stack and i == bracket_map[stack[-1]]:
                    stack.pop()
            else:
                return False
        return stack == []

    # 238
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # left, right, result = [0] * len(nums), [0] * len(nums), [0] * len(nums)
        # left[0], right[-1] = 1, 1

        # for idx in range(1, len(nums)):
        #     left[idx] = nums[idx-1] * left[idx-1]

        # for idx in reversed(range(len(nums)-1)):
        #     right[idx] = nums[idx+1] * right[idx+1]
        
        # for i in range(len(nums)):
        #     result[i] = left[i] * right[i]
        
        # return result

        result = []
        tmp = 1 # reset first entry from left
        for idx in range(len(nums)):
            result.append(tmp)
            tmp *= nums[idx]
        tmp = 1 # reset last entry from right
        for idx in reversed(range(len(nums))):
            result[idx] = result[idx] * tmp
            tmp *= nums[idx]
        return result

    # 53
    def maxSubArray(self, nums: List[int]) -> int:
        cur_sum, max_sum = nums[0], nums[0]
        for idx in range(1, len(nums)):
            cur_sum = max(nums[idx], cur_sum + nums[idx])
            max_sum = max(max_sum, cur_sum)
        return max_sum

    # 167 twoSum sorted
    def twoSumSorted(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            if target - numbers[left] == numbers[right]:
                return [left + 1, right + 1]
            if target - numbers[left] < numbers[right]:
                right -= 1
            else:
                left += 1
        return []
    
    # 15
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res, dups = set(), set()
        seen = {}

        for i, val1 in enumerate(nums):
            if val1 not in dups:
                dups.add(val1)
                for j, val2 in enumerate(nums[i+1:]):
                    complement = -val1 - val2
                    if complement in seen and seen[complement] == i:
                        res.add(tuple([val1, val2, complement]))
                    seen[val2] = i
        return list(res)

    # 56
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        results = []
        intervals.sort(key=lambda x: x[0])

        for interval in intervals:
            if  not results or results[-1][1] < interval[0]:
                results.append(interval)
            else:
                results[-1][1] = max(results[-1][1], interval[1]) 
        
        return results

    # 49
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = collections.defaultdict(set)
        for word in strs:
            groups[tuple(sorted(word))].add(word)
        
        return sorted([sorted(x) for x in groups.values()], key=len)

    # 152
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 0
        max_tmp, min_tmp, result = 1, 1, nums[0]

        for num in nums:
            vals = (num, num * max_tmp, num * min_tmp)
            max_tmp, min_tmp = max(vals), min(vals)
            result = max(result, max_tmp)
        
        return result

    # 33
    def search(self, nums: List[int], target: int) -> int:   
        left, right = 0, len(nums)-1
        
        while left < right:
            mid = left + (right - left) // 2

            if target < nums[0] < nums[mid]: # -inf
                left = mid + 1
            elif target >= nums[0] > nums[mid]: # +inf
                right = mid
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid
            else:
                return mid
        return -1



solution = Solution()
print(solution.search(nums=[4,5,6,7,0,1,2], target=3)) # 4
print(solution.maxProduct([2,3,-2,4])) # 6
print(solution.groupAnagrams(["eat","tea","tan","ate","nat","bat"])) # [["bat"],["nat","tan"],["ate","eat","tea"]]
print(solution.merge([[1,3],[2,6],[8,10],[15,18]]))
print(solution.threeSum([-1,0,1,2,-1,-4])) # # [[-1,-1,2],[-1,0,1]]
print(solution.twoSumSorted([2,7,11,15], target = 9)) # [1,2]
print(solution.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])) # 6
print(solution.productExceptSelf([1,2,3,4]))
print(solution.isValid("/(/)/[/]/{/}"))
print(solution.isAnagram(s = "anagram", t = "nagaram"))
print(solution.maxProfit([7,1,5,3,6,4]))
print(solution.containsDuplicate([1,2,3,1]))
print(solution.twoSum(nums=[2,7,11,15], target=9)) # [0,1]
