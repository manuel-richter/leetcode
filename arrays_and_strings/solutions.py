"""Solutions to:
    - 724. Find Pivot Index
    - 747. Largest Number At Least Twice of Others
    - 66. Plus One
    - 3324. String Sequence
    - 3325. Number of Substrings with K distinct characters
    - 929. Unique Email Addresses
    - 482. License Key Formatting
    - 1048. Longest String Chain
    - 859. Buddy Strings
"""
from typing import List
from collections import Counter

class Solution:
    def pivot_index(self, nums: List[int]) -> int:
        """724. Find Pivot Index - Find and return pivot index of array where left sum == right sum.

        Args:
            nums (List[int]): array of numbers.

        Returns:
            int: Pivot index.
        """
        complete_sum = sum(nums)
        left_sum = 0
        for idx, val in enumerate(nums):
            if left_sum == complete_sum - left_sum - val:
                return idx
            left_sum += val
        return -1

    def dominant_index(self, nums: List[int]) -> int:
        """747. Largest Number At Least Twice of Others.

        Args:
            nums (List[int]): array of numbers.

        Returns:
            int: Index of largest number with at least double its value.
        """
        largest = max(nums)
        return (
            nums.index(largest)
            if all(largest == num or num * 2 <= largest for num in nums)
            else -1
        )

    def plus_one(self, digits: List[int]) -> List[int]:
        """66. Plus One.

        Args:
            digits (List[int]): _description_

        Returns:
            List[int]: _description_
        """

    def string_sequence(self, target: str) -> List[str]:
        """3324. String Sequence.

        Args:
            target (str): target string.

        Returns:
            List[str]: list of strings.
        """
        result = []
        substr = ""

        for _, char in enumerate(target):
            s = "a"
            result.append(substr + s)

            while s != char:
                s = chr(ord(s) + 1)
                result.append(substr + s)
            substr += s
        return result

    def number_of_substrings(self, s: str, k: int) -> int:
        """3325. Number of Substrings with K distinct characters.

        Args:
            s (str): string.
            k (int): number of distinct characters.

        Returns:
            int: number of substrings with k distinct characters.
        """
        string_len = len(s)
        counts = Counter()
        
        amount_substrings = left = 0

        for cur_idx, char in enumerate(s):
            counts[char] += 1

            while counts[char] == k:
                amount_substrings += string_len - cur_idx
                counts[s[left]] -= 1
                left += 1
        
        return amount_substrings

    def num_unique_emails(self, emails: List[str]) -> int:
        """929. Unique Email Addresses.

        Args:
            emails (List[str]): list of emails.

        Returns:
            int: number of unique emails.
        """
        unique_emails = set()

        for email in emails:
            local, domain = email.split("@")
            local = local.split("+")[0].replace(".", "")
            unique_emails.add(local + "@" + domain)
        return len(unique_emails)

    def license_key_formatting(self, s: str, k: int) -> str:
        """482. License Key Formatting.

        Args:
            s (str): string.
            k (int): number of characters per group.

        Returns:
            str: formatted string.
        """
        s = s.replace("-", "").upper()[::-1]
        return '-'.join(s[i:i+k] for i in range(0, len(s), k))[::-1]

    def longest_string_chain(self, words: List[str]) -> int:
        """1048. Longest String Chain.

        Args:
            words (List[str]): list of words.

        Returns:
            int: length of longest string chain.
        """
        words.sort(key=len)
        count_by_word = {}

        for word in words:
            count_by_word[word] = max(count_by_word.get(word[:i] + word[i+1:], 0) + 1 for i in range(len(word)))
        return max(count_by_word.values())
    
        """
        words.sort()
        longest = 1
        cnt = {word:1 for word in words}
        
        for word in words:
            for i in range(len(word)):
                predecessor = word[:i] + word[i+1:]
                if predecessor in cnt:
                    cnt[word] = cnt[predecessor] + 1
                    longest = max(longest, cnt[word])
        return longest
        """

    def buddy_strings(self, s: str, goal: str) -> bool:
        """859. Buddy Strings.

        Args:
            s (str): string.
            goal (str): goal string.

        Returns:
            bool: True if s and goal are buddy strings.
        """
        if len(s) != len(goal):
            return False
        if s == goal:
            return len(set(s)) < len(s)
        diff = [(a, b) for a, b in zip(s, goal) if a != b]
        return len(diff) == 2 and diff[0] == diff[1][::-1]

    def move_zeros(self, nums: List[int]) -> None:
        """283. Move Zeros.

        Args:
            nums (List[int]): list of numbers.
        """
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1

solution = Solution()

print(solution.pivot_index([1, 7, 3, 6, 5, 6]))
print(solution.string_sequence("abc"))
print(solution.number_of_substrings("abacb", 2))
print(solution.num_unique_emails(["test.email+alex@leetcode.com", "test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com"]))
print(solution.license_key_formatting("2-5g-3-J", 2))
print(solution.longest_string_chain(["a", "b", "ba", "bca", "bda", "bdca"]))

