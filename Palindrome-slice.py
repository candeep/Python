def isPalindrome(n):
    a=str(n)
    b=a[::-1]
    if b==a:
        print(a)
for n in range(1,1001):
    isPalindrome(n)
