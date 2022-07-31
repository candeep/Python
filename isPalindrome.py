def isPalindrome(n,l):
    m,pal=n,0
    while n>0:
        rem=n%10
        pal=pal+rem*(10**l)
        n=n//10
        l=l-1
    if pal==m:
        return True
n=1
while n<1000:
    b=str(n)
    l=len(b)-1
    if isPalindrome(n,l):
        print(n)
    n=n+1
