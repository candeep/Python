def Factorial(rem):
    fact=1
    while rem>0:
        fact=fact*rem
        rem=rem-1
    return fact
def isStrong(n):
    m,sum=n,0
    while n>0:
        rem=n%10
        sum=sum+Factorial(rem)
        n=n//10
    if sum==m:
        return True
    else:
        return False
n=int(input('Enter Number : '))
print(isStrong(n))
