def factorial(rem):
    fact=1
    for i in range(1,rem+1):
        fact=fact*i
    return fact
def isStrong(n):
    m,sum=n,0
    while n>0:
        rem=n%10
        sum=sum+factorial(rem)
        n=n//10
    if sum==m:
        return True
    else:
        return False
n=int(input('Enter A Number : '))
print(isStrong(n))
