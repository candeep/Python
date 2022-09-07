from cmath import *
a=int(input('Enter a Value :'))
b=int(input('Enter b Value :'))
c=int(input('Enter c Value :'))
d=b*b-4*a*c
x1=(-b+sqrt(d))/(2*a)
x2=(-b-sqrt(d))/(2*a)
if d==0:
    print('Roots Are Equal')
    print('root1 =',x1,'root2 =',x2)
elif d>0:
    print('Roots Are real and distinict')
    print('root1 =',x1,'root2 =',x2)
else:
    print('Roots Are Imaginary')
    print('root1 =',x1,'root2 =',x2)
