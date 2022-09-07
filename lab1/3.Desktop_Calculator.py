a=int(input('Enter a vale: '))
b=int(input('Enter b vale: '))
x=1
while(x==1):
    print('1.Addition\n2.Subtraction\n3.Multiplication')
    print('4.Division')
    ch=int(input('Enter Your Choice: '))
    if ch==1:
        print('Addition =',a+b)
    elif ch==2:
        print('Subtraction =',a-b)
    elif ch==3:
        print('Multiplication =',a*b)
    elif ch==4:
        print('Civision =',a/b)
    else:
        x=2
    
