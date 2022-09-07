def leap(year):
    if year%400==0 or (year%100!=0 and year%4==0):
        return True
    else:
        return False
year=int(input('Enter Year'))
mm=input('Enter Month Name (in 3 letters)')
if (mm=='jan'or mm=='mar'or mm=='may'or mm=='jul'or mm=='aug'or mm=='oct'or mm=='del'):
   print('31 DAYS')
elif(mm=='feb'):
    if leap(year):
        print('29 DAYS')
    else:
        print('28 DAYS')
else:
    print('30 DAYS')

    
