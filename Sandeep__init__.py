class student:
    def __init__(self,roll,m,age):
        self.roll=roll
        self.marks=m
        self.age=age

a,b,c=input('Enter Roll Number'),input('Enter Marks'),input('Enter Age')
sandeep=student(a,b,c)
print('Roll Number',sandeep.roll)
print('Marks',sandeep.marks)
print('age',sandeep.age)
