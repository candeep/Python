import numpy as np
def opnot(a):
    n=len(a)
    i=0
    c=[]
    while(i<n):
        s=int(not(a[i]))
        c.append(int(not(a[i])))
        i=i+1
    return c
def operation(a,b,op):
    c=[]
    n1=len(a)
    n2=len(b)
    i,j=0,0
    if(op=='AND'):
        while(i<n1 and j<n2):
            c.append(a[i] & b[j])
            i=i+1
            j=j+1
        return c
    else:
        while(i<n1 and j<n2):
            c.append(a[i] | b[j])
            i=i+1
            j=j+1
        return c
def prec(i):
    match i:
        case 'AND' : return 2
        case 'OR' : return 1
        case 'NOT' : return 3
def queproc(query):
    k,operand,operator,res=[],[],[],[]
    k=query.split()
    top,top1=-1,-1
    for i in k:
        if i not in ['AND','OR','NOT']:
            operand.append(incidence[i])
    for i in k:
        if i in ['AND','OR','NOT']:
            if(top==-1):
                operator.append(i)
                top=top+1
            elif(i=='NOT'):
                opn=operand.pop()
                res=opnot(opn)
                operand.append(res)
            elif(prec(i) < prec(operator[top])):
                a=operand.pop()
                b=operand.pop()
                oper=operator.pop()
                res=operation(a,b,oper)
                operand.append(res)
                operator.append(i)
    if(operator[top]!=None):
        a=operand.pop()
        b=operand.pop()
        op=operator.pop()
        res=operation(a,b,op)
        operand.append(res)
    print("\n")
    print("Result of the query is : ",operand[0])
import numpy as np
import pandas as pd
n=int(input("enter no of documents : "))
docs=[]
for i in range(n):
  docs.append(input("enter the doc : "))
j=0
for i in docs:
  print("DOC",j," : ",i)
  j+=1
print("\n")
token=[]
for i in docs:
  for j in i.split():
    token.append(j)
print(token)
print("\n")
incidence={}
for i in token:
    incidence[i]=[]
    for j in docs:
        if i in j:
            incidence[i].append(1)
        else:
            incidence[i].append(0)
print("Term Document Incidence Matrix is : \n")
print("Terms","\t\t","Incidence \n")
for i,j in incidence.items():
  print(i,"\t\t",*j)
print("\n")
query=input("enter your query : ")
queproc(query)
