def merge(a,b):
    c=[]
    i=0
    j=0
    while i<len(a) and j<len(b):
        if a[i]<b[j]:
            c.append(a[i])
            i=i+1
        elif b[j]<a[i]:
            c.append(b[j])
            j=j+1
        else:
            c.append(a[i])
            i=i+1
            j=j+1
    while i<len(a):
        c.append(a[i])
        i=i+1
    while j<len(b):
        c.append(b[j])
        j=j+1
    return c

a=[3,5,7,9]
b=[1,2,8,20,22,67]
c=merge(a,b)
print('After Merging :',c)
