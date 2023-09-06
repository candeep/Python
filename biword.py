# Python
#biwordindex
r1="RVR is in guntur city".split(" ")
r2="In guntur RVR is good college".split(" ")
r3="I am happy with the college".split(" ")
l=[]
for i in range(0,len(r1)-1):
    x=r1[i]+" "+r1[i+1]
    l.append(x)
for i in range(0,len(r2)-1):
    x=r2[i]+" "+r2[i+1]
    l.append(x)
for i in range(0,len(r3)-1):
    x=r3[i]+" "+r3[i+1]
    l.append(x)
l=set(l)
l=list(l)
print(l)
