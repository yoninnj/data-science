ak=int(input())
lik=[]
for i in range(ak):
    n=int(input())
    l=[]
    for k in range(n):
        akl = list(map(int,input().strip().split()))[:5]
        l.append(akl)
    lik.append(l)
def superio(jl,kil):
    a=0
    if(jl[0]<kil[0]):
        a+=1
    if(jl[1]<kil[1]):
        a+=1
    if(jl[2]<kil[2]):
        a+=1
    if(jl[3]<kil[3]):
        a+=1
    if(jl[4]<kil[4]):
        a+=1
    if(a>=3):
        return True
    else:
        return False
def rank(il):
    max=il[0]
    for i in il:
        if(superio(max,i)):
            continue
        else:
            max=i
            c=il.index(i)+1
    ji=1
    for k in il:
        if(superio(max,k)):
            ji+=1
    if(ji==len(il)):
        print(c)
    else:
        print('-1')

for i in lik:
    rank(i)

    
            
        
        
    
    

         
    

            
            
        