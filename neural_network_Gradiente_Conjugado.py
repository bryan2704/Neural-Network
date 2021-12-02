import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def Load():
    X = data_2.values
    m = X.shape[0]
    n = X.shape[1]
    nbr_classes = 10
    X = np.append(np.ones([m,1]),X,axis=1)
    y = data_3.replace(10,0).values
    aux = []
    for i in range(m):
        aux.append(np.array([1 if y[i] == j else 0 for j in range(nbr_classes)]))
    Y = np.array(aux).reshape(-1, nbr_classes)
    return X, Y, m, n, nbr_classes

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    R=1/(1+np.exp(-z))
    return R*(1-R)

def Compute_Cost(vector,X,Y,m,n,b,nbr_classes,k,a,dim):
    thetas=Matrix(vector, b, a, m, n, dim, nbr_classes)
    A=[]
    aux=X
    m=X.shape[0]
    for i in range(0,b-2):
        a=sigmoid(aux @ thetas[i].T)
        a=np.hstack((np.ones((m,1)), a))
        A.append(a)
        aux=a
    A.append(sigmoid(A[-1] @ thetas[-1].T))  
    J=0
    for h in range(nbr_classes):
        J = J + np.sum(-Y[:,h]*np.log(A[-1][:,h])-(1-Y[:,h])*np.log(1-A[-1][:,h]))
    J=(1/m)*J
    for g in thetas:
        J=J+(k/(2*m))*(np.sum(g[:,1:]**2))
    grads=[]
    erro=[]
    F=A[-1]-Y
    grad=F.T @ A[-2]
    grads.append(grad)
    erro.append(F)
    error=F
    for i in range(len(thetas)-2,0,-1):
        error=np.multiply((error @ thetas[i+1][:,1:]),(sigmoidGradient(A[i-1] @ thetas[i].T)))
        grad=error.T @ A[i-1]
        erro.append(error)
        grads.append(grad)
    error=np.multiply((error @ thetas[1][:,1:]),(sigmoidGradient(X @ thetas[0].T)))
    grad= error.T @ X
    erro.append(error)
    grads.append(grad)
    grads=list(reversed(grads))
    erro=list(reversed(erro))
    for r in range(0,len(grads)):
        grads[r]=(1/m)*grads[r]
        grads[r]=grads[r]+(k/m)*np.hstack((np.zeros((thetas[r].shape[0],1)),thetas[r][:,1:]))    
    return J,grads,A,erro
              
def Random_Weight(a,b):
    i=(6**(1/2))/((a+b)**(1/2))
    return np.random.rand(b,a+1)*(2*i)-i

def Theta_Initialization(b,a,X,Y):
    f=[]   
    f.append(Random_Weight(X.shape[1]-1,a[0]))
    for i in range(1,b-2):
        f.append(Random_Weight(a[i-1],a[i]))
    f.append(Random_Weight(a[-1],Y.shape[1]))
    return f

def Prediction(Y,X,b,m,thetas):
    A=[]
    aux=X
    m=X.shape[0]
    for i in range(0,b-2):
        a=sigmoid(aux @ thetas[i].T)
        a=np.hstack((np.ones((m,1)), a))
        A.append(a)
        aux=a
    A.append(sigmoid(A[-1] @ thetas[-1].T)) 
    j=np.argmax(A[-1],axis=1)
    np.array(j).reshape((m, 1))
    r=np.argmax(Y,axis=1)
    u=0
    for k in range(m):
       if j[k]==r[k]:
            u=u+1
    return j,r,u

def missFeedback(X, Y, pred):
    m = len(Y)
    y_labels = []
    for row in Y:
        y_labels.append(np.argmax(row))
    y_labels = np.array(y_labels).reshape((m, 1))
    for i in range(m):
        if not (y_labels[i] == pred[i]):
            print(f'Label: \t {y_labels[i]}')
            print(f'Predição: {pred[i]}')
            pixels = X[i, 1:].reshape((20, 20))
            plt.imshow(pixels, cmap='hot')
            plt.show()
            print('----------------------------------') 

def AssistWeight(e,s):
    W=np.zeros((s,e))
    W=np.random.rand(W.shape[1],W.shape[0]+1)
    return W

def Vectorize(thetas):
    dim=[]
    for h in thetas:
        dim.append(h.shape[0]*h.shape[1])
    vector=thetas[0].flatten()
    for r in thetas[1:]:
        vector=np.hstack([vector,r.flatten()])
    return vector, dim

def Matrix(vector,b,a,m,n,dim,nbr_classes):
    matrix=[]
    matrix.append(vector[:dim[0]].reshape(a[0],n+1))
    aux=cp.deepcopy(dim)
    for i in range(1,b-2):
        aux=cp.deepcopy(dim)
        matrix.append(vector[aux[i-1]:(aux[i-1]+aux[i])].reshape(a[i],a[i-1]+1))
        aux[i]=aux[i-1]+aux[i]
    g=np.sum(dim[:-1])
    matrix.append(vector[g:].reshape(nbr_classes,a[-1]+1))
    return matrix
def assist(vector,X,Y,m,n,b,nbr_classes,k, a, dim):
    return Compute_Cost(vector, X, Y, m, n, b, nbr_classes, k, a, dim)[0]

data_2=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\imageMNIST.csv')
data_3=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\labelMNIST.csv')
X,Y,m,n,nbr_classes=Load()
a=1
X_Assist=np.zeros((10,X.shape[1]))
Y_Assist=np.zeros((10,Y.shape[1]))
for t in range(0,9):
    Y_Assist[t,:]=Y[a,:]
    X_Assist[t,:]=X[a,:]
    a=a+500
X_Assist[9,:]=X[4800,:]
Y_Assist[9,:]=Y[4800,:]
X=X_Assist
Y=Y_Assist
m=X.shape[0]
n=X.shape[1]-1    
b=int(input("Qual seria o número de camadas? "))
a = list(map(int,input("\nQual seria o número de neurônios em cada camada escondida? ").strip().split()))[:b]
thetas=Theta_Initialization(b,a,X,Y)
vector,dim=Vectorize(thetas)
verifica=cp.deepcopy(vector)
matrix=Matrix(vector,b,a,m,n,dim,nbr_classes)
alpha=int(input("\nQual seria o número de iterações? "))
lambd=float(input("\nQual seria o valor de lambda? "))
t=minimize(assist,vector,args=(X,Y,m,n,b,nbr_classes,lambd,a,dim), method='CG', jac=None, tol=None, callback=None, options={'maxiter':alpha,'disp':True , 'gtol':1e-4})
g=int(input("\nGostaria de ver os resultados gerados pela função minimize? 1 para sim, 2 para não "))
if g==1:
    print("\n")
    print(t)
thetas=Matrix(t.x,b,a,m,n,dim,nbr_classes)
pred,real,total=Prediction(Y,X,b,m,thetas)
print("\nO modelo obteve um total de ",total," classificações corretas" )
print("\nRepresentando uma taxa de eficácia de: ","{:.2f}".format((total/m)*100),"%")
q=int(input("\nGostaria de ver os dígitos que foram classificados incorretamente? (1 para sim, 2 para não) "))
if q==1:
    missFeedback(X,Y,pred)


    