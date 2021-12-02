import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def TesteLambda(X_train,Y_train,m_train,thetas,apren,alpha,n,nbr_classes,X_validation,Y_validation,m_validation,X_test,Y_test,m_test):
  lambdas=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 0.005, 0.05, 0.5, 2, 0.0001]
  lambd=100
  aux=1000
  lambdas.sort()
  custos=[]
  for k in lambdas:
      J_validation=GradientDescent(X_train,Y_train,m_train,thetas,apren,alpha,k,n,nbr_classes,X_validation,Y_validation,m_validation,X_test,Y_test,m_test)[5]
      custos.append(J_validation[-1])
      if J_validation[-1] < aux:
         aux=cp.deepcopy(J_validation[-1])
         lambd=k  
  plt.plot(lambdas,custos,"r",label="Custo em Função de Lambda")
  plt.legend()
  plt.show()       
  return lambd        

 
def Division(div,data_2,data_3,m,n,nbr_classes):
    data_joined = data_2.join(data_3)
    data_random = data_joined.sample(frac=1)
    m_train=int(m*div[0])
    m_test=int(m*div[1])
    m_valid=int(m*div[2])
    df_train = data_random.iloc[:m_train,:]
    X_train, y_train = df_train.iloc[:,:-1].values, df_train.iloc[:,-1].values
    X_train = np.append(np.ones([m_train,1]),X_train,axis=1)
    df_test = data_random.iloc[m_train:(m_train+m_test),:]
    X_test, y_test = df_test.iloc[:,:-1].values, df_test.iloc[:,-1].values
    X_test = np.append(np.ones([m_test,1]),X_test,axis=1)
    df_valid=data_random.iloc[(m_train+m_test+2):,:]
    X_valid, y_valid= df_valid.iloc[:,:-1].values, df_valid.iloc[:,-1].values
    X_valid=np.append(np.ones([m_valid,1]),X_valid,axis=1)
    aux_1= []
    for i in range(m_train):
        aux_1.append(np.array([1 if y_train[i] == j else 0 for j in range(nbr_classes)]))
    Y_train = np.array(aux_1).reshape(-1, nbr_classes)
    aux_2= []
    for k in range(m_test):
        aux_2.append(np.array([1 if y_test[k] == h else 0 for h in range(nbr_classes)]))
    Y_test = np.array(aux_2).reshape(-1, nbr_classes)
    aux_3= []
    for t in range(m_valid):
        aux_3.append(np.array([1 if y_valid[t] == l else 0 for l in range(nbr_classes)]))
    Y_valid=np.array(aux_3).reshape(-1,nbr_classes)
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid,m_train,m_test,m_valid

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

def CostFunction(nbr_classes,k,A,Y,m,thetas):
    J=0
    for h in range(nbr_classes):
        J = J + np.sum(-Y[:,h]*np.log(A[-1][:,h])-(1-Y[:,h])*np.log(1-A[-1][:,h]))
    J=(1/m)*J
    for g in thetas:
        J=J+(k/(2*m))*(np.sum(g[:,1:]**2))    
    return J    

    
def ForwardPropagation(X,b,thetas):
    A=[]
    aux=X
    m=X.shape[0]
    for i in range(0,b-2):
        a=sigmoid(aux @ thetas[i].T)
        a=np.hstack((np.ones((m,1)), a))
        A.append(a)
        aux=a
    A.append(sigmoid(A[-1] @ thetas[-1].T))   
    return A
    
def BackPropagation(X,Y,A,thetas):
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
    return list(reversed(grads)),list(reversed(erro))

def GradientDescent(X,Y,m,thetas,alpha,p,k,n,nbr_classes,X_aux,Y_aux,m_aux,X_aux2,Y_aux2,m_aux2):
    J=[]
    J_aux=[]
    J_aux2=[]
    for i in range(p):
        e,grads,A,erro,Jaux,Jaux2=Compute_Cost(X,Y,m,thetas,n,b,nbr_classes,k,X_aux,Y_aux,m_aux,X_aux2,Y_aux2,m_aux2)
        for j in range(0,len(thetas)):
            thetas[j]=thetas[j]-alpha*grads[j]
        J.append(e)
        J_aux.append(Jaux)
        J_aux2.append(Jaux2)
    return thetas, J, A, grads, erro, J_aux, J_aux2        

def Compute_Cost(X,Y,m,thetas,n,b,nbr_classes,k,X_aux,Y_aux,m_aux,X_aux2,Y_aux2,m_aux2):
    A=ForwardPropagation(X,b,thetas)    
    A_aux=ForwardPropagation(X_aux,b,thetas)
    A_aux2=ForwardPropagation(X_aux2,b,thetas)
    J=CostFunction(nbr_classes,k,A,Y,m,thetas)
    J_aux=CostFunction(nbr_classes,k,A_aux,Y_aux,m_aux,thetas)
    J_aux2=CostFunction(nbr_classes,k,A_aux2,Y_aux2,m_aux2,thetas)
    grads,erro=BackPropagation(X,Y,A,thetas)
    for r in range(0,len(grads)):
        grads[r]=(1/m)*grads[r]
        grads[r]=grads[r]+(k/m)*np.hstack((np.zeros((thetas[r].shape[0],1)),thetas[r][:,1:]))    
    return J,grads,A,erro, J_aux, J_aux2
              
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


data_2=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\imageMNIST.csv')
data_3=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\labelMNIST.csv')
X,Y,m,n,nbr_classes=Load()
div = list(map(float,input("Qual seria a porcentagem de dados para treino,teste e validação? Oferecer os valores na ordem e em formato decimal ").strip().split()))[:3]
b=int(input("\nQual seria o número de camadas? "))
X_train, X_test, X_validation, Y_train, Y_test, Y_validation,m_train,m_test,m_validation=Division(div,data_2,data_3,m,n,nbr_classes)
a = list(map(int,input("\nQual seria o número de neurônios em cada camada escondida? ").strip().split()))[:b]
thetas=Theta_Initialization(b,a,X_train,Y_train)
save=cp.deepcopy(thetas)
alpha=int(input("\nQual seria o número de iterações? "))
apren=float(input("\nQual seria a taxa de aprendizado? "))
lambd=TesteLambda(X_train,Y_train,m_train,thetas,apren,100,n,nbr_classes,X_validation,Y_validation,m_validation,X_test,Y_test,m_test)
thetas=save
J_train=[]
J_validation=[]
J_teste=[]
print("O valor ótimo para lambda obtido automaticamente foi:",lambd)
thetas,J_train,A,grads,erro,J_validation, J_teste=GradientDescent(X_train,Y_train,m_train,thetas,apren,alpha,lambd,n,nbr_classes,X_validation,Y_validation,m_validation,X_test,Y_test,m_test)
l=int(input("\nGostaria de ver as curvas de custo por iteração ? (1 para sim, 2 para não) "))
if l==1:    
    plt.plot(range(len(J_teste)),J_teste,"g",label="Custo de Teste")
    plt.legend()
    plt.show()
    plt.plot(range(len(J_validation)),J_validation,"r",label="Custo de Validação")
    plt.plot(range(len(J_train)),J_train,"b",label="Custo de Treino")
    plt.legend()
    plt.show()


    