import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def image_visualization(thetas):
    visu=cp.deepcopy(thetas[0])
    for i in range(visu.shape[0]):
        visu[i,1:]=(visu[i,1:]-np.min(visu[i,1:]))/(np.max(visu[i,1:])-np.min(visu[i,1:]))
        plt.figure()
        plt.imshow(visu[i,1:].reshape(20,20),cmap="gray")
        plt.title("Unidade Escondida {}".format(i+1))
        plt.show()    

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

def GradientDescent(X,Y,m,thetas,alpha,p,k,n,nbr_classes):
    J=[]
    for i in range(p):
        e,grads,A,erro=Compute_Cost(X,Y,m,thetas,n,b,nbr_classes,k)
        for j in range(0,len(thetas)):
            thetas[j]=thetas[j]-alpha*grads[j]
        J.append(e)
    return thetas, J, A, grads, erro        

def Compute_Cost(X,Y,m,thetas,n,b,nbr_classes,k):
    A=ForwardPropagation(X,b,thetas)    
    J=CostFunction(nbr_classes,k,A,Y,m,thetas)
    grads,erro=BackPropagation(X,Y,A,thetas)
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
    A=ForwardPropagation(X,b,thetas)
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


def GradientChecking(lambd,b):
    aux_theta=[]
    input_layer_size=3
    hidden_layer_size=5
    num_labels=3
    m=5
    aux_theta.append(AssistWeight(hidden_layer_size,input_layer_size))
    for i in range(1,b-2):
        aux_theta.append(AssistWeight(hidden_layer_size, hidden_layer_size))
    aux_theta.append(AssistWeight(num_labels, hidden_layer_size))
    aux_X=np.append(np.ones((m,1)),AssistWeight(m,input_layer_size-1), axis=1)
    n=aux_X.shape[1]-1
    aux_Y=np.matrix('1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1')
    aux_grads=Compute_Cost(aux_X,aux_Y,m,aux_theta,n,b,num_labels,lambd)[1]
    epsilon=1e-4
    grads_aux=[]
    for p in range(0,len(aux_theta)):
        grads_aux.append(np.zeros(aux_theta[p].shape))
    cont=0
    for x in aux_theta:
        for y in range(0,x.shape[0]):
            for z in range(0,x.shape[1]):
                assist=cp.deepcopy(aux_theta)
                minnum=cp.deepcopy(x)
                minnum[y,z]=minnum[y,z]-epsilon
                maxnum=cp.deepcopy(x)
                maxnum[y,z]=maxnum[y,z]+epsilon
                assist[cont]=cp.deepcopy(minnum)
                costmin=Compute_Cost(aux_X, aux_Y, m, assist, n, b, num_labels, lambd)[0]
                minnum[y,z]=minnum[y,z]+epsilon
                assist[cont]=cp.deepcopy(maxnum)
                costmax=Compute_Cost(aux_X, aux_Y, m, assist, n, b, num_labels, lambd)[0]
                maxnum[y,z]=maxnum[y,z]-epsilon
                grads_aux[cont][y,z]=(costmax-costmin)/(2*epsilon)
                assist[cont]=x
        cont=cont+1        
    for u in range(0,len(grads_aux)):
        grads_aux[u]=(1/m)*grads_aux[u]
        grads_aux[u]=grads_aux[u]+(lambd/m)*np.hstack((np.zeros((aux_theta[u].shape[0],1)),aux_theta[u][:,1:]))
    l=[]
    for h in range(0,len(grads_aux)):
       l.append(np.mean(np.abs(grads_aux[h]-aux_grads[h])))
    diff=np.mean(l)
    return aux_theta,aux_grads, grads_aux, diff  

data_2=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\imageMNIST.csv')
data_3=pd.read_csv(r'C:\Users\USUARIO\Desktop\Neural Network\labelMNIST.csv')
X,Y,m,n,nbr_classes=Load()
b=int(input("Qual seria o número de camadas? "))
a = list(map(int,input("\nQual seria o número de neurônios em cada camada escondida? ").strip().split()))[:b]
thetas=Theta_Initialization(b,a,X,Y)
alpha=int(input("\nQual seria o número de iterações? "))
apren=float(input("\nQual seria a taxa de aprendizado? "))
lambd=float(input("\nQual seria o valor de lambda? "))
w=int(input("\nGostaria de verificar a checagem de gradiente? 1 para sim, 2 para não "))
if w==1:
    aux_theta,aux_grads,grads_aux, diff=GradientChecking(lambd,b)
    print("\nA diferença na aproximação é de:",diff)
thetas,J,A,grads,erro=GradientDescent(X,Y,m,thetas,apren,alpha,lambd,n,nbr_classes)
v=int(input("\nGostaria de ter uma visualização de como cada unidade escondida atua na rede? 1 para sim, 2 para não "))
if v==1:
    image_visualization(thetas)
pred,real,total=Prediction(Y,X,b,m,thetas)
print("\nO modelo obteve um total de ",total," classificações corretas" )
print("\nRepresentando uma taxa de eficácia de: ","{:.2f}".format((total/m)*100),"%")
l=int(input("\nGostaria de ver a curva de custo por iteração? (1 para sim, 2 para não) "))
q=int(input("\nGostaria de ver os dígitos que foram classificados incorretamente? (1 para sim, 2 para não) "))
if q==1:
    missFeedback(X,Y,pred)
if l==1:    
    plt.plot(range(len(J)),J)
    plt.xlabel('Número de iterações')
    plt.ylabel('Custo')
    plt.show()


    