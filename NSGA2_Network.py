from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import math
import random
import matplotlib.pyplot as plt

min_x=-55
max_x=55
_SwarmSize=50
max_gen = 100
gen_no=0
RAND_MAX = 32767
path = 'D://Jupyter_path//ML_ASS2//creditcard.csv'



# plot ROC Curve



def draw_roc(y_test,preds):
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
# random value generato
def randn(nmin, nmax):
    thisRand = ((random.randint(0, RAND_MAX) / (RAND_MAX + 1.0))) * (nmax - nmin) + nmin;
    return thisRand



def get_para(model):
    para = []
    for j in range(len(model.coefs_)):
        for k in range(len((model.coefs_[j]))):
            # if len((model.coefs_[j]))==1:
            #     coeff.append((model.coefs_[j]))
            for l in range((len(model.coefs_[j][k]))):
                para.append(model.coefs_[j][k][l])
    for j in range (len(model.intercepts_)):
        for k in range(len((model.intercepts_[j]))):
            para.append(model.intercepts_[j][k])
    return para

def set_para(model,solution):
    count = 0
    for j in range(len(model.coefs_)):
        for k in range((len(model.coefs_[j]))):
            for l in range((len(model.coefs_[j][k]))):
                model.coefs_[j][k][l] = solution[count]
                count += 1
    for j in range(len(model.intercepts_)):
        for k in range((len(model.intercepts_[j]))):
            model.intercepts_[j][k] = solution[count]
            count += 1



def get_coeff(model):
    coeff=[]
    for j in range (len(model.coefs_)):
        for k in range(len((model.coefs_[j]))):
            # if len((model.coefs_[j]))==1:
            #     coeff.append((model.coefs_[j]))
            for l in range((len(model.coefs_[j][k]))):

                coeff.append(model.coefs_[j][k][l])
    return coeff

def set_coeff(model,solution):
    count = 0
    for j in range(len(model.coefs_)):
        for k in range((len(model.coefs_[j]))):
            for l in range((len(model.coefs_[j][k]))):
                model.coefs_[j][k][l] = solution[count]
                count += 1
def get_bias(model):
    bias=[]
    for j in range (len(model.intercepts_)):
        for k in range(len((model.intercepts_[j]))):
            bias.append(model.intercepts_[j][k])
    return bias

def set_bias(model,solution):
    count = 0
    for j in range(len(model.intercepts_)):
        for k in range((len(model.intercepts_[j]))):
            model.intercepts_[j][k] = solution[count]
            count += 1

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))

        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            #如果q没被主宰，加入到S里
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            # 如果被主宰了，未被主宰次数+1
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1)+0.00000001)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2)+0.00000001)
    return distance

#Function to carry out the crossover
def crossover(a,b):
    child1=np.zeros([len(a)])
    child2=np.zeros([len(a)])
    gama1=randn(0,1)
    gama2 = randn(0, 1)
    pos_sep=random.randint(1,len(a)-1)
    for i in range (pos_sep):
        child1[i] = gama1 * a[i] + (1 - gama1) * b[i]
        child2[i] = gama1 * b[i] + (1 - gama1) * a[i]
    for i in range (pos_sep,len(a)):
        child1[i] = gama2 * a[i] + (1 - gama2) * b[i]
        child2[i] = gama2 * b[i] + (1 - gama2) * a[i]
    return child1

#Function to carry out the mutation operator
def mutation(solution):
    for i in range (len(solution)):
        if randn(0, 1)<0.05:
            solution[i]=randn(0, 1)*solution[i]
    return solution



# import data
df = pd.read_csv(path)

def preprocessing (df):
    count_class_0, count_class_1 = df.Class.value_counts()
    # over sampling
    df_class_0 = df[df['Class'] == 0]
    df_class_1 = df[df['Class'] == 1]
#     print(df_class_1)
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    df_test_over=df_test_over.sample(n=2000)

    x = df_test_over.drop('Class',1)
    # y = df_test_over.Class.astype('category',copy=False)
    y = df_test_over.Class.astype('category',copy=False)

    return x,y

X,y = preprocessing(df)
print(len(X))

xf_train, xf_valid, yf_train, yf_valid= train_test_split(X,y,random_state=42, test_size=0.2)


mlpi=MLPClassifier()
cv = KFold(n_splits=_SwarmSize, random_state=42, shuffle=False)

mlpi=MLPClassifier()
mlp=[]
f1=[]
f2=[]
for train_index, test_index in cv.split(xf_train):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)
    # cross validation
    X_train, X_test, y_train, y_test = xf_train.iloc[train_index], xf_train.iloc[test_index], yf_train.iloc[train_index], yf_train.iloc[test_index]
    mlpi=MLPClassifier(hidden_layer_sizes=(10,5,2),max_iter=100)
    # print(len(X_train),len(y_train))
    mlpi.fit(X_train,y_train)
    predi=mlpi.predict(xf_valid)
    accuracyi=accuracy_score(yf_valid, predi)
    # print("accuracy_score = ", accurayi)
    roci=roc_auc_score(yf_valid, predi)
    f1.append(accuracyi)
    f2.append(roci)
    parai=get_para(mlpi)
    mlp.append(np.asarray(parai))
    # validate models before NSGA-II
    yf_predi=mlpi.predict(xf_valid)

f1_np=np.asarray(f1)
f2_np=np.asarray(f2)

# print("Before GA, accuracies for models are: ",1-f1_np)
print("the best accuracy is: ",f1_np.max())


bestind=f1_np.argmax()
# print("Before GA, AUC for models are,",1-f2_np)
print("the most accurate one's AUC is: ",f2_np[bestind])

set_para(mlpi,mlp[bestind])
yp_test=mlpi.predict(xf_valid)
cm=confusion_matrix(yf_valid, yp_test)
print("The confusion_matrix of the most accurate one before GA is :")
print(cm)


draw_roc(yf_valid,yp_test)


# into NSGA_II loop
solution=mlp[:]
# print(solutions)
pop_size=len(solution)

# test value
# minind=f1_np.argmin()
# set_para(mlpi,solution[minind])
# test_y=mlpi.predict(xf_valid)
# print("score after extract and implementation is: ",accuracy_score(yf_valid, test_y))
#
#

print("before GA, ",f1_np.argmax(),"the most accurate particle is: ",f1_np.max())
print("into GA.")
function1_values = [f1[i] for i in range (len(f1))]
function2_values = [f2[i] for i in range (len(f2))]
# print(f1)
# print(function1_values)
    # 构建一个以 solution为参数的neural network, 得到f1,f2的值
while(gen_no<max_gen):

    #这里的出的是non-dominated解的index
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    # print(non_dominated_sorted_solution)
    # print("The best front for Generation number ", " is")
    # for valuez in non_dominated_sorted_solution[0]:
    #     print(round(solution[valuez],3),end=" ")
    # print("\n")

    # print("into crowd calculation")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    # print("initiate solution2")
    solution2 = solution[:]
    # print("Generating offsprings")
    # print("before crossover, the most accurate one is: ", max(function1_values))
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        if randn(0,1)>0.5:
            solution2.append(crossover(solution[a1],solution[b1]))
    # mutate offsprings
    for i in range (pop_size,len(solution2)):
        solution2[i]=mutation(solution2[i])


    # 必须用test_set测试model
    function1_values2=[]
    function2_values2=[]
    # print("evaluation children...")
    for i in range(len(solution2)):
        set_para(mlpi,solution2[i])
        y_pred1=mlpi.predict(xf_valid)
        function1_values2.append(accuracy_score(yf_valid, y_pred1))
        function2_values2.append(roc_auc_score(yf_valid, y_pred1))
    # print("after crossover and mutation, the most accurate one is: ", max(function1_values2))
    # print(len(function1_values))
    # print(len(function1_values2))

    # function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    # function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    # print("Sorting children...")
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    # print("after sorting, the most accurate one is: ", max(function1_values2))

    # print("get new solutions")
    # print(len(new_solution))
    # 得到最终的child,此处加入BP
    solution = [solution2[i] for i in new_solution]
    # solution,f1,f2 =BP(soluiton)，输入和输出都是n个solution(network的参数)
    # update fitness values
    function1_values=[]
    function2_values=[]
    for i in range (pop_size):
        set_para(mlpi,solution[i])
        y_pred2=mlpi.predict(xf_valid)
        function1_values.append(accuracy_score(yf_valid, y_pred2))
        function2_values.append(roc_auc_score(yf_valid, y_pred2))

    gen_no = gen_no + 1

# dist=[mlp[i]-solution[i] for i in range (len(solution))]
# print(dist)

print("visualize")
# Lets plot the final front now
function1 = [i  for i in function1_values]
function2 = [j  for j in function2_values]
#

f=np.asarray(function1)+np.asarray(function2)

best_ind=f.argmax()
set_para(mlpi,solution[best_ind])
yf_pred=mlpi.predict(xf_valid)
cm2=confusion_matrix(yf_valid, yf_pred)
print("The confusion_matrix for the best ANNClassifier is :")
print(cm2)

print("After GA, the best accuracy is: ",function1[best_ind])
print("After GA, the best AUC is: ",function2[best_ind])

draw_roc(yf_valid,yf_pred)

# plt.xlabel('Function 1', fontsize=15)
# plt.ylabel('Function 2', fontsize=15)
# print(function1)
# print(function2)
# plt.scatter(function1, function2)
# plt.show

