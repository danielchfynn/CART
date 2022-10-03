#Future ideas 
#cross validation 
#check if between variance is ok, swap for deviance
#speed up multiprocessing or joblib 



import itertools
import math
import numpy as np # use numpy arraysfrom
from  statistics import mean,variance,mode
from anytree import Node, RenderTree, NodeMixin
from collections import Counter
import matplotlib.pyplot as plt
import math
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
from anytree.exporter import DictExporter
import pydot
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
from pygments import highlight

        
# define base class for nodes
class MyBaseClass(object):  # Just a basic base class
    value = None            # it only brings the node value


class MyNodeClass(MyBaseClass, NodeMixin):  # Add Node feature
    
    children = []
    value_soglia_split = []
    
    
    def __init__(self, name, indexes, split=None, parent=None,node_level= 0,to_pop = False):
        super(MyNodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        #self.impurity = impurity          # vue in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        self.node_level = node_level       # Tiene traccia del livello dei nodi all'interno dell albero in ordine crescente : il root node avrà livello 0
        self.to_pop = to_pop
        

    def get_value(self, y, problem):
        if problem =='regression':
            return mean(y[self.indexes])
        else:
            response_dict ={}
            for response in y[self.indexes]:        #determing majority in  nodes
                if response in response_dict:
                    response_dict[response] +=1
            else:
                response_dict[response] =1
        return max(response_dict, key = response_dict.get)


    def get_name_as_number(self):
        '''
        new name's node defination with integer
        '''
        return int(self.get_name()[1:])
    

    def get_children(self):
        '''
        ritorna il figlio
        se esiste 
        altrimenti none
        
        '''
        return self.children
    

    def get_value_thresh(self):
        return self.value_soglia_split[0][0:2] + [self.value_soglia_split[0][3]]
        

    def set_to_pop(self):
        '''
        Durante il growing tiene traccia dei nodi da potare.
        '''
        self.to_pop = True 


    def get_name(self):
        return self.name
    

    def get_level(self):
        return self.node_level
    

    def set_features(self,features):
        self.features = features
    

    def get_parent(self):
        '''
        return the parent node 
        if the the parent node is None , is the root.
        '''
        return self.parent
    

    def set_children(self,lista:list):#lista di nodi    
        for i in lista:
            self.children.append(i)
    
    
    def set_split(self,value_soglia):
        self.value_soglia_split = value_soglia
        
        
    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        if var_name in feat:         #is_numeric(var) :      # split for numerical variables
            var = self.features[var_name]    # obtains the var column by identifiying the feature name 
            self.split = var_name + ">" + str(soglia) # compose the split string (just for numerical features)
            parent = self.name
            select = var[(self.indexes)] > soglia              # split cases belonging to the parent node
        elif  var_name in feat_nominal:         #is_numeric(var) :      # split for nominal variables
            var = feat_nominal[var_name]    # obtains the var column by identifiying the nominal feature name 

            if type(soglia) is tuple:
                self.split = var_name + " in " + str(soglia) # compose the split string (just for numerical features)
            else:
                self.split = var_name + " in " + "'" +str(soglia)+"'" 

            parent = self.name
            select = np.array([i in soglia for i in var[(self.indexes)]]) # split cases belonging to the parent node
        else :
            print("Var name is not among the supplied features!")
            return
        
        left_i = self.indexes[~select]                      # to the left child criterion FALSE
        right_i = self.indexes[select]                      # to the right child criterion TRUE
        child_l = "n" + str(int(parent.replace("n",""))*2)
        child_r = "n" + str(int(parent.replace("n",""))*2 + 1)         
        return MyNodeClass(child_l, left_i, None, parent = self,node_level=self.node_level+1), MyNodeClass(child_r, right_i, None, parent = self,node_level=self.node_level+1)   # instantiate left & right children
            
    # add a method to fast render the tree in ASCII
    def print_tree(self):
        for pre, _, node in RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.split, node.indexes)


class CART:
    '''
    bigtree =  []
    nsplit = 0
    father = []
    root = []
    tree = []
    father_to_pop = []
    node_prop_list = []
    grow_rules = {}
    leaf = []
    all_node = []
    prediction_cat = []
    prediction_reg = []
    '''

    def __init__(self,y,features,features_names,n_features \
                    ,n_features_names, impurity_fn, user_impur=None, problem = "regression"  \
                    ,min_cases_parent = 10 \
                    ,min_cases_child = 5\
                    ,min_imp_gain=0.01
                    ,max_level = 10):

        self.y = y
        self.features = features
        self.features_names = features_names
        self.n_features = n_features
        self.n_features_names = n_features_names
        self.problem = problem
        self.impurity_fn = impurity_fn
        if problem =="regression":
            self.devian_y = len(self.y)*variance(self.y) # impurity function will equal between variance 
        elif problem == "classifier":
            n_class = len(np.unique(self.y))
            pro = []
            c = Counter(y) 
            c = list(c.items())
            for i in  c:
                prob = i[1]/len(self.y)
                pro.append(math.log(i[1]/len(y),2)*i[1]/len(y))
            pro = np.array(pro)

            self.devian_y = len(y)*np.sum(pro)*(-1) 
        self.user_impur = user_impur
        self.max_level = max_level
             
        self.grow_rules = dict({'min_cases_parent':min_cases_parent \
                                     ,'min_cases_child':min_cases_child \
                                     ,'min_imp_gain':min_imp_gain})
    
        self.bigtree =  []
        self.nsplit = 0
        self.father = []
        self.root = []
        self.tree = []
        self.father_to_pop = []
        self.node_prop_list = []
        #self.grow_rules = {}
        self.leaf = []
        self.all_node = []
        self.prediction_cat = []
        self.prediction_reg = []

    def user_impur_fn(self, func, node):
        return func(self, node)
        

    def impur(self,node, display = False):
        if self.problem =='regression':
            if self.impurity_fn =="between_variance":
                return (mean(self.y[node.indexes])**2)*len(self.y[node.indexes])
            
            elif self.impurity_fn =="user_defined":
                if self.user_impur:
                    return self.user_impur_fn(self.user_impur, node)
                else:
                    print("Must define 'user_impur' if selecting 'user_defined' for 'impur_fn'")
                 
            else:
                print("Impurity-fn only defined for between variance for regression problem.")
        elif self.problem == 'classifier':

            prom = 0
            c = Counter(self.y[node.indexes]) #Creates a dictionary {"yes":number, "no"}
            c = list(c.items())

            if self.impurity_fn =="gini":
                for i in  c:
                    prob_i = float((i[1]/len(self.y[node.indexes])))**2 #probability squared
                    
                    if display:
                        prom += prob_i
                    else:
                        prom += prob_i*i[1] #original weighted, only looking at purity
                        #prom += prob_i
                if display:
                    return 1-prom 
                else:  
                    return prom
            elif self.impurity_fn =="entropy":
                
                for i in c:
                    prob_i = float((i[1]/len(self.y[node.indexes]))) #probabillity

                    prom += prob_i*math.log(prob_i,2)  #*i[1]

                    #prom += prob_i*math.log2(prob_i)  #*i[1]
                return -prom#/len(c)
            elif self.impurity_fn =="user_defined":
                if self.user_impur:
                    return self.user_impur_fn(self.user_impur, node)
                else:
                    print("Must define 'user_impur' if selecting 'user_defined' for 'impur_fn'")

            else:
                print("For classification problem, impurity_fn must be set to either 'gini' or 'entropy' or 'user_defined'")
        else:
            print("'problem' must be classified as either 'regression' or 'classifier'")     


    def get_number_split(self):
        return self.nsplit
    

    def get_leaf(self):
        leaf = [inode for inode in self.bigtree if inode not in self.get_father() ]
        le = []
        for i in leaf:
            if i not in le:
                le.append(i)
        self.leaf = [inode for inode in le if inode.to_pop == False]
        return   [inode for inode in le if inode.to_pop == False]
    

    def get_father(self):
        '''
        return all the node father
        '''
        return [inode for inode in self.father if inode not in self.father_to_pop]


    def get_root(self):
        return self.root


    def __get_RSS(self,node):
        '''
        return the RSS of a node
        this funcion is for only internal uses (private_funcion)
        '''
        mean_y = mean(self.y[node.indexes])
        return (1/len(node.indexes)*sum((self.y[node.indexes] - mean_y)**2))


    def get_all_node(self):
        foglie = [nodi for nodi in self.get_leaf()]
        self.all_node = foglie + self.get_father()
        return foglie + self.get_father()


    def __node_search_split(self,node:MyNodeClass,features,features_names):

        '''
        The function return the best split thath the node may compute.
        Il calcolo è effettuato effettuando ogni possibile split e 
        calcolando la massima between variance 
        tra i nodi figli creati.
       
       Attenzione: questo è un metodo privato non chiamabile a di fuori della classe.
        '''
        
        impurities_1=[]
        between_variance=[]
        splits=[]
        variables=[]
        combinazioni=[]
        distinct_values=np.array([])
        t=0
        
        node.set_features(self.features)
        



        ''' removed with problems from adaboost implementation 24/97/22, it removes pure nodes? 

        if Counter(self.y[node.indexes]).most_common(1)[0][1] == len(self.y[node.indexes]):
            
            print("node name", node.name, "counter", Counter(self.y[node.indexes]).most_common,   Counter(self.y[node.indexes]).most_common(1)[0][1], "len y for node index", len(self.y[node.indexes]))
            print("A - This split isn't good now i cut it")
            node.get_parent().set_to_pop()
            node.get_parent().set_to_pop()
            self.father_to_pop.append(node)

            return None
        '''




        if len(node.indexes) >= self.grow_rules['min_cases_parent']:
            
            for var in self.n_features_names:

                distinct_values=np.array([])                
                distinct_values=np.append(distinct_values,np.unique(self.n_features[str(var)]))
                
                for i in range(1,len(distinct_values)):
                    combinazioni.append(list(itertools.combinations(np.unique(self.n_features[str(var)]), i)))
                combinazioni=combinazioni[1:]
                for index in combinazioni: 
                    for i in index:
                        stump = node.bin_split(self.features, self.n_features, str(var),i)
                        if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                            and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                            impur0 = self.impur(stump[0])
                            impur1 = self.impur(stump[1])
                            if self.problem == 'classifier':    
                                #impur_father = self.impur(stump[0].get_parent())
                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)#stump[0].indexes + stump[1].indexes
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    between_variance.append(inf_gain)                                
                                else:
                                    between_variance.append((impur0) + (impur1))
                            else: 
                                impurities_1.append(impur0)
                                impurities_1.append(impur1)
                                between_variance.append(sum(impurities_1[t:]))
                    
                            splits.append(i)
                            variables.append(str(var))
                            t+=2
                    else:
                        continue
                        
                combinazioni=[]
                distinct_values=np.array([])
                distinct_values=list(np.append(distinct_values,np.unique(self.n_features[str(var)])))
                    
                for i in range(len(distinct_values)):
                    stump = node.bin_split(self.features, self.n_features, str(var),distinct_values[i])
                    if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child']  \
                    and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                        impur0 = self.impur(stump[0])
                        impur1 = self.impur(stump[1])
                        if self.problem == 'classifier':    
                            if self.impurity_fn =="entropy":
                                entropy_parent = self.impur(node)#stump[0].indexes + stump[1].indexes
                                inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                between_variance.append(inf_gain)                        
                            else:        
                                impur_father = self.impur(stump[0].get_parent())
                                between_variance.append((impur0) +(impur1))
                        else: 
                            impurities_1.append(impur0)
                            impurities_1.append(impur1)
                            between_variance.append(sum(impurities_1[t:]))
                        splits.append(distinct_values[i])
                        variables.append(str(var))
                        t+=2
                    else:
                        continue
                                
            for var in self.features_names:
                for i in range(len(self.features[str(var)])):
                        stump = node.bin_split(self.features, self.n_features, str(var),self.features[str(var)][i])
                        if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                            impur0 = self.impur(stump[0])
                            impur1 = self.impur(stump[1])
                            if self.problem == 'classifier':    
                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)#stump[0].indexes + stump[1].indexes
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    between_variance.append(inf_gain)       
                                else:
                                    #impur_father = self.impur(stump[0].get_parent())
                                    between_variance.append((impur0) + (impur1))
                            
                            else: 
                                impurities_1.append(impur0)
                                impurities_1.append(impur1)
                                between_variance.append(sum(impurities_1[t:]))
                            
                            splits.append(self.features[str(var)][i])
                            variables.append(str(var))
                            t+=2
                        else: 
                            continue
        
        '''
        if self.problem == 'classifier':         
            try:
                return variables[between_variance.index(min(between_variance))],splits[between_variance.index(min(between_variance))],between_variance[between_variance.index(min(between_variance))]
            except:
                print(between_variance,impurities_1)
        else:
            '''
     
        try:
            return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))]
        except:
            node.get_parent().set_to_pop()
            node.get_parent().set_to_pop()
            self.father_to_pop.append(node)
            return None
    

    def control(self):
        for i in self.get_leaf():
            for j in self.get_leaf():
                if i.get_parent() == j.get_parent():
                    if mode(self.y[i.indexes]) == mode(self.y[j.indexes]):
                        #i.set_to_pop()
                        #set_to_pop()
                        self.father_to_pop.append(i.get_parent)
        
    ''' 
    def __ex_devian(self,varian,nodo):
        if self.problem =='regression':
            return varian - len(self.y)*mean(self.y)**2
        elif self.problem == 'classifier':
            
            prop = Counter(self.y[nodo.indexes])
            prop = list(prop.items())
            som = []
            for i in prop:
                som.append((i[1]/len(self.y[nodo.indexes]))**2)
            
            return varian - sum(som)              
    '''     
        
    
    def deviance_cat(self,node):
        #entropy
        pro = []
        c = Counter(self.y[node.indexes])
        c = list(c.items())
        p = len(self.y[node.indexes])
        for i in  c:
            
            prob = i[1]/p
            
            pro.append(math.log(prob,2)*prob)
            
        pro = np.array(pro)
        ex_deviance = -1*np.sum(pro)  
        return ex_deviance
    

    def prop_nodo(self,node):
        c = Counter(self.y[node.indexes])
        c = list(c.items())
        p = len(self.y[node.indexes])
        xlen = len(self.y)
        somm=  0
        for i in  c:
            
            prob = i[1]/p
            somm +=prob
        
        return prob
            
    
    def fit(self,node:Node,rout='start',propotion_total=0.9):
        
        value_soglia_variance = []
        mini_tree = [] 

        try:
            
            value,soglia,varian = self.__node_search_split(node,self.features,self.features_names)                

        except TypeError:

            #self.father_to_pop.append(node)
            
            return None
        

        level = node.get_level()

        if level > self.max_level:
            return None 

        value_soglia_variance.append([value,soglia,varian,level])
    
        
        self.root.append((value_soglia_variance,rout))

        left_node,right_node = node.bin_split(self.features, self.n_features, str(value),soglia)
        node.set_children((left_node,right_node))
        node.set_split(value_soglia_variance)

        mini_tree.append((node,left_node,right_node))
        self.tree.append(mini_tree) 
        self.bigtree.append(node)
        if rout != 'start':
            self.father.append(node) #append in 
        self.bigtree.append(node)#append nodo padre
        self.bigtree.append(left_node)#append nodo figlio sinistro
        self.bigtree.append(right_node)#append nodo figlio desto
        print("i find new_split : ",value_soglia_variance,rout)

    ###### Calcolo della deviance nel nodo  

        if rout == 'start':
            self.father.append(node)
            if self.problem=='regression':
                ex_deviance = varian - len(self.y)*mean(self.y)**2
            elif self.problem == "classifier":

                ex_deviance = self.deviance_cat(left_node) + self.deviance_cat(right_node)
                          
        else:
            ex_deviance_list= []
            for inode in self.bigtree:
                if inode not in self.father:
                    #print("inode figlio ", inode)
                    if self.problem == 'regression':
                        ex_deviance_list.append(len(self.y[inode.indexes])*(mean(self.y[inode.indexes])-mean(self.y))**2)
                    elif self.problem == 'classifier':
                        ex_deviance_list.append(self.deviance_cat(inode))

                    #ex_deviance_list.append(0)
            ex_deviance = sum(ex_deviance_list)

        node_propotion_total = ex_deviance/ self.devian_y   
        print("node_propotion_total ",node_propotion_total)
        self.node_prop_list.append(node_propotion_total)
        
        if self.problem == "regression":
            if len(self.node_prop_list)>1:
                delta = self.node_prop_list[-1] - self.node_prop_list[-2]
                print("Node_proportionale_gain ",delta)
                if delta < self.grow_rules['min_imp_gain'] :#all utente  :Controllo delle variazione nei nodi figli
                    print("This split isn't good now i cut it")
                    left_node.set_to_pop()
                    right_node.set_to_pop()
                    self.father_to_pop.append(node)
                    self.root.pop()

                    return None
    
        else:
        
            if len(self.node_prop_list)>1:
                if self.impurity_fn == "entropy":
                    entropy_parent = self.impur(node)#stump[0].indexes + stump[1].indexes
                    delta = entropy_parent - ((len(left_node.indexes) / len(node.indexes)) * self.impur(left_node) + (len(right_node.indexes) / len(node.indexes)) * self.impur(right_node))
                else:
                    delta = +self.deviance_cat(node) - (self.deviance_cat(right_node) + self.deviance_cat(left_node))
                print("Node_proportionale_gain ",delta)
                
                '''
                p = Counter(self.y[node.indexes]).most_common(1)
                #c = len(self.y[node.indexes])-p[0][1] 
                
                p1 = Counter(self.y[left_node.indexes]).most_common(1)
                c1 = len(self.y[left_node.indexes])-p[0][1]  
                p2 = Counter(self.y[right_node.indexes]).most_common(1)
                c2 = len(self.y[right_node.indexes])-p[0][1] 
                
                if c < (c1+c2):
                    print("This split isn't good now i cut it")
                    left_node.set_to_pop()
                    right_node.set_to_pop()
                    self.father_to_pop.append(node)
                    self.root.pop()
                    return None 
                '''
                
                if abs(delta) < self.grow_rules['min_imp_gain'] :#all utente  :Controllo delle variazione nei nodi figli
                    print("This split isn't good now i cut it")
                    left_node.set_to_pop()
                    right_node.set_to_pop()
                    self.father_to_pop.append(node)
                    self.root.pop()
                    return None
                
        if self.problem=="regression":
            if node_propotion_total >= propotion_total: 

                return None
        
        else:
            if node_propotion_total >= propotion_total: 

                return None
        
        self.nsplit += 1
        return self.fit(left_node,"left"),self.fit(right_node,"right")

    
    def get_key(self, my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key
    
        return "key doesn't exist"


    def identify_subtrees(self, father, leaves):
        '''Will associate each node with it's children, grandchildren etc., thus creating subtrees for each node, as if the node was the root'''
        all_nodes_dict = {}
        all_nodes_list =[]
        relative_dict={}

        for node in father:                                 
            all_nodes_dict[node] = int(node.name[1:])      #Creating a dictionary for each node as a key with their node number as the value
            all_nodes_list.append(int(node.name[1:]))      #Creating a list of all node numbers 
               
        for node in father:                             #Iterating though all nodes that have children and have the ability to have a subtree. 
            level = int(node.node_level)                #Using the level for the while loop, ensuring a stopping element, that makes sense as you progress down the tree to the leaves 
            
            if (int(node.name[1:]) *2) in all_nodes_list:  #Using the property of node numbers being related to their parents, in this case assessing the left child, which as a node number twice that of the parent 
                if node.name in relative_dict:
                    relative_dict[node].append(node.get_name_as_number()*2) #adding multiple value to a dictionary key
                else:
                    relative_dict[node] = [node.get_name_as_number()  *2] #adding the first value to a dictionary key
            if (int(node.name[1:])*2+1) in all_nodes_list:     #Same as above but assessing for the right node, which is twice the parents node number +1
                if node in relative_dict:           
                    relative_dict[node].append(node.get_name_as_number()*2+1)
                else:
                    relative_dict[node] = [node.get_name_as_number()*2+1]      
            while level > -1 and node in relative_dict: #-1 was use for the while loop, as the root node exists at level 0
                level += -1                             
                for child in relative_dict[node]:       #Allows the continual adding of children to the subtree, based on the node numbers within the dictionary. 
                                
                    if child*2 in all_nodes_list and child*2 not in relative_dict[node]:
                        if node in relative_dict:
                            relative_dict[node].append(int(child)*2)
                        else:
                            relative_dict[node] = [int(node.name[1:])*2]                        
                    if child*2+1 in all_nodes_list and child*2+1 not in relative_dict[node]:
                        if node in relative_dict:
                            relative_dict[node].append(int(child)*2+1)                  
                        else:
                            relative_dict[node].append(int((node.name[1:]))*2+1)
       
        only_leaves_dictionary ={}
        for element in relative_dict:
            for child in relative_dict[element]:
                if self.get_key(all_nodes_dict, child) in leaves:
                    if element in only_leaves_dictionary:
                        only_leaves_dictionary[element].append(child)
                    else:
                        only_leaves_dictionary[element] =[child]
        
        new_dict = {}   
        for key in relative_dict: #only_leaves_dictionary:
            node = []
            for i in relative_dict[key]: # only_leaves_dictionary[key]:
                for j in father:# self.get_all_node():
                    if i == int(j.name[1:]): #get_name_as_number():
                        node.append(j)
            node2 =[]
            for i in only_leaves_dictionary[key]:
                for j in father:# self.get_all_node():
                    if i == int(j.name[1:]): #get_name_as_number():
                        node2.append(j)            
            node3 = []
            node3.append(node2)
            node3.append(node)                          #when pruning need to ensure both parent and children nodes in subtree are removed
            new_dict.update({key:node3})
        
        return(new_dict)        
    

    def print_alpha(self,alpha):
        '''
        chiamare questa funzione dopo aver effettuato il calcolo degli alpha.
        Stampa a schermo tutti gli alpha.
        '''
        for i in alpha:
            print(i)    
    

    def pop_list(self,lista,lista_to_pop):
        #funzione di pura utilità.
        for i in lista_to_pop:
            lista.pop(lista.index(i))
        return lista


    def alpha_calculator(self,new_dict):
        '''
        Questa funzione ritorna il l'alpha minimo calcolato su un albero di classificazione o regressione,
        il parametro problem : stabilisce il tipo di problema
        valori accettai sono (regression,classification)
        '''
        
        alpha_tmp = []
        deviance = []
        if self.problem == 'regression':
            for key in new_dict: #key  padre
                rt_children__ = []
                rt_father= sum((self.y[key.indexes] - mean(self.y[key.indexes]))**2)
                for figli in new_dict[key][0]:
                    rt_children__.append(sum((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2))
                    deviance.append((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2)
                rt_children = sum(rt_children__)
                deviance_tot = sum(deviance)
                denom = (len(new_dict[key][0])-1)
                alpha_par = (-rt_children + rt_father)/denom         
                alpha_tmp.append((alpha_par,key,deviance_tot))
        elif self.problem == 'classifier':
            
            for key in new_dict: #key  padre
                c = Counter(self.y[key.indexes])
                p = c.most_common(1)
                c = len(self.y[key.indexes])-p[0][1]
                rt_father = c
                rt_children = 0
                for figli in new_dict[key][0]:
                    c = Counter(self.y[figli.indexes])
                    p = c.most_common(1)
                    c = len(self.y[figli.indexes])-p[0][1]
                    rt_children += c
                    
                denom = (len(new_dict[key][0])-1)
                
                if(denom <= 0):
                        denom = 0.000000001
                alpha_par = (-rt_children + rt_father)/denom
                alpha_tmp.append((alpha_par,key))
        else:
            print("error")
            exit(1)
        if len(alpha_tmp)<=1:
            alpha_tmp.append((0,None))
        return min(alpha_tmp,key=lambda l:l[0]) #alphamin
    
  
    def set_new_all_node(self,lista):
        '''
        Funzione di utilità richiamata dopo il cut
        per ridurre la dimensione dell'albero in termini della quantitò di nodi utilizzati
        '''
        self.leaf = lista
    
    
    def set_new_leaf(self,lista):
        '''
        Funzione di utilità richiamata dopo il cut
        per ridurre la dimensione dell'albero in termini della quantitò di nodi utilizzati
        come nodi foglia.
        '''
        self.all_node = lista
    

    def miss_classifications(self,list_node):
        
        if self.problem == "classifier":
            
            s = 0
            for i in list_node:
                s += len(self.y[i.indexes])-Counter(self.y[i.indexes]).most_common(1)[0][1]
        
        elif self.problem == "regression":
            
            s = 0
            comparison = []
            for node in list_node:
                #s += (mean(self.y[i.indexes])**2)*len(self.y[i.indexes]) #will need changing 
                mean_y = mean(self.y[node.indexes])
                for val in self.y[node.indexes]:
                    s+= (val - mean_y)**2
                    comparison.append([val, mean_y])
            #print("c1",comparison, "s", s, s/len(self.y))
            s = s/len(self.y)
        return s
            
        
    def pruning(self, features_test, n_features_test, y_test):
        '''
        call this function after the growing tree
        perform the pruning of the tree based on the alpha value
        Alfa = #########
        
        per ogni nodo prendi ogni finale prendi i suoi genitori verifica il livello  se è il massimo prendi i genitori
        
        '''
        all_node = self.get_all_node().copy()
        leaves = self.get_leaf().copy()


        alpha=[]  #(alpha,node) lista degli alpha minimi
        miss =[]
        leaves_for_prune = []
        leaves_mse = {}
        leaves_miss ={}
        result = []


        #Creating alpha value for full tree
        alpha.append((0, None))
        leaves_for_prune.append(len(leaves))
        miss.append(self.miss_classifications(leaves))       
        if self.problem =="classifier":
            result.append((f"Alpha = {alpha[0][0]}",f"value soglia = {alpha[0][1]}",f"misclassification = {miss[0]}",f"leaves = {leaves_for_prune[0]}"))
        else:
            result.append((f"Alpha = {alpha[0][0]}",f"value soglia = {alpha[0][1]}",f"deviance = {miss[0]}",f"leaves = {leaves_for_prune[0]}"))


        #Running through original prediction for full tree 
        mse = 0
        miss_val = 0
        mse_list =[]
        if self.problem =='regression':
            for i in range(len(y_test)):      #iterates through number of rows in n_feature_test 
                for node in all_node:
                    if node.name =="n1":           

                        new = []
                        new_n = []            
                        for name in self.features_names:
                            new.append(features_test[name][i])
                        for n_name in self.n_features_names:
                            new_n.append(n_features_test[n_name][i])        

                        d = dict(zip(self.features_names, new))
                        dn = dict(zip(self.n_features_names, new_n))
                        d.update(dn)
                        self.pred_x(node, d, all_node, leaves)


                        mse += (y_test[i] - self.prediction_reg[-1])**2
                        #mse_list.append= (y_test[i],self.prediction_reg[-1])
            leaves_mse[leaves_for_prune[-1]] = mse/len(y_test)

            '''
            print("len_ytest",len(y_test), "len_prediction_reg",len(self.prediction_reg))
            comparison = []
            for i in len(y_test):
                comparison.append([y_test[i], self.prediction_reg[i]])
            
            print("c2",comparison)
            '''


        else:
            for i in range(len(y_test)):       
                for node in all_node:
                    if node.name =="n1":           

                        new = []
                        new_n = []            
                        for name in self.features_names:
                            new.append(features_test[name][i])
                        for n_name in self.n_features_names:
                            new_n.append(n_features_test[n_name][i])

                        d = dict(zip(self.features_names, new))
                        dn = dict(zip(self.n_features_names, new_n))
                        d.update(dn)
                        self.pred_x(node, d, all_node, leaves)              

                        if y_test[i] != self.prediction_cat[-1]:
                            miss_val +=1

            leaves_miss[leaves_for_prune[-1]] = miss_val

        pruned_trees =[]
        
        #Start Pruning Process, continuing until root node
        while len(all_node) >=3:
            
            new_dict = self.identify_subtrees(all_node,leaves)
            
            cut = self.alpha_calculator(new_dict)
            
            alpha.append(cut)  #(alpha,node)
            
            if(cut[1])==None:
                break
            all_node = self.pop_list(all_node, lista_to_pop = new_dict[cut[1]][1]) #pop on all node
            
            leaves = self.pop_list(leaves, lista_to_pop = new_dict[cut[1]][0]) #pop on leaf
            leaves.append(cut[1])
            miss.append(self.miss_classifications(leaves))

            leaves_for_prune.append(len(leaves))

            pruned_trees.append([len(leaves), all_node.copy(), leaves.copy()])

            mse = 0
            miss_val = 0
            mse_list =[]
            if self.problem =='regression':
                for i in range(len(y_test)):      #iterates through number of rows in n_feature_test 
                    for node in all_node:
                        if node.name =="n1":           

                            new = []
                            new_n = []            
                            for name in self.features_names:
                                new.append(features_test[name][i])
                            for n_name in self.n_features_names:
                                new_n.append(n_features_test[n_name][i])

                            d = dict(zip(self.features_names, new))
                            dn = dict(zip(self.n_features_names, new_n))
                            d.update(dn)
                            self.pred_x(node, d, all_node, leaves)

                            mse += (y_test[i] - self.prediction_reg[-1])**2
                            #mse_list.append= (y_test[i],self.prediction_reg[-1])
                leaves_mse[leaves_for_prune[-1]] = mse/len(y_test)

            else:
                for i in range(len(y_test)):       
                    for node in all_node:
                        if node.name =="n1":           
                            
                            new = []
                            new_n = []            
                            for name in self.features_names:
                                new.append(features_test[name][i])
                            for n_name in self.n_features_names:
                                new_n.append(n_features_test[n_name][i])

                            d = dict(zip(self.features_names, new))
                            dn = dict(zip(self.n_features_names, new_n))
                            d.update(dn)
                            self.pred_x(node, d, all_node, leaves)                    

                            if y_test[i] != self.prediction_cat[-1]:
                                miss_val +=1
                
                leaves_miss[leaves_for_prune[-1]] = miss_val   
         
        if self.problem =='regression':
            print("{leaves : mean square error} = ", leaves_mse)
            minimum = 100000
            key_min = 100000
            for key in leaves_mse:
                if leaves_mse[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_mse[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with a deviance of: {minimum} ")
            self.graph_results(leaves_for_prune,miss,"Training Set", list(leaves_mse.keys()),list(leaves_mse.values()),"Testing Set")
            
            for i in pruned_trees:
                if i[0] == key_min:

                    self.print_tree(i[1], i[2], "CART_tree_pruned.png","tree_pruned.dot")

        else:
            print("{leaves : misclassification count} = ", leaves_miss)
            minimum = 10000
            key_min = 10000 
            for key in leaves_miss:
                if leaves_miss[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_miss[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with misclassification count {minimum} ")           
            self.graph_results(leaves_for_prune,miss,"Training Set", list(leaves_miss.keys()),list(leaves_miss.values()),"Testing Set")

            for i in pruned_trees:
                if i[0] == key_min:

                    self.print_tree(i[1], i[2], "CART_tree_pruned.png", "tree_pruned.dot")
        
        if self.problem =="classifier":
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"misclassification = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        else:
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"deviance = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        
        if self.problem =="classifier":
            deviance = 0
            for node in leaves:
                c = Counter(self.y[node.indexes]) #Creates a dictionary {"yes":number, "no"}
                c = list(c.items())
                for i in c:

                    #deviance += 2 * i[1] * math.log10(i[1]/) 
                    p = i[1]/len(self.y[node.indexes])
                    deviance += p * math.log2(p)
            #print(f"WANT TO CHECK Deviance for classification problem {-deviance/len(self.y)} {-deviance} {len(self.y)}")
                
        return result
    
    
    def cut_tree(self,how_many_leaves:int):
        if how_many_leaves>len(self.get_leaf())-1:
            print("error on cut")
            exit(1)
        
        all_node = self.get_all_node()
        leaves = self.get_leaf()
        
        alpha=[]  #(alpha,node) lista degli alpha minimi
        
        while len(self.leaf) != how_many_leaves:
               
            new_dict = self.identify_subtrees(all_node,leaves)
            
            cut = self.alpha_calculator(new_dict)
            alpha.append(cut)  #(alpha,node)
            
            if cut[1] == None:
                break
            
            all_node = self.pop_list(all_node, lista_to_pop = new_dict[cut[1]][1]) #pop on all node
            self.all_node  = all_node
            
            leaves = self.pop_list(leaves, lista_to_pop = new_dict[cut[1]][0]) #pop on leaf
            leaves.append(cut[1])
            self.leaf = leaves


    def build_tree_recursively(self,nodenum, parent_node, parent_children, all_node,leaf_list, leaf_dict, graph, parent_node2):
        '''Creates a tree structire, placing the generated nodes from fit() into this required structure for printing'''
        
        for child in parent_children[nodenum]:          #iterating throught the values in the dictionary for the nodenum key
            for node2 in all_node:                      #Iterate through the all node dictionary
                if int(node2.name[1:]) == child:        #Matched the node to that in the dictionary, in order to apply the lines data below, and applyign the corresponding value 
                    if child not in leaf_list:
                        if self.impurity_fn =="gini":
                            child_node = pydot.Node(int(node2.name[1:]), label = f"{node2.split}\n{self.impurity_fn}: {round(self.impur(node2, display = True),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, node2.split])    #creates the new child node, if not a terminal node, to show the split information in "lines"
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))                           
                            
                            child_node2 = Node([str(child),node2.name, node2.split, round(self.impur(node2, display = True),2)], parent=parent_node2, lines =[node2.name, node2.split, round(1-self.impur(node2)/len(node2.indexes),2)])                        
                        else:
                            child_node = pydot.Node(int(node2.name[1:]), label = f"{node2.split}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, node2.split])    #creates the new child node, if not a terminal node, to show the split information in "lines"
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))

                            child_node2 = Node([str(child),node2.name, node2.split, round(self.impur(node2),2)], parent=parent_node2, lines =[node2.name, node2.split, round(self.impur(node2),2)])                        

                    else:                     
                        if self.problem == "classifier":        #For classifier problem
                            count_y = 0
                            response_dict ={}
                            for response in self.y[(self.get_key(leaf_dict,child)).indexes]:        #determing majority in terminal nodes
                                
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1

                            class_node = max(response_dict, key = response_dict.get)
                            if self.impurity_fn =="gini":
                                child_node = pydot.Node(int(node2.name[1:]), label = f"Class: {class_node}\n{self.impurity_fn}: {round(self.impur(node2, display = True),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, class_node]) #creates a new child with th lines set to the class of the node
                                graph.add_node(child_node)
                                graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))
                                
                                child_node2 = Node([str(child),node2.name, class_node, round(self.impur(node2, display = True),2)], parent=parent_node2, lines =[node2.name, class_node, round(1-self.impur(node2)/len(node2.indexes),2)])                            
                            else:
                                child_node = pydot.Node(int(node2.name[1:]), label = f"Class: {class_node}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, class_node]) #creates a new child with th lines set to the class of the node
                                graph.add_node(child_node)
                                graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))

                                child_node2 = Node([str(child),node2.name, class_node, round(self.impur(node2),2)], parent=parent_node2, lines =[node2.name, class_node, round(self.impur(node2),2)])                    

                        else:
                            mean_y = mean(self.y[(self.get_key(leaf_dict,child)).indexes])

                            child_node = pydot.Node(int(node2.name[1:]), label = f"Bin Value: {round(mean_y,2)}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, round(mean_y,2)]) #creates a new child node, when it is a terminal node, so instead present the mean of the y values in the node
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))
                            
                            child_node2 = Node([str(child), node2.name, round(mean_y,2)], parent=parent_node2, lines =[node2.name, round(mean_y,2)]) #creates a new child node, when it is a terminal node, so instead present the mean of the y values in the node

            if child in parent_children:            #Continues the growing only if the child has a key value in parent_children, and therefore has children
                self.build_tree_recursively(child, child_node, parent_children,all_node,leaf_list, leaf_dict, graph, child_node2)


    def print_tree(self, all_node = None,leaf= None, filename="CART_tree.png", treefile = "tree.dot"):
        '''Print a visual representation of the formed tree, showing splits at different branches and the mean of the leaves/ terminal nodes.'''

        if not all_node:
            all_node = self.get_all_node()
        if not leaf:
            leaf = self.get_leaf()
              
        leaf_list =[]
        leaf_dict ={}
        for node in leaf:                           #creates a list of the node numbers and a dictionary connecting nodes with their node numbers
            leaf_list.append(int(node.name[1:]))
            leaf_dict[node] = int(node.name[1:])
        father_list =[]
        for node in all_node:
            father_list.append(int(node.name[1:]))
        
        parent_child =[]                            #list for having child with their parent, for use in dictionary below
        for node in all_node:
            if (int(node.name[1:]) *2) in father_list:
            
                parent_child.append([int(node.name[1:]), int(node.name[1:])*2])
            if (int(node.name[1:])*2+1) in father_list:
            
                parent_child.append([int(node.name[1:]), int(node.name[1:])*2+1])   

        parent_children = {}                        #dictionary for parents with children, only numbers
        for parent, child in parent_child: 
            if parent in parent_children:
                parent_children[parent].append(child)
            else:
                parent_children[parent] = [child]

        node_num = 1                            #The first node
        for node in all_node:
            if node.name =="n1":                #ensuring to start at "n1"
                
                graph = pydot.Dot("my_graph", graph_type="digraph", dir="forward", shape="ellipse", spines = "line")

                if self.impurity_fn =="gini":
                    tree = pydot.Node (int(node.name[1:]),  label =f"{node.split}\n{self.impurity_fn} : {round(self.impur(node, display = True),2)}\nSamples : {len(node.indexes)}" )#, lines =[node.name, node.split])         #creates root node
                    tree2 = Node([str(node_num), node.split, round(self.impur(node, display = True),2)], lines =[node.name, node.split])         #creates root node
                
                else:
                    tree = pydot.Node (int(node.name[1:]),  label =f"{node.split}\n{self.impurity_fn} : {round(self.impur(node),2)}\nSamples : {len(node.indexes)}" )#, lines =[node.name, node.split])         #creates root node
                    tree2 = Node([str(node_num), node.split, round(self.impur(node),2) ], lines =[node.name, node.split])         #creates root node

                graph.add_node(tree)
                self.build_tree_recursively(node_num, tree, parent_children,all_node,leaf_list, leaf_dict, graph, tree2) #starts applying parent and child names to respective instances


        #Old print method
        '''
        for pre, fill, node in RenderTree(tree):                #renders the tree for printing using the RengerTree function from anytree
            print("{}{}".format(pre, node.lines[0]))
            for line in node.lines[1:]:
                print("{}{}".format(fill, line)) 
        '''
        
        #Dot exporter and dot to png

        try:                              
            DotExporter(tree2).to_dotfile(treefile)   #was tree
            graph.write_png(filename) 
        except: 
            DotExporter(tree2).to_dotfile(treefile)


        #igraph Graph
        
        nr_vertices = max(father_list)                              # make too many to allow for missing nodes
        v_label = list(map(str, father_list) )                      # create node labbels 
        G = Graph.Tree(nr_vertices, 2)                              # 2 stands for children number
        lay = G.layout_reingold_tilford(root=[0])

        position = {k: lay[k-1] for k in father_list}               # assigning nodes to positions 

        Y = [lay[k][1] for k in range(len(father_list))]
        M = max(Y)

        es = EdgeSeq(G)                                             # sequence of edges
        E = [e.tuple for e in G.es] # list of edges

        L = len(position)
        Xn = [position[k][0] for k in father_list]
        Yn = [2*M-position[k][1] for k in father_list]

        a = 0
        while a<10:                                                 # When the value is removed it skips to the next index value, jumping, a<10 is just overkill

            for edge in E:
                if edge[0] +1 not in position or edge[1]+1 not in position:
                    E.remove(edge) 
            a+=1

        Xe = []
        Ye = []
        for edge in E:                    
            Xe+=[position[edge[0]+1][0],position[edge[1]+1][0], None]                   # edited for +1 poisiotn as the expected 0 root node it 1 in our dictionary
            Ye+=[2*M-position[edge[0]+1][1],2*M-position[edge[1]+1][1], None]         

        #change labels here, edited to display more information than the node.name
        
        for label in range(len(v_label)):
            for node in all_node:
                if v_label[label] == node.name[1:]:
                    #print(v_label[label], leaf_list, type(leaf_list[0]))
                    if int(v_label[label]) in leaf_list:
                        if self.problem == "classifier":        #For classifier problem
                            count_y = 0
                            response_dict ={}
                            for response in self.y[node.indexes]:        #determing majority in terminal nodes
                                
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            
                            class_node = max(response_dict, key = response_dict.get)
                            if self.impurity_fn == "gini":
                                v_label[label] = f"Class: {class_node}, {self.impurity_fn} : {round(self.impur(node, display = True),2)}, Samples : {len(node.indexes)}" 
                            else:
                                v_label[label] = f"Class: {class_node}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}" 

                        else:
                            mean_y = mean(self.y[node.indexes])
                            v_label[label]=  f"Bin Value: {round(mean_y,2)}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}"
                    else:
                        if self.impurity_fn == "gini":
                            v_label[label] = f"{node.split}, {self.impurity_fn} : {round(self.impur(node, display = True),2)}, Samples : {len(node.indexes)}"
                        else:
                            v_label[label] = f"{node.split}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}"

        labels = v_label

        # Drawing using plotly library 

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
        fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='markers',
                        name='Nodes',
                        marker=dict(symbol='circle-dot',
                                        size=18,
                                        color='#6175c1',    #'#DB4551',
                                        line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                        text=labels,
                        hoverinfo='text',
                        opacity=0.8
                        ))
        fig.update_layout(
            title=filename[:-4],    #chops off ".png"
            )
        fig.show()
  

    def pred_x(self,node, x, all_node, leaves): #-> tree :
        '''Provides a prediction for the y value (based on the mean of the terminal node), for a new set of unsupervised values'''
                
        #all_node = self.get_all_node()
        #leaves = self.get_leaf()
        
        node_list =[]
        node_dict ={}
        for node1 in all_node:                      #Creating dictionaries and lists to move between nodes and node numbers
            node_list.append(int(node1.name[1:]))
            node_dict[node1] = int(node1.name[1:])
        
        if node in leaves:                            #Provides the final output for the predicted node
            if self.problem =="classifier":         #checks if the problem is classification
                for response in self.y[node.indexes]:        #determing majority in terminal nodes
                    response_dict ={}
                    if response in response_dict:
                        response_dict[response] +=1
                    else:
                        response_dict[response] =1
                class_node = max(response_dict, key = response_dict.get)
                self.prediction_cat.append(class_node)
                

            else:
                self.prediction_reg.append(mean(self.y[node.indexes]))
            return node 
        
        else:
            if eval(node.split, x):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                self.pred_x(new_node, x, all_node, leaves) # go to the right child
            else:
                new_node = self.get_key(node_dict, int(node.name[1:])*2)
                self.pred_x(new_node, x, all_node, leaves) # go to the left child
    

    def misclass(self, y):
        
        comparison = []
        if self.problem =="classifier":         #checks if the problem is classification        
            for i in range(len(y)):
                comparison.append([y[i], self.prediction_cat[i]])

            count = 0
            for i in comparison:
                if i[0] != i[1]:
                    count +=1
            print("Misclassification", str(round((count/len(self.prediction_cat))/100,6))+ "%")  
        
        else:
            for i in range(len(y)):
                comparison.append([y[i], self.prediction_reg[i]])

            mse = 0
            for i in comparison:
                mse += (i[0] - i[1])**2
            mse = mse/ len(self.prediction_reg)
            print("Deviance", round(mse,2))              


    def prints(self):
        for i in self.get_leaf():
            print(len(self.y[i.indexes]),Counter(self.y[i.indexes]))


    def graph_results(self, x1, y1,  dataset1, x2, y2, dataset2):
        plt.plot(x1, y1, label = dataset1)
        plt.plot(x2, y2, label = dataset2)

        if self.problem =="regression":

            y_label = 'Deviance'
        else:
            y_label = 'Misclassification'

        plt.xlabel('Leaves')
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs Leaves for Training and Test Set for {self.impurity_fn}")
        
        plt.legend()
        
        plt.axis([max(x1+x2)*1.05, min(x1+x2)*.95, min(y1+y2)*0.95, max(y1+y2)*1.05])


        plt.show()
        return