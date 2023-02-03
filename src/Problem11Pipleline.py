import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.svm
import matplotlib.pyplot as plt
import tqdm
import itertools
from stochastic_optimizer import stochastic_optimizer
np.random.seed(42) 

def create_embedding_dict(glove_dim:int=300)->dict:
    """create a dictionary of glove embeddings

    Args:
        glove_dim (int, optional): dimensions of glove. Defaults to 300.

    Returns:
        dict: dictionary of glove embeddings
    """
    embeddings_dict = {}
    dimension_of_glove = glove_dim
    with open(f"glove/glove.6B.{glove_dim}d.txt", 'rb') as f: # if 'r' fails with unicode error, please use 'rb',→
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def clean_word(word:str)->str:
    """clean the word by removing non-alphabetical characters and converting to lowercase

    Args:
        word (str): word to be cleaned

    Returns:
        str: cleaned word
    """

    word=word.lower()
    cleaned_word = ""
    for char in word:
        if char in "abcdefghijklmnopqrstuvwxyz":
            cleaned_word += char
    return cleaned_word

def create_embeddings(embeddings_dict, df,columns=['full_text',"summary","keywords"],embedding_dims=300,normalize="l2")->list:

    embeddings = []
    for column in columns:
        embedding=np.zeros((df.shape[0],embedding_dims))
        for i in range(df.shape[0]):
            text=df[column][i][1:-1]
            #split the summary into words
            if column!="keywords":
                words=text.split(' ')
            else:
                words=text.split(',')
            for word in words:
                word=clean_word(word)
                try:
                    embedding[i,:]+=embeddings_dict[word.encode('ascii')]
                except:
                    pass
            if normalize=="l2":
                embedding[i,:]=embedding[i,:]/np.linalg.norm(embedding[i,:])
            elif normalize=="l1":
                embedding[i,:]=embedding[i,:]/np.sum(np.abs(embedding[i,:]))
            elif normalize=="mean":
                embedding[i,:]=embedding[i,:]/len(words)
        embeddings.append(embedding)
    return embeddings


def fit_model(X,Y,n_folds=5):
    kf=sklearn.model_selection.KFold(n_splits=n_folds,shuffle=True,random_state=42)
    scores=[]
    models=[]
    for (train_index,val_index) in kf.split(X):
        X_train=X[train_index,:]
        Y_train=Y[train_index]
        X_val=X[val_index,:]
        Y_val=Y[val_index]
        model=sklearn.svm.SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(X_train,Y_train)
        score=model.score(X_val,Y_val)
        scores.append(score)
        models.append(model)
    return scores,models

def combined_embeddings(embeddings,weights):
    combined_embeddings=np.zeros((embeddings[0].shape[0],embeddings[0].shape[1]))
    for i in range(len(embeddings)):
        combined_embeddings+=weights[i]*embeddings[i]
    return combined_embeddings

def find_optimal_bisection(embeddings:list,Y:np.ndarray,embeddings_weight_range:tuple=((-1,1),(-1,1),(-1,1)),threshold:float=0.001):
    """find the optimal weight combination for the embeddings"""

    #mutliple dimension binary search
    embeddings_weight_permutations_temp=[]
    weight_space=(embeddings_weight_range[0][1]-embeddings_weight_range[0][0])/4
    for i in range(len(embeddings_weight_range)):
        embeddings_weight_permutations_temp.append((embeddings_weight_range[i][0]+weight_space,embeddings_weight_range[i][1]-weight_space))
    embeddings_weight_permutations=itertools.product(*embeddings_weight_permutations_temp)
    #iter through all the possible weight combinations, remove that sum to 1 and
    #add the last weight which is 1-sum of the other weights
    # embeddings_weight_permutations=[list(x)+[1-sum(x)] for x in embeddings_weight_permutations if 1-sum(x)>=0]
    # #if weight space is smaller than threshold, return the center of the weight range
    if weight_space<threshold:
        return [(embeddings_weight_range[i][0]+embeddings_weight_range[i][1])/2 for i in range(len(embeddings_weight_range))]

    #multiple dimension binary search
    best_score=0
    best_weights=None
    for weights in embeddings_weight_permutations:
        X=combined_embeddings(embeddings,weights)
        scores,models=fit_model(X,Y)
        if np.mean(scores)>best_score:
            best_score=np.mean(scores)
            best_weights=weights
            best_models=models
    
    new_weight_ranges=[]
    for i in range(len(embeddings_weight_range)):
        new_weight_ranges.append((best_weights[i]-weight_space,best_weights[i]+weight_space))
    #recursively call the function
    return find_optimal_bisection(embeddings,Y,embeddings_weight_range=new_weight_ranges,threshold=threshold)

def stochastic_find_optimal(embeddings:list,Y:np.ndarray,optimizer_kwargs,epochs=100):
    print(optimizer_kwargs)
    optimizer=stochastic_optimizer(**optimizer_kwargs)

    def get_score(weights):
        X=combined_embeddings(embeddings,weights)
        scores,models=fit_model(X,Y)
        return np.mean(scores)
    params=[]
    overall_accuracies=[]
    for i in range(epochs):
        losses=[]
        for weights in optimizer.get_params():
            losses.append(-get_score(weights)) #negative score because we want to maximize the score
        optimizer.update(np.array(losses))
        print("Epoch: ",i," Score: ",-np.mean(losses))
        params.append(optimizer.get_params())
        overall_accuracies.append(-np.mean(losses))

    params=np.array(params)
    print(params.shape)
    for j in range(len(embeddings)):
        for i in range(len(params[0])):
            plt.plot(params[:,i,j])
        plt.xlabel("Epoch")
        plt.ylabel("Weight "+str(j))
        plt.savefig("weight_"+str(j)+".png")
        plt.close()
    
    plt.plot(overall_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Overall Accuracy")
    plt.savefig("overall_accuracy.png")

    
    return optimizer.get_average_params()
    


def val(models:list,X_val,Y_val):
    preds=np.zeros(Y_val.shape)
    for model in models:
        preds+=model.predict(X_val)
    preds/=len(models)
    preds=(preds>0.5).astype(int)
    return np.mean(preds==Y_val)

def pipeline(df:pd.DataFrame,columns=['full_text',"summary","keywords"],glove_embedding_dims=300,normalize="l2",optimizer_func=find_optimal_bisection,
    **kwargs)->tuple:
    """The pipeline function to train the model

    Args:
        df (pd.DataFrame): the data dataframe
        columns (list, optional): The columns of the data to train on. Defaults to ['full_text',"summary","keywords"].
        glove_embedding_dims (int, optional): the dimensions of the glove embedding to use. Defaults to 300.
        normalize (str, optional): the normalization function to use. Defaults to "l2".
        optimizer_func (_type_, optional): The optimizer function to use. Defaults to find_optimal_bisection.
        **kwargs: the kwargs for the optimizer function
    Returns:
        tuple: the best weights, the best models, the best score, and the embeddings as a list and unsumed
    """
    embeddings_dict=create_embedding_dict(glove_embedding_dims)
    print("Embeddings dict created")
    embeddings=create_embeddings(embeddings_dict,df,columns=columns,embedding_dims=glove_embedding_dims,normalize=normalize)
    print("Embeddings created")
    Y=(df['root_label']=='sports').to_numpy()
    train_test_split=sklearn.model_selection.train_test_split(*embeddings,Y,test_size=0.2,random_state=42)
    Y_train=train_test_split[-2]
    Y_test=train_test_split[-1]
    embeddings_train=[]
    embeddings_test=[]
    for i in range(len(train_test_split)-2):
        if i%2==0:
            embeddings_train.append(train_test_split[i])
        else:
            embeddings_test.append(train_test_split[i])
    
    #find optimal weights
    print("Finding optimal weights")
    embeddings_weight_range=[]
    for i in range(len(embeddings_train)):
        embeddings_weight_range.append((-1,1))
    if optimizer_func==find_optimal_bisection:
        weights=optimizer_func(embeddings_train,Y_train,embeddings_weight_range=embeddings_weight_range,threshold=threshold)
    else:
        weights=optimizer_func(embeddings_train,Y_train,optimizer_kwargs=kwargs["optimizer_kwargs"],epochs=kwargs["epochs"])
    print("Optimal weights found:",weights)
    #train on these optimal weights
    X_train=combined_embeddings(embeddings_train,weights)
    X_test=combined_embeddings(embeddings_test,weights)
    train_scores,models=fit_model(X_train,Y_train)
    test_accuracy=val(models,X_test,Y_test)
    print("val score:",np.mean(train_scores))
    print("test score:",test_accuracy)

    return test_accuracy,models,weights,embeddings
