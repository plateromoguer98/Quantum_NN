import pandas as pd
import numpy as np
##### DATASET TO TRANSFORM ##### 

path="/home/carlos/red_neuronal_dipolos/miguel/input_neural_network_todo.dat"
##############################

df=pd.read_csv(path,sep=",",header=None)


def normalize(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return(normalized_v)
def module(v):
    return(np.sum(v**2))
def cordenate_change(moment,hidrogin1,hidrogin2,oxigin):
    #first calculate new axis after calculate new coponente
    
    VECTOR1=normalize(hidrogin1-oxigin)
    VECTOR2=normalize(hidrogin2-oxigin)
    if(module(VECTOR1+VECTOR2)!=0):
        x_AXIS=normalize(VECTOR1+VECTOR2)
        z_AXIS=normalize(np.cross(VECTOR1,VECTOR2))
        y_AXIS=normalize(np.cross(z_AXIS,x_AXIS))
        new_moment=[]
        for i in range(len(moment)):
            new_moment_x=np.dot(moment,x_AXIS)
            new_moment_y=np.dot(moment,y_AXIS)
            new_moment_z=np.dot(moment,z_AXIS)
    return(round(new_moment_x,5),round(new_moment_y,5),round(new_moment_z,5))


answer=0
hidrogin1=np.array([76,0,21]);hidrogin2=np.array([18,22,33]);oxigin=np.array([31,2,33])
oxigin=df.iloc[0:2];hidrogin1=df.iloc[2:5];hidrogin2=df.iloc[5:8]
print("000000",type(oxigin))
for i in range (df.shape[0]):    
    oxigin=df.iloc[i,0:3].to_numpy()
    hidrogin1=df.iloc[i,3:6].to_numpy()
    hidrogin2=df.iloc[i,6:9].to_numpy()
    for j in range(9,df.shape[1]-1,3):
        if(round(100*i/df.shape[0],2)!=answer):
            answer=round(100*i/df.shape[0],2)
            print("Loading ...",answer," % ")
        cordinate_to_transform=df.iloc[i,j:j+3].to_numpy()
        df.iloc[i,j],df.iloc[i,j+1],df.iloc[i,j+2]=cordenate_change(cordinate_to_transform,hidrogin1,hidrogin2,oxigin)
#Now we convert to a new file 

df.to_csv("rotation.txt",header=None,index=None,sep=",")
    














