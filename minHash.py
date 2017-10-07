'''
@author: nanthini, harshat
'''
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.sparse import find
import itertools
import numpy.matlib

# Our implementation to read in the csv file ,
# Due to computation time and memory constraints, we only consider the first 50,000 rows of the original csv file ,
# The remainder of the questions are solved with the matrix provided on piazza ,
'''
values = pd.read_csv('ml-20m/ratings.csv', usecols=['userId','movieId','rating'],nrows=50000) ,
values['rating'] = np.where(values['rating']>2.5,1,0) ,
new_values = values.pivot(index='movieId', columns = 'userId', values = 'rating') ,
new_values = new_values.fillna(0) ,
print new_values 
'''

# Jaccard Similarity for users in form of set ,
def my_jaccard(a,b):
    return float(len(a&b))/float(len(a|b))
# Jaccard Similarity for users in form of matrix ,
def jaccard_sim(a,b):
    if float(np.logical_or(a,b).sum())==0:
        return 0
    return float(np.logical_and(a,b).sum())/float(np.logical_or(a,b).sum())

print "Loading the values from ratings_np.npy ..."

values = np.load('data/ratings_np.npy')
rows,cols = values.shape
# Number of Hash Functions ,
n = 1000
# Large Prime number greater than number of rows ,
R = 29063
# Generate a and b for hash functions between  ,
# range of 1 and R so that there are no zeros ,
coeff=random.randint(1,R,size=[n,2])
srt = find(values)
# Sort the users and movies in order of the movies ,
arrSrt = srt[1].argsort()
movies = srt[0][arrSrt[::1]]
users = srt[1][arrSrt[::1]]
#Initialize list of sets for Question 1.3 ,
um = [set() for i in range(cols)] 
''' ,
# Question 1.2 ,
(i)Computing for random 10000 pairs
     (ii)max_10 has the maximum 10 of the random pairs
     (iii)plotting the histogram
'''
print "Generating 10000 random pairs and calculating their similarities... "

# Randomly selecting 10,000 pairs ,
indices = random.randint(0,values.shape[1], size = 20000)
sim = list()
for i in range(0,20000,2):
    # In the small chance that we select the same index, simply compare to next one ,
    if(indices[i]==indices[i+1]):
        indices[i+1]+=1
    sim.append(jaccard_sim(values[:,indices[i]],values[:,indices[i+1]]))
# Method to find the largest 10 similarities explained in report
max_10 = [0 for i in range(10)]
min_10 = 0
for i in range(len(sim)):
    if sim[i]>min_10:
        max_10[max_10.index(min_10)] = sim[i]
        min_10 = min(max_10)
print "The average Jaccard Similarity is: " ,np.asarray(sim).mean()
print "The max 10 Jaccard Similarity values are: "
print np.asarray(max_10)
print "Histogram for the data obtained..."
plt.hist(sim)
plt.title("Histogram")
plt.xlabel("Similarities")
plt.ylabel("Frequency")
plt.show()
'''
# Question 1.3 ,Efficient way to store the values

um: maps the users to a set of movies that they like
'''
u_indexj = 0
# Make a list of sets for the user ,
# That is, each user has a set of the movies which it liked ,
print "Storing the data efficiently in the form 'user:{movies}'"
for i in range(cols):
    if(u_indexj==len(movies)):
        break
    while(users[u_indexj]==i + 1):
        um[i].add(movies[u_indexj])
        u_indexj +=1
        if(u_indexj==len(users)):
            break 
# count the number os users that have actually liked movies ,
count1 = 0
for i in range (cols):
    if (bool(um[i])): 
        count1 +=1  

''' ,
# Question 1.4 , Partioning the signature matrix into bands,
    then into buckets and finally finding candidate pairs
  
h1: signature matrix
hash_bands: buckets for each band 
hash_bands[i]: buckets within a band
similar_pairs: candidate pairs that match the criteria of similarity>=0.65
''' 
# Process to find the signature matrix is described in detail in report 
print "Generating a signature matrix for the dataset..."
bMatrix = np.matlib.repmat(coeff[:,1],rows,1)
aMatrix = coeff[:,0]
mTemp = ([i+1 for i in range (rows)])
mNew = np.asarray(mTemp)
aMatrix.reshape(n,1)
myHash = np.outer(aMatrix,mTemp)
myHashFinal = np.mod(myHash+bMatrix.T,R)
h1 = np.zeros((n,cols))
for i in range(cols):
    indices = um[i]
    if (bool(um[i])):
        h1[:,i] = myHashFinal[:,list(indices)].min(1)

print "Signature matrix generated with shape ",h1.shape
# We choose 100 bands of 10 rows each for 1000 hash functions ,
# Our decision to choose these values is explained in the report ,
band = 100
r = 10
# Choose a very large prime number
#RDash = 112519
RDash = 1299827
final = np.empty([band,cols])
a = random.randint(1,RDash, size = r)
a = a.reshape([r,1])
b = np.asarray(random.randint(1,RDash, size = r))
b = b.reshape([r,1])
b = np.matlib.repmat(b,1,cols)
hash_bands = {}
for i in range(band):
    final[i] = np.sum(np.mod(np.multiply(a,h1[(i*r):(i*r)+r,:])+b,RDash),axis=0)
    for j in range(final[i].shape[0]): 
        if i not in hash_bands:
            hash_bands[i] = {}
        if final[i][j] not in hash_bands[i]:
            hash_bands[i][final[i][j]] = [j]
        else:
            hash_bands[i][final[i][j]].append(j)
similar_pairs = set()
for i in hash_bands:
    for hash_num in hash_bands[i]:
        if len(hash_bands[i][hash_num]) > 1:
            for pair in itertools.combinations(hash_bands[i][hash_num], r=2):
                u,v = pair
                if float(len(um[v].union(um[u])))!=0 and (len(um[v].intersection(um[u])) / float(len(um[v].union(um[u])))>=0.65):
                       similar_pairs.add(pair)
print "The number of similar pairs is: ", len(similar_pairs)

'''
1.5: Given a random point, find the nearest neighbor

test = the random point

maxSim = determines the threshold that has to be satisfied
'''
#test = input("Enter a random query point(userID) whose nearest neighbor should be found: ")
test = random.randint(0,values.shape[0])
neighbors = set()
maxSim = 0.8
print "Checking if ",test," is present in similar pairs computed before"
flag = False
for pair in similar_pairs:
    if test in pair:
        comp = jaccard_sim(values[:,pair[0]],values[:,pair[1]])
        if comp>maxSim:
            maxSim = comp
            maxInd = pair
            flag = True
if flag:
    print "Found the nearest neighbor as user ",maxInd," with similarity ",maxSim
    
if not flag:
    print "Could not find a pair within similar_pairs, using brute force..."
    flag = False
    for j in range(values.shape[1]):
        if j not in similar_pairs and j!=test:
            comp = jaccard_sim(h1[:,test],h1[:,j])
            if comp>maxSim:
                maxSim = comp
                maxInd = j
                flag = True
    if flag:
        print "Found the nearest neighbor as user ",maxInd," with similarity ",maxSim
    if not flag:
        print "The user doesn't have any other user with similarity > ",maxSim
