#Zidaan Dutta
import pandas as pd 
import random
import numpy as np
from matplotlib import pyplot as pyplot

#Design task
class MyKmeans:
	data=[]
	clusterIndex=[]
	points=[]
	def readData(self,filename):
		self.data = pd.read_csv(filename,header=None)
		self.points=np.array(list(zip(self.data[2].values,self.data[3].values)))


	def cluster(self,iterCount,k,centroids=[]):
		C=[] #holds the xy format of centroids
		#picking centroids
		if(centroids):
			x=[]
			y=[]
			for i in centroids:
				x.append(self.data[2].values[i])
				y.append(self.data[3].values[i])
				#C.append([self.data[2].values[i],self.data[3].values[i]])
			C=np.array(list(zip(x,y)))
		else:
			randomIndex = random.sample(range(len(self.data)),k)
			x=[]
			y=[]
			for i in randomIndex:
				x.append(self.data[2].values[i])
				y.append(self.data[3].values[i])
			C=np.array(list(zip(x,y)))

		#main algorithm
		clusters=np.zeros(len(self.data))
		for i in range(iterCount):
			for j in range(len(self.points)):
				distances=np.linalg.norm(self.points[j]-C,axis=1)
				index = np.argmin(distances)
				clusters[j] = index
			for j in range(k):
				cluster=[]
				for x in range(len(self.points)):
					if(clusters[x] == j):
						cluster.append(self.points[x])
						C[j] = np.mean(cluster,axis=0)
		self.clusterIndex=clusters
		result=[0]*k
		for i in range(k):
			result[i] = []
			matches = np.where(clusters==i)
			for match in matches:
				result[i].append(self.data[0].values[match])
		#print result
		return result











km = MyKmeans()
km.readData('digits-embedding.csv')
#print km.data
km.calculateSC(km.cluster(1,10))
#km.cluster(10,10)

