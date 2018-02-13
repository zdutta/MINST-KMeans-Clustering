#Zidaan Dutta
import pandas as pd 
import random
import numpy as np
from matplotlib import pyplot as plt

#Design task
class MyKmeans:
	data=[]
	cluster=[]
	clusterIndex=[]
	points=[]
	k=0
	def readData(self,filename):
		self.data = pd.read_csv(filename,header=None)
		self.points=np.array(list(zip(self.data[2].values,self.data[3].values)))


	def cluster(self,iterCount,k,centroids=[]):
		self.k = k
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
		self.cluster = result
		return result

	def avgDistance(self,id,clusterId, clusters):
		clusterArray = clusters[int(clusterId)]
		clusterPoints = [np.array(self.points[j]) for j in clusterArray]
		return np.mean(np.linalg.norm(self.points[id]-clusterPoints[0],axis=1))


	def calculateSC(self,clusters):
		sc = 0
		for i in range(len(self.clusterIndex)):
			clusterId = self.clusterIndex[i]
			#clusterArray = clusters[int(cluster)] #the cluster it belongs to
			#calculating the avg distance to points in same cluster
			#clusterPoints = [np.array(self.points[j]) for j in clusterArray]
			A = self.avgDistance(i,clusterId,clusters)
			#minimum avg distance
			B = min([self.avgDistance(i,j,clusters) for j in range(self.k) if j != clusterId])
			sc = sc + ((B-A)/max(A,B))
		sc = sc/len(self.data)
		print sc

	def plotCluster(self):
		plt.scatter(self.data[2],self.data[3], c=0.5, s=7)
		plt.show()








#km = MyKmeans()
#print km.data
#km.calculateSC(km.cluster(1,10))

#tasks
km = MyKmeans()
km.readData('digits-embedding.csv')
km.cluster(1,10)
km.plotCluster()
