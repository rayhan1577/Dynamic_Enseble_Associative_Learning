x=[]
x.append([26, 25, 8, 28, 29, 26, 28, 13, 33, 2, 2, 6, 2, 30, 12, 19, 25, 21, 13, 1, 4, 14, 22, 14, 4, 10, 5, 16, 16, 10])
x.append([18, 31, 3, 16, 19, 11, 3, 18, 16, 8, 7, 31, 9, 18, 31, 33, 7, 28, 19, 3, 0, 25, 27, 7, 11, 6, 21, 31, 20, 18])
x.append([18, 34, 27, 25, 23, 3, 10, 17, 7, 31, 14, 5, 6, 31, 0, 13, 14, 19, 4, 10, 15, 28, 15, 4, 34, 12, 4, 16, 27, 18])
x.append([21, 30, 16, 5, 20, 4, 17, 33, 27, 34, 33, 20, 3, 31, 33, 13, 7, 24, 18, 25, 5, 29, 12, 29, 20, 13, 19, 32, 21, 23])

mat=[]
for i in x:
	lis=[]
	for j in x:
		lis.append(len(list(set(i).intersection(j)))/25)
	mat.append(lis)


for i in mat:
	print(i)