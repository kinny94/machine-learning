from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))


features2 = [[250, 0], [225, 0], [170, 1], [180, 1], [100, 2], [120, 2]]
labels2 = [0, 0, 1, 1, 2, 2]
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(features2, labels2)
print(clf2.predict([[190, 0]]))