from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)

def Log_Reg():
    model= LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, "Logistic Regression"
    
def Support():
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    for kernel in kernels:
        svm = SVC(kernel=kernel, gamma='scale', degree=3, C=1)
        svm.fit(x_train, y_train)
        y_pred_svc = svm.predict(x_test)
        acc_1 = accuracy_score(y_test, y_pred_svc)
        results[kernel] = acc_1
        
    best_kernel = max(results, key=results.get)
    return results[best_kernel], best_kernel

def Neighbours():
    Knn = KNeighborsClassifier(n_neighbors=5)
    Knn.fit(x_train, y_train)
    y_pred_knn = Knn.predict(x_test)
    acc_2 = accuracy_score(y_test, y_pred_knn)

    return acc_2, "KNN"
    
def DT():
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=33
    )
    clf.fit(x_train, y_train)
    y_pred_dt = clf.predict(x_test)
    acc_3 = accuracy_score(y_test, y_pred_dt)

    return acc_3, "Decision Tree"

def Solve():
    results = []
    results.append(("Logistic Regression",Log_Reg()))
    results.append(("SVM", Support()))
    results.append(("KNN", Neighbours()))
    results.append(("Decision Tree", DT()))

    best = max(results, key=lambda x: x[1][0])
    model_name = best[0]
    accuracy = best[1][0]
    detail = best[1][1]

    print("\nBest Model:", model_name)
    print("Accuracy:", accuracy)

    if model_name == "SVM":
        print("Best Kernel:", detail)

Solve()