#画像ファイルから数字を予測する
import sklearn.datasets
import sklearn.svm
import PIL.Image
import numpy

#画像ファイルを数値リストに変換
def imageToData(filename):
    grayImage = PIL.Image.open(filename).convert("L")
    grayImage = grayImage.resize((8,8),PIL.Image.LANCZOS)
    numImage = numpy.asarray(grayImage, dtype = float)
    numImage = numpy.floor(16 - 16 *(numImage / 256))
    numImage = numImage.flatten()

    return numImage

#数字を予測
def predictDigits(data):
    digits = sklearn.datasets.load_digits()
    clf = sklearn.svm.SVC(gamma = 0.001)
    clf.fit(digits.data, digits.target)
    n = clf.predict([data])
    print("予測=",n)

#画像ファイルを数値リストに変換
data = imageToData("./imgs/2.png")
predictDigits(data)