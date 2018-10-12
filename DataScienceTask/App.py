from flask import Flask
import Classifier as classifier
app=Flask(__name__)

@app.route('/')
def classifierfunc():
    return str(classifier.getAccuracy())

if(__name__=='__main__'):
    app.run()