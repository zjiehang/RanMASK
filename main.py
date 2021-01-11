from args import ClassifierArgs
from classifier import Classifier

if __name__ == '__main__':
    args = ClassifierArgs.parse(verbose=True)
    Classifier.run(args)