import logging
from args import ClassifierArgs
from classifier import Classifier


if __name__ == '__main__':
    args = ClassifierArgs()._parse_args()
    print(args)
    logging.info(args)
    Classifier.run(args)