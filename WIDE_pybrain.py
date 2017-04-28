import sys
import argparse
import re
import os
import os.path
from PyQt4.QtGui import *
from PyQt4.Qt import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import TanhLayer,SigmoidLayer, SoftmaxLayer, LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader



def init_brain(learn_data, epochs, TrainerClass=BackpropTrainer):
    if learn_data is None:
        return None
    print ("Building network")
    net = buildNetwork(64 * 64, 2, 62, hiddenclass=TanhLayer)
     #net = buildNetwork(64 * 64, 32 * 32, 8 * 8, 5)
    #net = buildNetwork(64 * 64, 62, hiddenclass=LinearLayer)
    # fill dataset with learn data
    trans = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,\
        'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,\
        'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,\
        'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,\
        't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51, '0': 52, '1': 53, '2': 54, '3': 55,\
        '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61
    }
    ds = ClassificationDataSet(4096, nb_classes=62, class_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',\
                                                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',\
                                                                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',\
                                                                  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',\
                                                                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',\
                                                                  'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',\
                                                                  '8', '9'])
    for inp, out in learn_data:
        ds.appendLinked(inp, [trans[out]])
    ds.calculateStatistics()
    print ("\tNumber of classes in dataset = {0}".format(ds.nClasses))
    print ("\tOutput in dataset is ", ds.getField('target').transpose())
    ds._convertToOneOfMany(bounds=[0, 1])
    print ("\tBut after convert output in dataset is \n", ds.getField('target'))
    trainer = TrainerClass(net, verbose=True)
    trainer.setData(ds)
    print("\tEverything is ready for learning.\nPlease wait, training in progress...")
    trainer.trainUntilConvergence(maxEpochs=epochs)
    print("\tOk. We have trained our network.")
    return net


def loadData(dir_name):
    list_dir = os.listdir(dir_name)
    list_dir.sort()
    list_for_return = []
    print ("Loading data...")
    for filename in list_dir:
        out = [None, None]
        print("Working at {0}".format(dir_name + filename))
        print("\tTrying get letter name.")
        lett = re.search("\w+_(\w)_\d+\.png", dir_name + filename)
        if lett is None:
            print ("\tFilename not matches pattern.")
            continue
        else:
            print("\tFilename matches! Letter is '{0}'. Appending...".format(lett.group(1)))
            out[1] = lett.group(1)
        print("\tTrying get letter picture.")
        out[0] = get_data(dir_name + filename)
        print("\tChecking data size.")
        if len(out[0]) == 64 * 64:
            print("\tSize is ok.")
            list_for_return.append(out)
            print("\tInput data appended. All done!")
        else:
            print("\tData size is wrong. Skipping...")
    return list_for_return


def get_data(png_file):
    img = QImage(64, 64, QImage.Format_RGB32)
    data = []
    if img.load(png_file):
        for x in range(64):
            for y in range(64):
                data.append(qGray(img.pixel(x, y)) / 255.0)
    else:
        print ("img.load({0}) failed!".format(png_file))
    return data


def work_brain(net, inputs):
    rez = net.activate(inputs)
    idx = 0
    data = rez[0]
    for i in range(1, len(rez)):
        if rez[i] > data:
            idx = i
            data = rez[i]
    return (idx, data, rez)


def test_brain(net, test_data):
    for data, right_out in test_data:
        out, rez, output = work_brain(net, data)
        print ("For '{0}' our net said that it is '{1}'. Raw = {2}".format(right_out,\
                                                                           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[out], output))
    pass



def main():
    app = QApplication([])
    p = argparse.ArgumentParser(description='PyBrain example')
    p.add_argument('-l', '--learn-data-dir', default="./learn", help="Path to dir, containing learn data")
    p.add_argument('-t', '--test-data-dir', default="./test", help="Path to dir, containing test data")
    p.add_argument('-e', '--epochs', default="1000", help="Number of epochs for teach, use 0 for learning until convergence")
    args = p.parse_args()
    learn_path = os.path.abspath(args.learn_data_dir) + "/"
    test_path = os.path.abspath(args.test_data_dir) + "/"
    if not os.path.exists(learn_path):
        print("Error: Learn directory not exists!")
        sys.exit(1)
    if not os.path.exists(test_path):
        print("Error: Test directory not exists!")
        sys.exit(1)
    learn_data = loadData(learn_path)
    test_data = loadData(test_path)
    #net = init_brain(learn_data, int(args.epochs), TrainerClass=RPropMinusTrainer)
    net = init_brain(learn_data, int(args.epochs), TrainerClass=BackpropTrainer)
    print ("Now we get working network. Let's try to use it on learn_data.")
    #print("Here comes a tests on learn-data!")
    #test_brain(net, learn_data)
    print("Here comes a tests on test-data!")
    test_brain(net, test_data)
    NetworkWriter.writeToFile(net, 'net_27_2_04_15_39.xml')

    return 0

if __name__ == "__main__":
    sys.exit(main())
