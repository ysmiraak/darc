import src_ud2 as ud2
import json


task_path = "/media/training-datasets/universal-dependency-learning/conll17-ud-trial-2017-03-19/"
# "/media/training-datasets/universal-dependency-learning/conll17-ud-development-2017-03-19/"

darc_path = "/home/darc/darc/"
udpipe_model_path = darc_path + "conll17/udpipe_model/"
udpipe_parse_path = darc_path + "conll17/udpipe_parse/"


if '__main__' == __name__:
    
    with open(task_path + "metadata.json") as file:
        metadata = json.load(file)

    with open(darc_path + "5.udpipe.sh", 'w') as file:
        for task in metadata:
            if task['ltcode'] in ud2.treebanks:
                file.write("udpipe --input horizontal --tokenize --tag --outfile {} {} {}\n"
                           .format(udpipe_parse_path + task['outfile'],
                                   udpipe_model_path + task['ltcode'] + ".udpipe",
                                   task_path + task['rawfile']))
            else:
                file.write("cp {} {}\n"
                           .format(task_path + task['psegmorfile'],
                                   udpipe_parse_path + task['outfile']))
