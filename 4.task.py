import src_ud2 as ud2
import json


udpipe_model_path = "./conll17/udpipe_model/"
udpipe_parse_path = "./conll17/udpipe_parse/"


def path_dir(path):
    """str -> str: valid dir path"""
    return path if path.endswith("/") else path + "/"


if '__main__' == __name__:
    from sys import argv
    assert 2 == len(argv)
    task_path = path_dir(argv[1])
    
    with open(task_path + "metadata.json") as file:
        metadata = json.load(file)

    with open("./5.udpipe.sh", 'w') as file:
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
