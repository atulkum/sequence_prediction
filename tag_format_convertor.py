import os

def bio2bioes(path, input_file, output_file):
    input_file = os.path.join(path, input_file)
    output_file = os.path.join(path, output_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "B-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" S-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" B-"+label_type+"\n")
                    elif "I-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" E-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" I-"+label_type+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
def bioes2bio(path, input_file, output_file):
    input_file = os.path.join(path, input_file)
    output_file = os.path.join(path, output_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "E-" in labels[idx]:
                        fout.write(words[idx]+" I-"+label_type+"\n")
                    elif "S-" in labels[idx]:
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def iob2bio(path, input_file, output_file):
    input_file = os.path.join(path, input_file)
    output_file = os.path.join(path, output_file)

    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "I-" in labels[idx]:
                    label_type = labels[idx].split('-')[-1]
                    if (idx == 0) or (labels[idx-1] == "O") or (label_type != labels[idx-1].split('-')[-1]):
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def choose_label(path, input_file, output_file):
    input_file = os.path.join(path, input_file)
    output_file = os.path.join(path, output_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    with open(output_file,'w') as fout:
        for line in fins:
            if len(line) < 3:
                fout.write(line)
            else:
                pairs = line.strip('\n').split(' ')
                fout.write(pairs[0]+" "+ pairs[-1]+"\n")


if __name__ == "__main__":
    from config import config

    #bio2bioes(config.data_dir, 'eng.testa', 'eng.testa.bioes')
    #bio2bioes(config.data_dir, 'eng.testb', 'eng.testb.bioes')
    #bio2bioes(config.data_dir, 'eng.train', 'eng.train.bioes')

    choose_label(config.data_dir, 'eng.testa', 'eng.testa.bio')
    choose_label(config.data_dir, 'eng.testb', 'eng.testb.bio')
    choose_label(config.data_dir, 'eng.train', 'eng.train.bio')

