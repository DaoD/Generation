
def make_next_dataset():
    with open("../data/personachat_processed/processed_train_self_original.txt", "r", encoding="utf-8") as fr:
        with open("../data/personachat_processed/more/train_self_original_next.txt", "w", encoding="utf-8") as fw:
            last_first_ut = ""
            last_line = ""
            for lid, line in enumerate(fr):
                line = line.strip().split("\t")
                context = (line[0] + " ").split(' _eos_ ')[:-1]
                if context[0] == last_first_ut:
                    last_responses = last_line[1].split("|")
                    last_label = int(last_line[2])
                    for idx, resp in enumerate(last_responses):
                        if idx == last_label:
                            fw.write(last_line[0] + "\t" + resp + "\t" + "1\t" + last_line[3] + "\t" + last_line[4] + "\t" + context[-1] + "\n")
                        else:
                            fw.write(last_line[0] + "\t" + resp + "\t" + "0\t" + last_line[3] + "\t" + last_line[4] + "\t" + "NA" + "\n")
                else:
                    if last_line != "":
                        last_responses = last_line[1].split("|")
                        last_label = int(last_line[2])
                        for idx, resp in enumerate(last_responses):
                            if idx == last_label:
                                fw.write(last_line[0] + "\t" + resp + "\t" + "1\t" + last_line[3] + "\t" + last_line[4] + "\t" + "NA" + "\n")
                            else:
                                fw.write(last_line[0] + "\t" + resp + "\t" + "0\t" + last_line[3] + "\t" + last_line[4] + "\t" + "NA" + "\n")
                last_line = line
                last_first_ut = context[0]
                # if lid == 3:
                #     break
            for idx, resp in enumerate(last_responses):
                if idx == last_label:
                    fw.write(last_line[0] + "\t" + resp + "\t" + "1\t" + last_line[3] + "\t" + last_line[4] + "\t" + "NA" + "\n")
                else:
                    fw.write(last_line[0] + "\t" + resp + "\t" + "0\t" + last_line[3] + "\t" + last_line[4] + "\t" + "NA" + "\n") 

make_next_dataset()