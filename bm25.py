from gensim import corpora
from gensim.summarization import bm25
import json
import numpy as np
from evaluate import get_prf


def run_bm25(docs, query, topk=10):
    """
    gensim 3.8.3
    :param docs: [string, string, string, ...]
    :param query: string
    :return: list of inds of the topk doc
    """
    texts = [doc.split() for doc in docs] # you can do preprocessing as removing stopwords
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)
    query_doc = dictionary.doc2bow(query.split())
    scores = bm25_obj.get_scores(query_doc)
    best_docs_inds = sorted(range(len(scores)), key=lambda i: scores[i])[-topk:]
    return best_docs_inds

def file_len(fname):
    import subprocess
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def load_data(filename="/data/jinning/mesh/allMeSH_2021.json"):
    # load instance
    instances = []
    labelname2id = {}

    total_line = file_len(filename)
    print("Start processing raw data. Total lines: {}".format(total_line))

    with open(filename, encoding="ISO-8859-1") as f:
        for j, line in enumerate(f):

            if j > 500000: break

            if j % 100000 == 0:
                print("[{:.2f}%] {}/{}".format(100.0 * j / total_line, j, total_line), end="\r")

            line = line.strip()
            if "meshMajor" in line:

                # comma
                try:
                    # the last line contain redundant "]}"
                    ins = json.loads(line[:-1] if j != total_line else line[:-2])
                except Exception as e:
                    print("Error in line {}/{}".format(j, total_line))
                    print(line[:-3])
                    raise e

                # ignore empty label
                if len(ins["meshMajor"]):
                    # remove irelevent item
                    for i, lab in enumerate(ins["meshMajor"]):
                        if lab not in labelname2id:
                            labelname2id[lab] = len(labelname2id)
                        ins["meshMajor"][i] = labelname2id[lab]

                    ins["id"] = j
                    ins["text"] = ins.pop("abstractText")
                    ins["label"] = ins.pop("meshMajor")

                    instances.append(ins)

                    # print(line)
                    # print(self.instances[-1])
                    # embed()

        for i, ins in enumerate(instances):
            tmp = np.zeros(len(labelname2id))
            for label_id in ins["label"]:
                tmp[label_id] = 1
            instances[i]["label"] = tmp

    print("Done processing raw data.")
    print("label_dim: {}".format(len(labelname2id)))

    return instances, labelname2id

instances, labelname2id = load_data()
label_dim = len(labelname2id)
val_size = 100
val_data = instances[:val_size]

val_preds = np.zeros((val_size, label_dim))
val_labels = np.array([ins["label"] for ins in val_data])

id2labelname = {value: key for key, value in labelname2id.items()}
docs = [ins["text"] for ins in val_data]
print("Running bm25...")
with open("./res/bm25.log", "w") as fout:
    for index, query in sorted(id2labelname.items()):
        print("Run bm25 on query [{}/{}]: {}".format(index, label_dim, query))
        topk_inds = run_bm25(docs, query, topk=2)
        for topk_ind in topk_inds:
            val_preds[topk_ind][index] = 1
        print(docs[topk_inds[0]])
        if index < 10:
            fout.write("\n")
            fout.write("--------------- [{}/{}] Query: {} ---------------\n".format(index, label_dim, query))
            for ind in topk_inds[:5]:
                fout.write(docs[ind] + "\n")

precision, recall, f1 = get_prf(val_labels, val_preds)
print(precision, recall, f1)
