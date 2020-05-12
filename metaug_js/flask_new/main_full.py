from flask import Flask, render_template 
#import torch


import numpy as np
from random import shuffle

# Load a list of abstract language descriptors
def load_languages(language_file):
    fi = open(language_file, "r")
    lang_list = []

    for line in fi:
        parts = line.strip().split("\t")

        ranking = [int(x) for x in parts[0].split(",")]
        vowel_inventory = parts[1].split(",")
        consonant_inventory = parts[2].split(",")

        lang = [ranking, vowel_inventory, consonant_inventory]

        lang_list.append(lang)

    return lang_list

# Load the file input/output correspondences
def load_io(io_file):
    fi = open(io_file, "r")

    io_correspondences = {}

    for line in fi:
        parts = line.strip().split("\t")
        ranking = tuple([int(x) for x in parts[0].split(",")])

        value = parts[1]
        value_groups = value.split("&")

        value_list = []

        for group in value_groups:
            components = group.split("#")
            inp = components[0]
            outp = components[1]
            steps = components[2].split(",")

            value_list.append([inp, outp, steps])

        io_correspondences[ranking] = value_list

    return io_correspondences

# Load a language that is just Cs and Vs
def load_dataset(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        dev_set = [elt.split(",") for elt in parts[1].split()]
        test_set = [elt.split(",") for elt in parts[2].split()]
        vocab = parts[3].split()
        key_string = parts[4].split(",")

        v_list = key_string[0].split()
        c_list = key_string[1].split()
        ranking = [int(x) for x in key_string[2].split()]

        key = [v_list, c_list, ranking]

        langs.append([train_set, dev_set, test_set, vocab, key])

    return langs



# Load a language that is just Cs and Vs
def load_dataset_scramble(dataset_file):
    fi = open(dataset_file, "r")

    all_train_sets = []
    all_dev_sets = []
    all_test_sets = []

    n_tasks = 0

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        dev_set = [elt.split(",") for elt in parts[1].split()]
        test_set = [elt.split(",") for elt in parts[2].split()]
        all_train_sets += train_set
        all_dev_sets += dev_set
        all_test_sets += test_set

        vocab = parts[3].split()

        n_tasks += 1

    shuffle(all_train_sets)
    shuffle(all_dev_sets)
    shuffle(all_test_sets)

    train_len = len(train_set)
    dev_len = len(dev_set)
    test_len = len(test_set)


    for i in range(n_tasks):
        train_set = all_train_sets[i*train_len:(i+1)*train_len]
        dev_set = all_dev_sets[i*dev_len:(i+1)*dev_len]
        test_set = all_test_sets[i*test_len:(i+1)*test_len]

        v_list = "scrambled"
        c_list = "scrambled"
        ranking = "scrambled"

        key = [v_list, c_list, ranking]

        langs.append([train_set, dev_set, test_set, vocab, key])

    return langs




# Load a language that is just Cs and Vs
def load_dataset_cv(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        test_set = [elt.split(",") for elt in parts[1].split()]
        vocab = parts[2].split()

        langs.append([train_set, test_set, vocab])

    return langs


# Break a list into batches of the desired size
def batchify_list(lst, batch_size=100):
    batches = []
    this_batch_in = []
    this_batch_out = []

    for index, elt in enumerate(lst):
        #print(elt)
        this_batch_in.append(elt[0])
        this_batch_out.append(elt[1])

        if (index + 1) % batch_size == 0:
            batches.append([this_batch_in, this_batch_out])
            this_batch_in = []
            this_batch_out = []

    if this_batch_in != []:
        batches.append([this_batch_in, this_batch_out])

    return batches

# Trim the excess from the end of an output string
def process_output(output):
    if "EOS" in output:
        return output[:output.index("EOS")]
    else:
        return output
import random
from random import shuffle
from collections import OrderedDict


# Redefine a basic PyTorch model to allow
# for double gradients and manual modification
# of weights
class ModifiableModule():
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


    def load_state_dict(self, sdict, same_var=False):
        for name in sdict:
            param = sdict[name]
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)

            self.set_param(name, param)

    def state_dict(self):
        return OrderedDict(self.named_params())

# Redefined linear layer
class GradLinear(ModifiableModule):
    def __init__(self, inp_size, outp_size):
        super(GradLinear, self).__init__()
        self.weights = np.random.rand(outp_size, inp_size)
        self.bias = np.random.rand(outp_size)

    def forward(self, x):
        
        return np.matmul(self.weights,x) + self.bias

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def logsoftmax(x):
    return np.log(softmax(x))

# Redefined LSTM
class GradLSTM(ModifiableModule):
    def __init__(self, input_size, hidden_size):
        super(GradLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wi_bias = np.random.rand(hidden_size)
        self.wf_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wf_bias = np.random.rand(hidden_size)
        self.wg_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wg_bias = np.random.rand(hidden_size)
        self.wo_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wo_bias = np.random.rand(hidden_size)


    def forward(self, inp, hidden):
        hx, cx = hidden

        input_plus_hidden = np.concatenate([inp.flatten(), hx.flatten()])

        i_tpre = np.matmul(self.wi_weights,input_plus_hidden) + self.wi_bias
        i_t = sigmoid(i_tpre)
        f_tpre = np.matmul(self.wf_weights,input_plus_hidden) + self.wf_bias
        f_t = sigmoid(f_tpre)
        g_tpre = np.matmul(self.wg_weights,input_plus_hidden) + self.wg_bias
        g_t = tanh(g_tpre)
        o_tpre = np.matmul(self.wo_weights,input_plus_hidden) + self.wo_bias
        o_t = sigmoid(o_tpre)
        #print(i_t)
        #print(f_t)
        #print(g_t)
        #print(o_t)

        cx = f_t * cx + i_t * g_t
        hx = o_t * tanh(cx)

        #myhook = input_plus_hidden.register_hook(print_grad)

        return hx, (hx, cx), o_tpre, input_plus_hidden, i_tpre, f_tpre, g_tpre


    def named_leaves(self):
        return [('wi_weights', self.wi_weights), ('wi_bias', self.wi_bias),
                ('wf_weights', self.wf_weights), ('wf_bias', self.wf_bias),
                ('wg_weights', self.wg_weights), ('wg_bias', self.wg_bias),
                ('wo_weights', self.wo_weights), ('wo_bias', self.wo_bias)]


# Redefined embedding layer
class GradEmbedding(ModifiableModule):
    def __init__(self, vocab_size, emb_size):
        super(GradEmbedding, self).__init__()
        self.weights = np.random.rand(emb_size, vocab_size)


    def forward(self, x):
        return np.matmul(self.weights,x)

    def named_leaves(self):
        return [('weights', self.weights)]

def onehot(ind):
    oh = np.zeros(34)
    oh[ind] = 1.0
    
    return oh

onehot(6)

# Encoder/decoder model
class EncoderDecoder(ModifiableModule):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = GradEmbedding(vocab_size, input_size)
        self.enc_lstm = GradLSTM(input_size, hidden_size)

        self.dec_lstm = GradLSTM(input_size, hidden_size)
        self.dec_output = GradLinear(hidden_size, vocab_size)

        self.max_length = 20

        self.set_dicts("a e i o u A E I O U b c d f g h j k l m n p q r s t v w x z .".split())


    def forward(self, inp, outp_length=20):
        # Initialize the hidden and cell states
        hidden = (np.zeros([1,self.hidden_size]), np.zeros([1,self.hidden_size]))

        this_seq = []
        # Iterate over the sequence
        for elt in inp:
            ind = self.char2ind[elt]
            this_seq.append(ind)

        inp_length = len(inp)
        if inp_length > 0:

            # Pass the sequences through the encoder, one character at a time
            for index, elt in enumerate(this_seq):
                # Embed the character
                emb = self.embedding.forward(onehot(elt))

                # Pass through the LSTM
                output, hidden_new, _, _, i_tpre, f_tpre, g_tpre = self.enc_lstm.forward(emb, hidden)
                hidden_prev = hidden


                hidden = hidden_new

        encoding = hidden
        # Decoding

        # Previous output characters (used as input for the following time step)
        prev_output = "SOS"

        # Accumulates the output sequences
        out_string = ""



        # Probabilities at each output position (used for computing the loss)
        logits = []
        preds = []
        hiddens = []
        ots = []
        iphs = []
        hidden_prev = hidden
        its = []
        fts = []
        gts = []



        for i in range(min(self.max_length,outp_length)):
            # Determine the previous output character for each element
            # of the batch; to be used as the input for this time step

            # Embed the previous outputs
            emb = self.embedding.forward(onehot(self.char2ind[prev_output]))

            # Pass through the decoder
            output, hidden, o_t, iph, i_tpre, f_tpre, g_tpre = self.dec_lstm.forward(emb, hidden)
            #myhook = o_t.register_hook(print_grad)

            # Determine the output probabilities used to make predictions
            pred = self.dec_output.forward(output.flatten())
            probs = logsoftmax(pred)
            logits.append(probs)
            preds.append(pred)
            hiddens.append(hidden)
            ots.append(o_t)
            iphs.append(iph)
            its.append(i_tpre)
            fts.append(f_tpre)
            gts.append(g_tpre)

            # Discretize the output labels (via argmax) for generating an output character
            label = np.argmax(probs)

            char = self.ind2char[label]
            out_string += char
            prev_output = char


        return out_string, logits, encoding, preds, hiddens, ots, iphs, hidden_prev, its, fts, gts

    def named_submodules(self):
        return [('embedding', self.embedding), ('enc_lstm', self.enc_lstm),
                ('dec_lstm', self.dec_lstm), ('dec_output', self.dec_output)]

    # Create a copy of the model
    def create_copy(self, same_var=False):
        new_model = EncoderDecoder(self.vocab_size, self.input_size, self.hidden_size)
        new_model.copy(self, same_var=same_var)

        return new_model

    def set_dicts(self, vocab_list):
        vocab_list = ["NULL", "SOS", "EOS"] + vocab_list

        index = 0
        char2ind = {}
        ind2char = {}

        for elt in vocab_list:
            char2ind[elt] = index
            ind2char[index] = elt
            index += 1

        self.char2ind = char2ind
        self.ind2char = ind2char
encdec = EncoderDecoder(34,10,256)

encdec.enc_lstm.wo_weights = np.loadtxt("enc_lstm.wo_weights")
encdec.enc_lstm.wi_weights = np.loadtxt("enc_lstm.wi_weights")
encdec.enc_lstm.wg_weights = np.loadtxt("enc_lstm.wg_weights")
encdec.enc_lstm.wf_weights = np.loadtxt("enc_lstm.wf_weights")
encdec.enc_lstm.wo_bias = np.loadtxt("enc_lstm.wo_bias")
encdec.enc_lstm.wi_bias = np.loadtxt("enc_lstm.wi_bias")
encdec.enc_lstm.wg_bias = np.loadtxt("enc_lstm.wg_bias")
encdec.enc_lstm.wf_bias = np.loadtxt("enc_lstm.wf_bias")

encdec.dec_lstm.wo_weights = np.loadtxt("dec_lstm.wo_weights")
encdec.dec_lstm.wi_weights = np.loadtxt("dec_lstm.wi_weights")
encdec.dec_lstm.wg_weights = np.loadtxt("dec_lstm.wg_weights")
encdec.dec_lstm.wf_weights = np.loadtxt("dec_lstm.wf_weights")
encdec.dec_lstm.wo_bias = np.loadtxt("dec_lstm.wo_bias")
encdec.dec_lstm.wi_bias = np.loadtxt("dec_lstm.wi_bias")
encdec.dec_lstm.wg_bias = np.loadtxt("dec_lstm.wg_bias")
encdec.dec_lstm.wf_bias = np.loadtxt("dec_lstm.wf_bias")

encdec.embedding.weights = np.loadtxt("embedding.weights").transpose()
encdec.dec_output.weights = np.loadtxt("dec_output.weights")
encdec.dec_output.bias = np.loadtxt("dec_output.bias")





app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html", data=encdec.forward("do")[0])

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
