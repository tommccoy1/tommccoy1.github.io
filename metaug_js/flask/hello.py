from flask import Flask, request, render_template
app = Flask(__name__)



import numpy as np


@app.route('/')
def hello_world():
    return render_template('my-form.html', data="empty", preds="empty")


def change_model(new_model):
    global model
    model = new_model

def to_eos(string):
    if "EOS" in string:
        return string[:string.index("EOS")]
    else:
        return string

test_sets = load_dataset("yonc.test")
#for elt in test_sets[0][0]:
#    model = fit_example(model, elt[0], elt[1])

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    output = request.form['output']  

    global model


    outp = to_eos(model([text])[0][0])
    model_new = fit_example(model, text, output)

    change_model(model_new)


    preds = ",".join([to_eos(model([pair[0]])[0][0]) + "*" + to_eos(pair[1]) for pair in test_set])

    #processed_text = model(["text"], "textext......................")[0]
    return render_template('my-form.html', data=outp, preds=preds)

