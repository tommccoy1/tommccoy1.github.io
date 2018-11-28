

fi = open("tpr_demo_2.html", "r")
fo = open("tpr_demo_2.html.trimmed.html", "w")

def try_parse_float(s):
  try:
    return float(s)
  except ValueError:
    return False

def num_splitter(word):
    if word == "":
        return word

    #if "." in word and word.index(".") != len(word) - 1:
    #    return word[:word.index(".") + 1] + num_splitter(word[word.index(".") + 1:])

    if word[-1] == ",":
        return num_splitter(word[:-1]) + ","
    if word[-1] == "]":
        return num_splitter(word[:-1]) + "]"
    if word[-1] == ")":
        return num_splitter(word[:-1]) + ")"
    if word[-1] == ";":
        return num_splitter(word[:-1]) + ";"
    if word[0] == "[":
        return "[" + num_splitter(word[1:])
    if word[0] == "(":
        return "(" + num_splitter(word[1:])

    if try_parse_float(word):
        return word[:30]
    else:
        return word

for line in fi:
    words = line.strip().split()

    new_line = []
    for word in words:
        new_line.append(num_splitter(word))

    
    fo.write(" ".join(new_line) + "\n")



