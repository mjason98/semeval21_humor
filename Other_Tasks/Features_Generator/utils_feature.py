import numpy as np
# import pyfreeling
import os


def Preprocess(texts, type, max_seq_length, indexes):
    print('Freeling Tokenizing ', '0 %\r', end="")
    if "FREELINGDIR" not in os.environ:
        os.environ["FREELINGDIR"] = "/usr/local"
    DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

    pyfreeling.util_init_locale("default")
    LANG = "en"
    op = pyfreeling.maco_options(LANG)
    op.set_data_files("",
                      DATA + "common/punct.dat",
                      DATA + LANG + "/dicc.src",
                      DATA + LANG + "/afixos.dat",
                      "",
                      DATA + LANG + "/locucions.dat",
                      DATA + LANG + "/np.dat",
                      "",  # DATA + LANG + "/quantities.dat",
                      DATA + LANG + "/probabilitats.dat")

    tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
    sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
    mf = pyfreeling.maco(op)
    mf.set_active_options(False, True, True, True,  # select which among created
                          True, True, True, True,  # submodules are to be used.
                          True, True, False, True) # default: all created submodules are used

    tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
    sen = pyfreeling.senses(DATA + LANG + "/senses.dat")

    done = 0
    perc = 0
    top = len(texts)
    cont = 0
    all_tokens = []
    for i in range(len(texts)):

        done += 1
        z = done / top
        z *= 100
        z = int(z)
        if z - perc >= 1:
            perc = z
            print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

        x = texts[i]
        l = tk.tokenize(x)
        ls = sp.split(l)

        ls = mf.analyze(ls)
        ls = tg.analyze(ls)
        ls = sen.analyze(ls)
        
        tokens_a = []
        longer = 0
        for s in ls:
            cont  += 1
            ora = s.get_words()
            tmp = []
            for k in range(len(ora)):

                if ora[k].get_tag() == 'W':
                    tmp.append( 'data' )
                elif ora[k].get_tag() == 'Z':
                    tmp.append( 'number' )
                else:
                    if type == 'lemma':
                        tokens_a.append(ora[k].get_lemma().lower())
                    elif type == 'form':
                        tokens_a.append(ora[k].get_form().lower())

        if len(tokens_a ) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1

        one_token = []
        for j in tokens_a:
            if indexes.get(j) is not None:
                one_token.append(indexes[j])
            else: one_token.append(len(indexes))

        one_token = one_token +[len(indexes)]*(max_seq_length - len(tokens_a))
        all_tokens.append(one_token)

    print('Freeling Tokenizing ok        ')
    return np.array(all_tokens)

def convert_lines(example, max_seq_length ,tokenizer, indexes):
    
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        # print(example[i])
        tokens_a = tokenizer.tokenize(example[i])

        if len(tokens_a ) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1

        one_token = []
        for j in tokens_a:
            if indexes.get(j) is not None:
                one_token.append(indexes[j])
            else: one_token.append(len(indexes))

        one_token = one_token +[len(indexes)]*(max_seq_length - len(tokens_a))
        # print(tokens_a)
        all_tokens.append(one_token)
        
    print(longer)
    return np.array(all_tokens, dtype = np.int32)
