from nmt_utils import *

_, human_vocab, _, _ = load_dataset(10000)

EXAMPLES = ['3 May 1979']
#, '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:

    source = string_to_int(example, 30, human_vocab)
    print (len(source))
    print (len(human_vocab))
    print (list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    s1 = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    s1 = np.reshape(s1, (1, s1.shape[0], s1.shape[1]))
    print (s1.shape)