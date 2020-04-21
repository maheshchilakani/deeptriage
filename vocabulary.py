class Vocabulary:
    def __init__(self, name):
        self.name = name 
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "<pad>", self.sos_token : "<sos>", self.eosd_token:"<eos>"}
        self.num_words = 3 #during init, we just have 3 words in the vocab

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)

    def trim(self, min_count):
        keep_words = []
        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {: .4f}'.format(len(keep_words), len(self.word2count), len(keep_words)/len(self.word2count)))
        #for the purpose of the work, we are destroying the current dict and resetting it with words we want to keep.
        #Note that the words, with reps are kept in keep_word list so we can read off of it.
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "<pad>", self.sos_token : "<sos>", self.eosd_token:"<eos>"}
        self.num_words = 3 #during init, we just have 3 words in the vocab

        for word in keep_words:
            self.addWord(word)





