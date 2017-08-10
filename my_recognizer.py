import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set unknown_word
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set unknown_word
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    # Iterate through each word-id (unknown word) in the Test Set
    num_unknown_words = len(test_set.get_all_Xlengths())
    for unknown_word in range(0, num_unknown_words):
        unknown_word_sequences, unknown_word_sequence_lengths = test_set.get_item_Xlengths(unknown_word)
        word_match_likelihoods = {} # Map of vocabulary word -> log likelihood (log(P(X | Mi)) where i <- 1 .. len(vocabulary)

        # Calculate Log Likelihood logL for each potential word's model and append it to word probabilities
        for word, model in models.items():
            try:
                logL = model.score(unknown_word_sequences, unknown_word_sequence_lengths)
                word_match_likelihoods[word] = logL
            except:
                # Eliminate non-viable models from consideration
                word_match_likelihoods[word] = float("-inf")
                
        assert len(word_match_likelihoods) == len(models.items()), "word probabilities for unknown word ({}) should have been determined for each vocabulary word".format(unknown_word)
        probabilities.append(word_match_likelihoods)
        guesses.append(max(word_match_likelihoods.keys(), key = lambda word: word_match_likelihoods[word]))
        
    assert (len(probabilities)) == num_unknown_words, "Rows of word probability maps (#{}) should match the # of unknown words (#{})".format(len(probabilities), num_unknown_words)
    assert (len(guesses)) == num_unknown_words, "Rows of probabilities maps (#{}) should match the # of unknown words (#{})".format(len(guesses), num_unknown_words)

    return probabilities, guesses
