import math
import statistics
import warnings
import logging
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import traceback

"""
This file represents a set of model selectors.

A selector does not actually 'select' from a list of existing model objects. Rather
it creates a population of model objects based on different numbers of states
and then returns the one that is the 'best'. In that sense it 'selects' a model,
but selection here might be confused to mean that it picks from an existing pool
of model objects, which it doesn't. The selection process includes instantiating
different models, and comparing them using some criteria. This criteria for
comparison is what is subclassed from the ModelSelector base class below.

As for the models themselves, one might ask, what is the population of models being
considered and where does the diversity come from? Well, the models may differ
in the hyper parameters chosen. In the case of the HMM models (which are the focus
of this exercise), the only parameter that varies from one model to the next is
the 'number-of-states' in the model.

This file will implement 4 selectors:
    - SelectorConstant: A stub to return a model with the hard-coded
        number of states
    - SelectorBIC: Selection of the model with the lowest Bayesian Information
        Criterion score
    - SelectorDIC: Selection of the model with the lowest Discriminative
        Information Criterion score
    - SelectorCV: Selection of the model with the highest log likelihood it
        achieves through cross-validation on the training data. 

"""

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 default_num_states=3,
                 min_num_states=2, max_num_states=10,
                 random_seed=14, verbose=False):
        self.this_word = this_word
        self.other_words_sequences_and_lengths = {word: seqs_and_lens for word, seqs_and_lens in all_word_Xlengths.items() if word is not this_word}
        self.all_word_measurements = all_word_sequences
        self.all_word_sequences_and_lengths = all_word_Xlengths
        self.this_word_measurements = all_word_sequences[this_word]
        if len(self.this_word_measurements) == 0:
            raise RuntimeError("No measurements exist for {}:".format(self.this_word) )
        self.X, self.this_word_sequence_lengths = all_word_Xlengths[this_word]
        self.num_features = len(self.X[0])
        self.default_num_states = default_num_states
        self.min_num_states = min_num_states
        self.max_num_states = max_num_states
        self.random_seed = random_seed
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def __create_model__(self, num_states, measurements, measurement_lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, 
                                    covariance_type="diag", 
                                    n_iter=1000,
                                    random_state=self.random_seed, 
                                    verbose=False)\
                        .fit(measurements, measurement_lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
                print("failure on {} with {} states".format(self.this_word, num_states))
            raise

class SelectorConstant(ModelSelector):
    """ select the model with value self.default_num_states

    """

    def select(self):
        """ select based on default_num_states value

        :return: GaussianHMM object
        """
        best_num_components = self.default_num_states
        return self.__create_model__(best_num_components, self.X, self.this_word_sequence_lengths)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    Model selector that uses the Bayesian Information Criterion, calculated as:
        BIC = -2 * logL + p * logN

    where L is the log-likelihood of the fitted model, p is the number of parameters,
    and N is the number of data points (features). The term −2 log L decreases with
    increasing model complexity (more parameters), whereas the penalty p log N 
    increases with increasing complexity. The BIC applies a larger penalty
    when N > e^2 (N > 7.4). The lower the BIC value the better the model.
    [Note: BIC values can only be compared to other BIC values]
    
    From [2]:
        p = n*(n-1) + (n-1) + 2*d*n
          = n^2 + 2*d*n - 1
    Where d is the number of features.    
    
    Sources:
    [1] http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    [2] https://discussions.udacity.com/t/understanding-better-model-selection/232987/9
    """
    @staticmethod
    def __get_bic_score__(num_states, log_likelihood, num_features):
        num_trainable_params = ( num_states ** 2 ) + ( 2 * num_states * num_features ) - 1
        bic_score = (-2 * log_likelihood) + (num_trainable_params * np.log(num_features))
        return bic_score
    
    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_num_states and self.max_num_states
        
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model_to_bic_tuples = [] # This is a dictionary of { model: complexity score (bic) }
        for num_states in range(self.min_num_states, self.max_num_states + 1):
            try:
                model = self.__create_model__(num_states, self.X, self.this_word_sequence_lengths)
                logL = model.score(self.X, self.this_word_sequence_lengths)
                bic_score = SelectorBIC.__get_bic_score__(num_states, logL, self.num_features)
                model_to_bic_tuples.append((model, bic_score))
            except Exception as e:
                if self.verbose:
                    print (e)
                    traceback.print_exc()
                    print("failure on {} with {} states. continuing...".format(self.this_word, num_states))
        best_model = min(model_to_bic_tuples, key = lambda item: item[1])[0] if model_to_bic_tuples else None
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    Sources:
    http://www.stat.missouri.edu/~dsun/8640/Model_selection.pdf
    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model_to_dic_tuples = [] # list of (model, dic_score) tuples
        for num_states in range(self.min_num_states, self.max_num_states + 1):
            try:
                model = self.__create_model__(num_states, self.X, self.this_word_sequence_lengths)
                logL = model.score(self.X, self.this_word_sequence_lengths)
                
                # Score the same model with all the other words' data:
                anti_logLs = [] # There's no need to keep track of words by idx. We collect whatever we succeed in scoring
                for other, seqs_lens in self.other_words_sequences_and_lengths.items():
                    assert len(seqs_lens) == 2, "strnage: seq_lens should have been a tuple. but was size {}".format(len(seqs_lens))
                    try:
                        logL = model.score(*seqs_lens)
                        anti_logLs.append(logL)
                    except Exception as e:
                        if self.verbose:
                            print (e)
                            traceback.print_exc()
                            print("failure on {} with {} states. but continuing ...".format(other, num_states))

                assert len(anti_logLs) == len(self.other_words_sequences_and_lengths)

                anti_logL_mean = np.mean(anti_logLs)
                dic_score = logL - anti_logL_mean
                model_to_dic_tuples.append((model, dic_score))
            except Exception as e:
                if self.verbose:
                    print (e)
                    traceback.print_exc()
                    print("failure on {} with {} states. but continuing ...".format(self.this_word, num_states))

        # Pick the model with the highest DIC score:
        best_model = max(model_to_dic_tuples, key = lambda x: x[1])[0] if model_to_dic_tuples else None
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    Sources:
    [1] https://discussions.udacity.com/t/understanding-better-model-selection/232987/12
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        fold_splitter = KFold(n_splits = 3, shuffle = False, random_state = None)
        model_and_score_tuples = []
        for num_states in range(self.min_num_states, self.max_num_states + 1):
            log_likelihoods = []
            try:
                # Check sufficient data to split using KFold
                if len(self.this_word_measurements) > 2:
                    for train_indices, test_indices in fold_splitter.split(self.this_word_measurements):
                        print("TRAIN indices:", train_indices, "TEST indices:", test_indices)
                        train_sequences, train_sequence_lengths = combine_sequences(train_indices, self.this_word_measurements)
                        test_sequences, test_sequence_lengths = combine_sequences(test_indices, self.this_word_measurements)

                        model = self.__create_model__(num_states, train_sequences, train_sequence_lengths)
                        logL = model.score(test_sequences, test_sequence_lengths)
                else:
                    model = self.__create_model__(num_states, self.X, self.this_word_sequence_lengths)
                    logL = model.score(self.X, self.this_word_sequence_lengths)

                log_likelihoods.append(logL)

                # Find average Log Likelihood of CV fold
                avg_logL = np.mean(log_likelihoods)
                model_and_score_tuples.append(tuple([model, avg_logL]))
            except Exception as e:
                if self.verbose:
                    print (e)
                    traceback.print_exc()
                    print("failure on {} with {} states".format(self.this_word, num_states))
        best_model = max(model_and_score_tuples, key = lambda x: x[1])[0] if model_and_score_tuples else None
        return best_model