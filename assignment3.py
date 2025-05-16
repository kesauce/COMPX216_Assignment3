import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.

    # Create the outer dictionary
    outer_dict = {}

    # Create the inner dictionary
    inner_dict = {}

    # Loop through the sequence
    for word in sequence:
        # Check if word is in the inner dictionary
        if word in inner_dict:
            # Increment the count
            inner_dict[word] += 1
        else:
            # Add it to the inner dictionary
            inner_dict[word] = 1

    # Add the inner dictionary to the outer dictionary
    outer_dict[()] = inner_dict

    # Return the outer dictionary
    return outer_dict

def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.

    # Create the outer and inner dictionary
    outer_dict = {}

    # Store the current and previous word
    prev_word = ()
    curr_word = None

    # Loop through the sequence
    for word in sequence:
        # Set the current word
        curr_word = word

        # Check if the previous word is empty then skip
        if prev_word == ():
            prev_word = (curr_word,)
            continue

        # Check if the previous word is in the outer dictionary
        if prev_word in outer_dict:
            # Get the inner dictionary (value of the previous word)
            inner_dict = outer_dict[prev_word]

            # Check if the current word is in the inner dictionary
            if curr_word in inner_dict:
                # Increment that value
                inner_dict[curr_word] += 1
            else:
                # Add the current word to the inner dictionary
                inner_dict[curr_word] = 1
        else:
            # Add the previous word to the outer dictionary and make the value an empty dictionary
            outer_dict[prev_word] = {}
            inner_dict = outer_dict[prev_word]

            # Add the current word to the empty dictionary
            inner_dict[curr_word] = 1

        # Set previous word to current word in a tuple
        prev_word = (curr_word, )

    # Return the outer dictionary
    return outer_dict


def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.

    # Define outer dictionary
    outer_dictionary = {}

    # Loop through the sequence
    for i in range(len(sequence) - (n + 1)):

        # Grab the previous words and the current word
        previous_words = tuple(sequence[i:i + n - 1])
        current_word = sequence[i + n - 1]

        # Check if the previous words in the outer dictionary
        if previous_words not in outer_dictionary:
            # Add the previous words to outer dictionary
            outer_dictionary[previous_words] = {}

            # Initialise the inner dictionary and set it to 0
            outer_dictionary[previous_words][current_word] = 0

        # Check if previous words is already in the inner dictionary
        if current_word not in outer_dictionary[previous_words]:

            # Initialise the inner dictionary and set it to 0
            outer_dictionary[previous_words][current_word] = 0

        # Increment the value
        outer_dictionary[previous_words][current_word] += 1

    # Return the outer dictonary    
    return outer_dictionary




def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.

    # Check if the sequence is in the model
    if sequence in model:
        # Return the inner dictionary if so
        return model[sequence]
    else:
        return None

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    
    # Make a list to store each prediction from each model
    predictions_list = []

    # Loop through each of the models
    for model in models:
        # Loop through the keys of the dictionary and store in a list the length of each key
        key_list = []
        for key in model:
            key_list.append(len(key))

        # Find the min length in the key list
        if min(key_list) == 0:
            n_gram = 1
        elif min(key_list) == 1:
            n_gram = 2
        else:
            n_gram = min(key_list) + 1

        # Check whether the sequence is initialising
        if len(sequence) == 0:
            predictions_list.append(query_n_gram(model, ()))
        
        # Check whether the sequence has enough context for this specific model
        elif len(sequence) >= n_gram - 1:
            # Only grab the context from the sequence that's needed for the model
            if (n_gram == 1):
                context = ()
            else:
                context = tuple(sequence[-(n_gram - 1):])

            # Ensure that the context exists in the dictionary
            if query_n_gram(model, context) is not None:
                predictions_list.append(query_n_gram(model, context))
        else:
            continue

    # If prediction list is empty then return none
    if not predictions_list:
        return None

    # Grab the dictionary of blended probabilities
    blended_predictions = blended_probabilities(predictions_list)

    # Grab the list of keys and list of values
    blended_keys_list = list(blended_predictions.keys())
    blended_values_list = list(blended_predictions.values())

    #Randomly pick a word based on probability
    random_word = random.choices(blended_keys_list, weights = blended_values_list, k = 1)[0]
    return random_word


def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.

    # Initialise the probability list
    joint_probability_list = []

    # Initialise the log likelihood
    log_likelihood = 0.0

    # Loop through the words in the sequence
    for i in range(len(sequence)):
        # Get the current word
        curr_word = sequence[i]

        # If the sequence is empty
        if i == 0:
            # Grab the unigram
            model = models[-1]

            # Grab the previous words in a tuple
            prev_words = ()

            # Find the inner dictionary of the context words
            dictionary = model[()]

        # If the sequence has a context that is less than the amount of models available
        elif i < len(models):
            # Grab the model needed
            model_index = -(i + 1)
            model = models[model_index]

            # Grab the previous words in a tuple
            prev_words = tuple(sequence[:i])


            # Find the inner dictionary of the context words
            dictionary = query_n_gram(model, prev_words)
    
        else:
            # Grab the 10 gram
            model = models[0]

            # Grab the previous words in a tuple
            prev_words = tuple(sequence[i - (len(models) - 1):i])

            # Find the inner dictionary of the context words
            dictionary = query_n_gram(model, prev_words)
        
        # Ensuring dictionary is not none
        if dictionary is not None:
            # Find the total frequency of the context occuring
            sum_frequency = 0.0
            for value in dictionary.values():
                sum_frequency += float(value)

            # Ensures the current word is in the dictionary
            if curr_word in dictionary:
                # Grab the frequency of the current word, assuming previous words as context then log it
                curr_word_frequency = dictionary[curr_word]
                prob_of_curr_word = float(curr_word_frequency/sum_frequency)
                joint_probability_list.append(prob_of_curr_word)
            else:
                return -math.inf
        else:
            return -math.inf

    # Log each value in the joint probability list and add it to the log likelihood
    for value in joint_probability_list:
        logged_value = math.log(value)
        log_likelihood += logged_value

    return log_likelihood


def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.

    # Initialise the log likelihood
    log_likelihood = 0.0

    # Loop through the words sequence
    for i in range(len(sequence)):
        # Find the current word
        curr_word = sequence[i]

        # Initialise a predictions list
        predictions_list = []

        # Loop through the models
        for model in models:
            # Loop through the keys of the dictionary and store in a list the length of each key
            key_list = []
            for key in model:
                key_list.append(len(key))

            # Find the min length in the key list
            if min(key_list) == 0:
                n_gram = 1
            elif min(key_list) == 1:
                n_gram = 2
            else:
                n_gram = min(key_list) + 1

            # Check whether the sequence is initialising
            if i == 0:
                predictions_list.append(query_n_gram(model, ()))
            
            # Check whether the sequence has enough context for this specific model
            elif i >= (n_gram - 1):
                # Only grab the context from the sequence that's needed for the model
                context = tuple(sequence[i - (n_gram - 1):i])

                # Ensure that the context exists in the dictionary
                if query_n_gram(model, context) is not None:
                    predictions_list.append(query_n_gram(model, context))
            else:
                continue

        # Ensure that prediction list is not empty
        if not predictions_list:
            return -math.inf
        
        # Blend each of the predictions
        blended_predictions = blended_probabilities(predictions_list)

        if curr_word in blended_predictions:
            curr_word_probability = blended_predictions[curr_word]
            log_blended_predictions = math.log(curr_word_probability)

            # Add it to the log
            log_likelihood += log_blended_predictions
        else:
            return -math.inf
            
        
    # Return
    return log_likelihood


if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    model = build_unigram(sequence[:20])
    print(model)

    # Task 1.2 test code
    model = build_bigram(sequence[:20])
    print(model)

    # Task 1.3 test code
    model = build_n_gram(sequence[:20], 5)
    print(model)

    # Task 2 test code
    print(query_n_gram(model, tuple(sequence[:4])))

    # Task 3 test code
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()

    # Task 4.1 test code

    print(log_likelihood_ramp_up(sequence[:20], models))
    

    # Task 4.2 test code
    print(log_likelihood_blended(sequence[:20], models))
