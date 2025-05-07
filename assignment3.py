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
    # Replace the line below with your code.

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

    # Create outer dictionary
    outer_dict = {}
    
    # Begin with the minimum amount of context words
    prev_words = tuple(sequence[0:n-1])
    curr_word = None

    # Loop through the sequence
    for word in sequence:
        # Set the current word
        curr_word = word

        # Skip if the current word is in previous words
        if curr_word in prev_words:
            continue

        # Check if the previous word is in the outer dictionary
        if prev_words in outer_dict:
            # Get the inner dictionary (value of the previous word)
            inner_dict = outer_dict[prev_words]

            # Check if the current word is in the inner dictionary
            if curr_word in inner_dict:
                # Increment that value
                inner_dict[curr_word] += 1
            else:
                # Add the current word to the inner dictionary
                inner_dict[curr_word] = 1
        else:
            # Add the previous word to the outer dictionary and make the value an empty dictionary
            outer_dict[prev_words] = {}
            inner_dict = outer_dict[prev_words]

            # Add the current word to the empty dictionary
            inner_dict[curr_word] = 1

        # Add the current word to the previous words tuple and remove the first value
        previous_words_list = []

        first_value = True
        for word in prev_words:
            if first_value == True:
                first_value = False
                continue
        
            previous_words_list.append(word)
        
        previous_words_list.append(curr_word)

        prev_words = tuple(previous_words_list)

    # Return the outer dictionary
    return outer_dict




def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    #raise NotImplementedError

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
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    model = build_unigram(sequence[:20])
    #print(model)

    # Task 1.2 test code
    model = build_bigram(sequence[:20])
    #print(model)

    # Task 1.3 test code
    model = build_n_gram(sequence[:20], 5)
    #print(model)

    # Task 2 test code
    print(query_n_gram(model, tuple(sequence[:4])))

    # Task 3 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''
