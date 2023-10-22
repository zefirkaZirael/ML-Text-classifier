from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
# importing all necessary libraries

# Creating stemmer on russian language to simplify words to their basic form
stemmer = SnowballStemmer("russian")


# Function to open train file that takes file path as parameter
def load_train(file_path):
    # Creating dictionary 'data' that have two keys name and category
    data = {'name': [], 'category': []}
    # Open file by data path, clarifying that we want to Read from this file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Creating cycle to go through all lines
        for line in file:
            # Clarifying to look for TAB to separate keys in line
            parts = line.strip().split('\t')
            # Wait until we have two elements in parts by checking length of parts
            if len(parts) == 2:
                # Adding corresponding data into dictionary by keys. First element of parts is name and
                # second is category
                data['name'].append(parts[0])
                data['category'].append(parts[1])
    # Returning filled dictionary
    return data


# Function to open test file that takes file path as parameter
def load_test(file_path):
    # Creating dictionary 'data' that have two keys name and category
    data = {'name': [], 'category': []}
    # Open file by data path, clarifying that we want to Read from this file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Reading lines one by one
        lines = file.readlines()
    # Creating cycle to go through all lines
    for line in lines[1:]:
        # Clarifying to look for ',' symbol to separate elements in line
        parts = line.strip().split(',')
        # Wait until we have three elements parts by checking length of parts since there is two ',' in line
        if len(parts) == 3:
            # Adding corresponding data into dictionary by keys. First element is just  number and name with category is
            # second and third, so we start from element with index 1
            data['name'].append(parts[1])
            data['category'].append(parts[2])
    # Returning filled dictionary
    return data


'''
def preprocess_text(text):                  # Function to make text simpler for processing it 
    text = text.lower()                  # Make all letters same case
    stemmed_words = [stemmer.stem(word) for word in text.split()]              #Divide each word by split space and 
                                                            # then apply stemmer on each word and add into new list
    text = ' '.join(stemmed_words)                        # Save stemmed version of the text in new variable
    text = re.sub(r'\b\d+\b', 'digit', text)                  # Replace all numbers in text with ord digit
    return text                                       # Return changed text
'''


# The function takes a training data dictionary to train with it
def train_model(training_data):
    # Creating machine learning pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),       # Preparing data vectorizing it, so it is easier to computer to process it
        ('clf', SGDClassifier(loss='hinge'))        # Creating model to training
    ])
    # Training model to classify text
    pipeline.fit(training_data['name'], training_data['category'])
    # Returning model
    return pipeline


# Function to test model that takes text for testing and model that will tested
def test_model(testing_data, model):
    # Model predicts category based on input model
    predicted = model.predict(testing_data['name'])
    # Calculation if accuracy by comparing predicted category with actual ones
    accuracy = accuracy_score(testing_data['category'], predicted)
    # Opening file to write results
    with open('result.txt', 'w', encoding='utf-8') as result_file:
        # Skipping first line because first line not containing name
        result_file.write('\n')
        # Loop for writing into file
        for prediction in predicted:
            # Writing into file
            result_file.write(prediction + '\n')
    # Return accuracy
    return accuracy


if __name__ == '__main__':
    # Loading data from train file
    training_data = load_train('train.txt')
    # Loading data from test file
    testing_data = load_test('test.txt')

    # Train the model using the training data
    trained_model = train_model(training_data)

    # Test the model using the testing data
    accuracy = test_model(testing_data, trained_model)
    # Printing accuracy with testing data
    print(f"Model Accuracy on Testing Data: {accuracy * 100:.2f}%")

    # Test the model using the training data
    # accuracy2 = test_model(training_data, trained_model)
    # Printing accuracy with training data
    # print(f"Model Accuracy on Training Data: {accuracy2 * 100:.2f}%")

