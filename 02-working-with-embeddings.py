import json
import boto3
import os
import math
from tabulate import tabulate
from numpy import dot
from numpy.linalg import norm
from pydash import map_, for_each
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

bedrock = boto3.client(
    service_name='bedrock-runtime'
)


def get_titan_embedding(prompt):
    accept = 'application/json'
    contentType = 'application/json'
    model_id = os.environ.get("AMAZON_TITAN_EMBEDDING_MODEL_ID")

    body = json.dumps({'inputText': prompt})
    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding


def calculate_euclidean_distance(v1, v2):
    distance = math.dist(v1, v2)
    return distance


def calculate_dot_product_similarity(v1, v2):
    similarity = dot(v1, v2)
    return similarity


def calculate_distance_between_two_prompt(prompt1, prompt2):
    prompt1_titan_embeddings = get_titan_embedding(prompt1)
    prompt2_titan_embeddings = get_titan_embedding(prompt2)
    # return calculate_euclidean_distance(prompt1_titan_embeddings, prompt2_titan_embeddings)
    # return calculate_dot_product_similarity(prompt1_titan_embeddings, prompt2_titan_embeddings)
    return calculate_cousin_similarity(prompt1_titan_embeddings, prompt2_titan_embeddings)


def calculate_cousin_similarity(v1, v2):
    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
    return similarity


def distance_calculation():
    prompt1 = "hello"
    prompts = [
        'hi',
        'good day',
        'greetings',
        'how are you',
        'what is your name',
        "let's go shopping",
        'what is general relativity',
        'she sells sea shells on the sea shore'
    ]
    table = []
    for prompt2 in prompts:
        distance = calculate_distance_between_two_prompt(prompt1, prompt2)
        table.append([prompt2, distance])

    print(tabulate(table, headers=["Prompt", f'distance from {prompt1}'], tablefmt="github"))


def get_nearest_item(dataset_with_embeddings, query_input_embedding):
    for item in dataset_with_embeddings:
        item['distance'] = calculate_cousin_similarity(item['embedding'], query_input_embedding)

    dataset_with_embeddings.sort(key=lambda x: x['distance'], reverse=True)
    return dataset_with_embeddings[0]


def search_and_recommend():
    # t1 = """
    # The theory of general relativity says that the observed gravitational effect between masses results from their warping of spacetime. 
    # """  # Albert Einstein
    # t2 = """
    # Quantum mechanics allows the calculation of properties and behaviour of physical systems. It is typically applied to microscopic systems: molecules, atoms and sub-atomic particles. 
    # """  # Albert Einstein
    # t3 = """
    # Wavelet theory is essentially the continuous-time theory that corresponds to dyadic subband transforms — i.e., those where the L (LL) subband is recursively split over and over.
    # """  # Jean Morlet
    # t4 = """
    # Every particle attracts every other particle in the universe with a force that is proportional to the product of their masses and inversely proportional to the square of the distance between their centers.
    # """  # Isaac Newton
    # t5 = """
    # The electromagnetic spectrum is the range of frequencies (the spectrum) of electromagnetic radiation and their respective wavelengths and photon energies. 
    # """  # James Clerk Maxwell

    t1 = "The theory of relativity"
    t2 = "Quantum mechanics"
    t3 = "Wavelet theory"
    t4 = "The law of inertia"
    t5 = "Theory of evolution"

    query_input_1 = "Isaac Newton"
    query_input_2 = "Albert Einstein"
    query_input_3 = "Jean Morlet"
    query_input_4 = "Charles Darwin"
    dataset_with_embeddings = map_([t1, t2, t3, t4, t5], lambda x: {"text": x, "embedding": get_titan_embedding(x)})

    table = []
    for query_input in [query_input_1, query_input_2, query_input_3, query_input_4]:
        result = get_nearest_item(dataset_with_embeddings, get_titan_embedding(query_input))
        table.append([result['text'], query_input])
    print(tabulate(table, tablefmt='github'))


def classify_document():
    student_classes = [
        {'name': 'athletics', 'description': 'all students with a talent in sports'},
        {'name': 'musician', 'description': 'all students with a talent in music'},
        {'name': 'magician', 'description': 'all students with a talent in witch craft'}
    ]

    def cb(x): x["embedding"] = get_titan_embedding(x['description'])

    dataset_with_embeddings = for_each(student_classes, cb)

    query_input_1 = 'Ellison sends a spell to prevent Professor Wang from entering the classroom'
    query_input_2 = 'Steve helped me solve the problem in just a few minutes. Thank you for the great work!'
    query_input_3 = 'It took too long to get a response from your support engineer!'

    table = []
    for query_input in [query_input_1, query_input_2, query_input_3]:
        result = get_nearest_item(dataset_with_embeddings, get_titan_embedding(query_input))
        table.append([result['name'], query_input])
    print(tabulate(table, tablefmt='github'))


def k_means_clustering():
    names = ['Albert Einstein', 'Bob Dylan', 'Elvis Presley',
             'Isaac Newton', 'Michael Jackson', 'Niels Bohr',
             'Taylor Swift', 'Hank Williams', 'Werner Heisenberg',
             'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford']

    embeddings = map_(names, lambda x: get_titan_embedding(x))
    df = pd.DataFrame(data={'names': names, 'embeddings': embeddings})
    matrix = np.vstack(df.embeddings.values)

    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(matrix)

    df['cluster'] = kmeans.labels_
    print(df[['cluster', 'names']])
    plot_embeddings_to_chart(df)


def plot_embeddings_to_chart(df):
    # Reduce number of dimensions from 1536 to 2
    tsne = TSNE(random_state=0, n_iter=1000, perplexity=6)
    tsne_results = tsne.fit_transform(np.array(df['embeddings'].to_list(), dtype=np.float32))
    # Add the results to dataframe as a new column
    df['tsne1'] = tsne_results[:, 0]
    df['tsne2'] = tsne_results[:, 1]

    # Plot the data and annotate the result
    fig, ax = plt.subplots()
    ax.set_title('Embeddings')
    sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', ax=ax)
    for idx, row in df.iterrows():
        ax.text(row['tsne1'], row['tsne2'], row['names'], fontsize=6.5, horizontalalignment='center')

    plt.show()


def find_outliers_by_count(dataset, count):
    embeddings = map_(dataset, lambda x: x['embedding'])
    center = np.mean(embeddings, axis=0)
    for item in dataset:
        item['distance'] = calculate_euclidean_distance(item['embedding'], center)
    # sort the distances in reverse order
    dataset.sort(key=lambda x: x['distance'], reverse=True)
    # return N outliers
    return dataset[0:count]


def find_outliers():
    names = ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking',
             'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg',
             'Marie Curie', 'Ernest Rutherford', 'Michael Faraday',
             'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger',
             'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin']
    dataset = map_(names, lambda name: {'name': name, 'embedding': get_titan_embedding(name)})
    result = find_outliers_by_count(dataset, 2)
    for_each(result, lambda item: print(item['name']))


def main():
    # distance_calculation()
    # search_and_recommend()
    # classify_document()
    # k_means_clustering()
    find_outliers()


if __name__ == "__main__":
    main()
