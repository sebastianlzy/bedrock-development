import json
import boto3
import importlib
import random
from pydash import map_, for_each, get
import time
from tabulate import tabulate
import os
import psycopg2

working_with_embeddings = importlib.import_module("02-working-with-embeddings")
working_with_foundation_model = importlib.import_module("01-working-with-foundation-model")
get_titan_embedding = working_with_embeddings.get_titan_embedding
calculate_euclidean_distance = working_with_embeddings.calculate_euclidean_distance
calculate_distance_between_two_prompt = working_with_embeddings.calculate_distance_between_two_prompt
measure_time_taken = working_with_foundation_model.measure_time_taken
dataset_filepath = 'build/dataset.json'

# main function
bedrock = boto3.client(
    service_name='bedrock-runtime'
)
pg_conn = None


def get_random_item(arr):
    random_number = random.randint(0, len(arr) - 1)
    return arr[random_number]


def generate_embeddings_and_store_in_file():
    names = ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking',
             'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg',
             'Marie Curie', 'Ernest Rutherford', 'Michael Faraday', 'Richard Feynman']
    actions = ['plays basketball', 'teaches physics', 'sells sea shells',
               'collects tax', 'drives buses', 'researches into gravity',
               'manages a shop', 'supervises graduate students',
               'works as a support engineer', 'runs a bank']
    places = ['London', 'Sydney', 'Los Angeles', 'San Francisco', 'Beijing',
              'Cape Town', 'Paris', 'Cairo', 'New Delhi', 'Seoul']
    # create a data file
    count = 100
    with open(dataset_filepath, 'w') as outfile:
        while count > 0:
            text = '{name} {action} in {place}.'.format(
                name=get_random_item(names),
                action=get_random_item(actions),
                place=get_random_item(places)
            )
            embedding = get_titan_embedding(text)
            item = {'id': count, 'text': text, 'embedding': embedding}
            print(".", end="")
            json_object = json.dumps(item)
            outfile.write(json_object + '\n')
            count = count - 1


def load_dataset_from_local(filepath):
    dataset = []
    with open(filepath) as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset


def search_in_local_dataset(input_query):
    dataset = load_dataset_from_local(dataset_filepath)

    def cb(item):
        item["distance"] = calculate_euclidean_distance(
            item["embedding"],
            get_titan_embedding(input_query)
        )

    _, response_for_cal_distance_in_seconds = measure_time_taken(lambda: for_each(dataset, cb))
    _, response_for_sort_distance_in_seconds = measure_time_taken(lambda: dataset.sort(key=lambda x: x['distance']))
    return get(dataset, '0.text')


def get_pg_connection():
    global pg_conn
    if pg_conn is None:
        pg_conn = psycopg2.connect(
            # host=os.environ.get("PG_HOST_NAME"),
            host='localhost',
            port=os.environ.get("PG_HOST_PORT"),
            user=os.environ.get("PG_MASTER_USERNAME"),
            password=os.environ.get("PG_MASTER_PASSWORD"),
            database=os.environ.get("PG_DATABASE")
        )
    return pg_conn


def create_vector_table_in_pg():
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute('CREATE EXTENSION vector')
    cursor.execute('CREATE TABLE dataset (id SERIAL, content TEXT, embedding VECTOR(1536))')
    conn.commit()


def load_data_into_pg():
    conn = get_pg_connection()
    cursor = conn.cursor()
    sql = 'INSERT INTO dataset (content, embedding) VALUES(%s, %s)'
    dataset = load_dataset_from_local(dataset_filepath)
    for_each(dataset, lambda item: cursor.execute(sql, (item['text'], item['embedding'])))
    conn.commit()


def search_in_pg_database(input_query, limit=1):
    conn = get_pg_connection()
    cursor = conn.cursor()

    embedding = str(get_titan_embedding(input_query))
    sql = 'SELECT id, content FROM dataset ORDER BY embedding <-> %s LIMIT %s'
    cursor.execute(sql, (embedding, limit))

    for row in cursor:
        return row[1]


def dataset_setup():
    conn = get_pg_connection()
    generate_embeddings_and_store_in_file()  # One time setup
    create_vector_table_in_pg()  # One time setup
    load_data_into_pg()  # One time setup
    conn.close()


def main():
    input_query_1 = 'Lady Gaga purchased a necklace in Singapore.'
    input_query_2 = 'Taylor Swift flying a plane in Bangkok.'
    input_query_3 = 'Obama driving a car in New York.'

    table = []
    for input_query in [input_query_1, input_query_2, input_query_3]:
        local_response, local_response_in_seconds = measure_time_taken(lambda: search_in_local_dataset(input_query))
        table.append([input_query, local_response, local_response_in_seconds, "N/A"])

        pg_response, pg_response_in_seconds = measure_time_taken(lambda: search_in_pg_database(input_query))
        table.append([input_query, pg_response, "N/A", pg_response_in_seconds])

    print(tabulate(table, headers=["input", "search_result", "local_search (seconds)", "pg_vector (seconds)"]))


if __name__ == "__main__":
    # dataset_setup() #One time setup
    main()
