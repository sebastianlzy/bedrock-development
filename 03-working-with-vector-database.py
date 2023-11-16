import json
import boto3
import importlib
import random
from pydash import map_, for_each, get
from tqdm import tqdm
from tabulate import tabulate
import os
import psycopg2
from opensearchpy import OpenSearch, RequestsHttpConnection

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
os_client = None
os_index_name = 'demo-index'


def get_random_item(arr):
    random_number = random.randint(0, len(arr) - 1)
    return arr[random_number]


def generate_embeddings_and_store_in_file(count=100):
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

    with tqdm(total=count) as pbar:
        with open(dataset_filepath, 'w') as outfile:
            while count > 0:
                text = '{name} {action} in {place}.'.format(
                    name=get_random_item(names),
                    action=get_random_item(actions),
                    place=get_random_item(places)
                )
                embedding = get_titan_embedding(text)
                item = {'id': count, 'text': text, 'embedding': embedding}
                json_object = json.dumps(item)
                outfile.write(json_object + '\n')
                count = count - 1

                pbar.update(1)


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
    cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
    cursor.execute('DROP TABLE IF EXISTS dataset')
    cursor.execute('CREATE TABLE dataset (id SERIAL, content TEXT, embedding VECTOR(1536))')
    conn.commit()


def load_data_into_pg(dataset):
    conn = get_pg_connection()
    cursor = conn.cursor()
    sql = 'INSERT INTO dataset (content, embedding) VALUES(%s, %s)'

    with tqdm(total=len(dataset)) as pbar:
        def cb(item):
            cursor.execute(sql, (item['text'], item['embedding']))
            pbar.update(1)

        for_each(dataset, cb)

    cursor.execute('SELECT COUNT(*) from dataset')
    for record in cursor:
        print(f'{record[0]} records loaded')

    conn.commit()


def search_in_pg_database(input_query, limit=1):
    conn = get_pg_connection()
    cursor = conn.cursor()

    embedding = str(get_titan_embedding(input_query))
    sql = 'SELECT id, content FROM dataset ORDER BY embedding <-> %s LIMIT %s'
    cursor.execute(sql, (embedding, limit))

    for row in cursor:
        return row[1]


def get_opensearch_client():
    global os_client
    if os_client is None:
        os_client = OpenSearch(
            hosts=[{'host': os.environ.get('OS_LOCAL_HOST_NAME'), 'port': os.environ.get('OS_LOCAL_PORT')}],
            http_auth=(os.environ.get("OS_MASTER_USERNAME"), os.environ.get("OS_MASTER_PASSWORD")),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=100,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )

        # print(os_client.info())
    return os_client


def create_index_in_os():
    client = get_opensearch_client()

    headers = {'Content-Type': 'application/json'}
    document = {
        'settings': {
            'index.knn': True
        },
        'mappings': {
            'properties': {
                'embedding': {
                    'type': 'knn_vector',
                    'dimension': 1536
                },
                'content': {
                    'type': 'text'
                }
            }
        }
    }
    try:
        del_resp = client.indices.delete(os_index_name)
        print(f'del_resp: {del_resp}')
        create_resp = client.indices.create(os_index_name, body=document, headers=headers)
        print(f'create_resp: {create_resp}')
    except Exception as e:
        print(e)


def load_data_into_index_in_os(dataset=None):
    client = get_opensearch_client()
    headers = {'Content-Type': 'application/json'}

    with tqdm(total=len(dataset)) as pbar:
        def cb(item):
            resp = client.create(
                os_index_name,
                id=item['id'],
                body={'embedding': item['embedding'], 'content': item['text']},
                headers=headers
            )
            pbar.update(1)

        for_each(dataset, cb)


def search_in_opensearch(input_query, limit=1):
    client = get_opensearch_client()
    embedding = get_titan_embedding(input_query)
    headers = {'Content-Type': 'application/json'}

    document = {
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding,
                    'k': limit
                }
            }
        }}
    resp = client.search(body=document, index=os_index_name, headers=headers, size=limit)
    for item in get(resp, 'hits.hits'):
        return get(item, '_source.content')


def dataset_setup(is_local_setup=True, is_rds_setup=True, is_opensearch_setup=True, count=1000):
    # Local file setup
    if is_local_setup:
        print("Setting up local dataset")
        generate_embeddings_and_store_in_file(count)

    # RDS Setup
    if is_rds_setup:
        print("Setting up RDS PG Vector")
        conn = get_pg_connection()
        dataset = load_dataset_from_local(dataset_filepath)
        create_vector_table_in_pg()
        load_data_into_pg(dataset)
        conn.close()

    # Opensearch setup
    if is_opensearch_setup:
        print("Setting up Opensearch KNN vector")
        client = get_opensearch_client()
        dataset = load_dataset_from_local(dataset_filepath)

        create_index_in_os()
        load_data_into_index_in_os(dataset=dataset)
        client.close()


def close_all_connections():
    try:
        get_pg_connection().close()
    except Exception as e:
        pass

    try:
        get_opensearch_client().close()
    except Exception as e:
        pass


def main(is_local_search=True, is_rds_search=True, is_os_search=True):
    input_query_1 = 'Lady Gaga purchased a necklace in Singapore.'
    input_query_2 = 'Taylor Swift flying a plane in Bangkok.'
    input_query_3 = 'Obama driving a car in New York.'

    table = []

    for input_query in tqdm([input_query_1, input_query_2, input_query_3]):

        if is_local_search:
            local_response, local_response_in_seconds = measure_time_taken(lambda: search_in_local_dataset(input_query))
            table.append([input_query, local_response, local_response_in_seconds, "N/A", "N/A"])

        if is_rds_search:
            pg_response, pg_response_in_seconds = measure_time_taken(lambda: search_in_pg_database(input_query))
            table.append([input_query, pg_response, "N/A", pg_response_in_seconds, "N/A"])

        if is_os_search:
            os_response, os_response_in_seconds = measure_time_taken(lambda: search_in_opensearch(input_query))
            table.append([input_query, os_response, "N/A", "N/A", os_response_in_seconds])

    close_all_connections()
    print(tabulate(table, headers=["input", "search_result", "local_search (seconds)", "pg_vector (seconds)",
                                   "os_response (seconds)"],
                   tablefmt="github"))


if __name__ == "__main__":
    # dataset_setup(is_local_setup=False, is_rds_setup=False, is_opensearch_setup=False, count=1000)  # One time setup
    main(is_local_search=False)
