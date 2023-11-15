import boto3
import json
from pydash import get, map_
from tabulate import tabulate
import os
import time
from prompts import *

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500

def list_foundational_models():
    print("=================== list_foundational_models ===================")
    response = bedrock.list_foundation_models()
    models = response['modelSummaries']
    table = []
    for model in models:
        table.append([model.get('modelName'), model.get('modelId'), model.get('responseStreamingSupported')])
    print(tabulate(table, headers=["modelName", "modelId", "isResponseStreamingSupported"]))


def invoke_runtime_model(model_id, runtime_input, accept='application/json'):
    contentType = 'application/json'

    body = json.dumps(runtime_input)
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body


def invoke_jurrasic_runtime(prompt):
    model_id = os.environ.get("JURASSIC_MODEL_ID")
    input_for_model_runtime = {
        'prompt': prompt,
        'maxTokens': 1024,
        'temperature': temperature,
        'topP': top_p,
        'stopSequences': [],
        'countPenalty': {'scale': 0},
        'presencePenalty': {'scale': 0},
        'frequencyPenalty': {'scale': 0}
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_amazon_titan_runtime(prompt):
    model_id = os.environ.get("AMAZON_TITAN_MODEL_ID")
    input_for_model_runtime = {
        'inputText': prompt,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': temperature,
            'topP': top_p
        }
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_cohere_runtime(prompt):
    model_id = os.environ.get("COHERE_MODEL_ID")
    input_for_model_runtime = {
        'prompt': prompt,
        'max_tokens': 1024,
        'temperature': temperature,
        'k': top_k,
        'p': top_p,
        'stop_sequences': [],
        'return_likelihoods': 'NONE'
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_claude_runtime(prompt):
    model_id = model_id = os.environ.get("CLAUDE_MODEL_ID")
    input_for_model_runtime = {
        "prompt": f'\n\nHuman: {prompt} \n\nAssistant:',
        "max_tokens_to_sample": 300,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": [
            "\n\nHuman:"
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def pretty_print_runtime_response(completion, prefix="Ans"):
    print(f'{prefix}: {completion}')
    return completion


def write_to_file(output, filename, output_file_path="."):
    f = open(f'{output_file_path}/{filename}', "w")
    f.write(output)
    f.close()


def measure_time_taken(cb):
    start_time = time.time()
    response = cb()
    time_in_seconds = time.time() - start_time
    return response, time_in_seconds


def main(prompt):
    print(f'Prompt: {prompt}')

    jurassic_runtime_response, jurassic_response_in_seconds = measure_time_taken(lambda: invoke_jurrasic_runtime(prompt))
    pretty_print_runtime_response(
        get(jurassic_runtime_response, 'completions.0.data.text').strip(),
        "Jurassic"
    )

    claude_runtime_response, claude_response_in_seconds = measure_time_taken(lambda: invoke_claude_runtime(prompt))
    pretty_print_runtime_response(
        get(claude_runtime_response, 'completion'),
        "ClaudeV2"
    )

    cohere_runtime_response, cohere_response_in_seconds = measure_time_taken(lambda: invoke_cohere_runtime(prompt))
    pretty_print_runtime_response(
        " ".join(map_(get(cohere_runtime_response, 'generations'), "text")),
        "Cohere"
    )

    return jurassic_response_in_seconds, claude_response_in_seconds, cohere_response_in_seconds


if __name__ == "__main__":
    table = []

    prompts = {
        "simple_prompt": simple_prompt,
        "meeting_transcribe_prompt": meeting_transcribe_prompt,
        "article_summarisation_prompt": article_summarisation_prompt,
        "content_generation_prompt": content_generation_prompt,
        "create_a_table_of_product_description_prompt": create_a_table_of_product_description_prompt,
        "extract_topics_and_sentiment_from_reviews": extract_topics_and_sentiment_from_reviews,
        "generate_product_descriptions_prompt": generate_product_descriptions_prompt,
        "information_extraction_prompt": information_extraction_prompt,
        "multiple_choice_classification": multiple_choice_classification,
        "outline_generation_prompt": outline_generation_prompt,
        "question_and_answer_prompt": question_and_answer_prompt,
        "remove_pii_prompt": remove_pii_prompt,
        "summarise_the_key_takeaways": summarise_the_key_takeaways,
        "write_an_article": write_an_article,
        "write_a_promo_doc": write_a_promo_doc,
        "code_generation_prompt": code_generation_prompt,
        "receipe_generation_prompt": receipe_generation_prompt,
        "zero_shot_prompt": zero_shot_prompt,
        "few_shot_prompt": few_shot_prompt,
        # "chain_of_thoughts_prompt": chain_of_thoughts_prompt,
    }
    for key in prompts:
        jurassic_response_in_seconds, claude_response_in_seconds, cohere_response_in_seconds = main(prompts[key])
        table.append([key, jurassic_response_in_seconds, claude_response_in_seconds, cohere_response_in_seconds])

    print(tabulate(table, headers=["Prompt Type", "Jurassic Ultra", "ClaudeV2", "Cohere Text V14"], tablefmt="github"))
    # main(meeting_transcribe_prompt)
    # main(article_summarisation_prompt)
    # list_foundational_models()
    # write_to_file(completion, "01-output.md","build")
