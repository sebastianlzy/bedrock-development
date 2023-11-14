import boto3
import json
from pydash import get, map_
from tabulate import tabulate
import os
import time
from prompts import *

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')


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
        'temperature': 0.3,
        'topP': 1.0,
        'stopSequences': [],
        'countPenalty': {'scale': 0},
        'presencePenalty': {'scale': 0},
        'frequencyPenalty': {'scale': 0}
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_amazon_titan_runtime(prompt):
    model_id = os.environ.get("AMAZON_TITAN_ID")
    input_for_model_runtime = {
        'inputText': prompt,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': 0,
            'topP': 1
        }
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_cohere_runtime(prompt):
    model_id = os.environ.get("COHERE_MODEL_ID")
    input_for_model_runtime = {
        'prompt': prompt,
        'max_tokens': 1024,
        'temperature': 0.5,
        'k': 500,
        'p': 1.0,
        'stop_sequences': [],
        'return_likelihoods': 'NONE'
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_claude_runtime(prompt):
    model_id = model_id = os.environ.get("CLAUDE_MODEL_ID")
    input_for_model_runtime = {
        "prompt": f'\n\nHuman: {prompt} \n\nAssistant:',
        "max_tokens_to_sample": 300,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
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


def measure_time_taken(invoke, prompt):
    start_time = time.time()
    response = invoke(prompt)
    time_in_seconds = time.time() - start_time
    print("\n--- %s seconds ---" % time_in_seconds)
    return response, time_in_seconds


def main(prompt):
    print(f'Prompt: {prompt}')

    jurassic_runtime_response, jurassic_response_in_seconds = measure_time_taken(invoke_jurrasic_runtime, prompt)
    pretty_print_runtime_response(
        get(jurassic_runtime_response, 'completions.0.data.text').strip(),
        "Jurassic"
    )

    claude_runtime_response, claude_response_in_seconds = measure_time_taken(invoke_claude_runtime, prompt)
    pretty_print_runtime_response(
        get(claude_runtime_response, 'completion'),
        "ClaudeV2"
    )

    cohere_runtime_response, cohere_response_in_seconds = measure_time_taken(invoke_cohere_runtime, prompt)
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
        "article_summarisation_prompt": article_summarisation_prompt
    }
    for key in prompts:
        jurassic_response_in_seconds, claude_response_in_seconds, cohere_response_in_seconds = main(prompts[key])
        table.append([key, jurassic_response_in_seconds, claude_response_in_seconds, cohere_response_in_seconds])

    print(tabulate(table, headers=["Prompt Type", "Jurassic", "ClaudeV2", "Cohere"]))
    # main(meeting_transcribe_prompt)
    # main(article_summarisation_prompt)
    # list_foundational_models()
    # write_to_file(completion, "01-output.md","build")
