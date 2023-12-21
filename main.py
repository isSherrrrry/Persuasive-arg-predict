from json.decoder import JSONDecodeError
from openai import OpenAI
import os
import json
import sys
import time
import pandas as pd
import numpy as np

with open('./api_key.txt', 'r') as file:
    apikey = file.read().strip()

client = OpenAI(
    api_key=apikey
)

# calling gpt4
def call_gpt4(prompt):
    model_engine = "gpt-3.5-turbo-1106"
    response = client.chat.completions.create(
        model=model_engine,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# double check the output and retry if not the desired output 
def call_gpt_with_retries_json(user_prompt, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        raw_response = call_gpt4(user_prompt)

        try:
            decoded_response = json.loads(raw_response)

            # Check if 'predicted_scores' is an integer
            if isinstance(decoded_response.get('persuasive_comment'), int):
                return decoded_response
            else:
                print(f"'predicted_scores' is not an integer. Retrying...")
                retry_count += 1

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Problematic JSON string: {raw_response}")
            retry_count += 1

        if retry_count == max_retries:
            print("Max retries reached. Exiting.")
            sys.exit(1)

# find the scores for LLMs' Evaluation 
def find_score(op_title, op_text, comment_0, comment_1):
    # first prompt is direct comparision
    # second prompt is score then preidct 

    # prompt = f"On Reddit, an OP posted a discussion that they wanted their opinions to be challenged titled {op_title}. The OP has given some context to their opinion: {op_text}. The first comment was {comment_0} and the second comment was {comment_1}. Which one do you think the OP finds more persuasive? Please give me your best guess, do not return None. Return in JSON format with the comment number only: {{\"persuasive_comment\": 1or2}}"
    prompt = f"On Reddit, an OP posted a discussion that they wanted their opinions to be challenged titled {op_title}. The OP has given some context to their opinion: {op_text}. The first comment was {comment_0} and the second comment was {comment_1}. Please give a score ranges from 1 to 10 and explanation to each comment in terms of its persuasiveness to the OP. Please give me your best guess, do not return None. Then compare your scores and reasons, which comment do you think is more persuasive? Return in JSON format like: {{\"first_comment_score\": 1-10, \"first_comment_reasons\": \"something\", \"second_comment_score\": 1-10, \"second_comment_reasons\": \"something\", \"persuasive_comment\": 1 or 2}}"
    response = call_gpt_with_retries_json(prompt)
    return response



def find_score_and_features(op_title, op_text, comment_0, comment_1):
    # first prompt no info on the features
    # second prompt has info on the features
    # prompt = f"On Reddit, an OP posted a discussion that they wanted their opinions to be challenged titled {op_title}. The OP has given some context to their opinion: {op_text}. The first comment was {comment_0} and the second comment was {comment_1}. Please give a score ranges from 1 to 10 and explanation to each comment in terms of its persuasiveness to the OP. Please give me your best guess, do not return None. Then compare your scores and reasons, which comment do you think is more persuasive? List the top five features you considered to be the most important in your decision process to choose the most persuasive comment. Please be generic in describing the features like word count, do not include details about the comments or the OP's description. Return in JSON format like: {{\"first_comment_score\": 1-10, \"first_comment_reasons\": \"something\", \"second_comment_score\": 1-10, \"second_comment_reasons\": \"something\", \"persuasive_comment\": 1 or 2, \"features\": [\"one-word or short phrase feature\", \"another feature\"]}}"
    prompt = f"On Reddit, an OP posted a discussion that they wanted their opinions to be challenged titled {op_title}. The OP has given some context to their opinion: {op_text}. The first comment was {comment_0} and the second comment was {comment_1}. Please give a score ranges from 1 to 10 and explanation to each comment in terms of its persuasiveness to the OP. Please give me your best guess, do not return None. Then compare your scores and reasons, which comment do you think is more persuasive? Do you use the following features in your decision making process? Features include the following catgeories: 1. the similarity between OP and the comment (denoated as interplay), 2. The comment itself (denoted as style), including aspects from the number of words, word choices (such as pronouns and articles), feelings associate with the words (clam vs exicted, perceptible vs abstract, weak vs powerful, pleasant vs non-pleasent), entity-related features (such as number of paragraphs, word sentences, number of sentences), and markdown features (such as italics, bullet points, bolds, and numbered words). If so, marked 1 in your returned JSON. Return in JSON format like: {{\"first_comment_score\": 1-10, \"first_comment_reasons\": \"something\", \"second_comment_score\": 1-10, \"second_comment_reasons\": \"something\", \"persuasive_comment\": 1 or 2, \"interplay\": 0 or 1, \"style\": 0 or 1, \"number_of_words\": 0 or 1, \"word_choices\": 0 or 1, \"word_emotions\": 0 or 1, \"entity_related\": 0 or 1, \"markdown\": 0 or 1}}"
    response = call_gpt_with_retries_json(prompt)
    return response


def record_scores(df, path):
    if 'predicted_scores' not in df.columns:
        df['predicted_scores'] = np.nan

    for index, row in df.iterrows():
        if pd.isna(row['predicted_scores']):
            score_data = find_score(row['op_title'], row['op_text'], row['comment_0'], row['comment_1'])
            score = score_data["persuasive_comment"]
            df.at[index, 'predicted_scores'] = score if isinstance(score, int) else 10000
            df.to_csv(path, index=False)
            print(f"Processed row {index+1}")

# find the list of features - 1st method
def record_scores_and_feature(df, path, features_path):
    if 'predicted_scores' not in df.columns:
        df['predicted_scores'] = np.nan

    with open(features_path, 'a') as f:
        for index, row in df.iterrows():
            if pd.isna(row['predicted_scores']):
                score_data = find_score_and_features(row['op_title'], row['op_text'], row['comment_0'], row['comment_1'])
                score = score_data["persuasive_comment"]
                df.at[index, 'predicted_scores'] = score if isinstance(score, int) else 10000

                features = score_data.get('features', [])
                for feature in features:
                    f.write(f"{feature}\n")

                df.to_csv(path, index=False)
                print(f"Processed row {index+1}")

# find the feature usage - 2nd method
def record_scores_and_feature_binary(df, path):
    if 'predicted_scores' not in df.columns:
        df['predicted_scores'] = np.nan

    additional_features = ['interplay', 'style', 'number_of_words', 
                           'word_choices', 'word_emotions', 'entity_related', 'markdown']

    for feature in additional_features:
        if feature not in df.columns:
            df[feature] = np.nan

    for index, row in df.iterrows():
        if pd.isna(row['predicted_scores']):
            score_data = find_score_and_features(row['op_title'], row['op_text'], row['comment_0'], row['comment_1'])

            score = score_data["persuasive_comment"]
            df.at[index, 'predicted_scores'] = score if isinstance(score, int) else 10000

            for feature in additional_features:
                feature_details = score_data[feature]
                df.at[index, feature] = feature_details if isinstance(feature_details, int) else 10000

            df.to_csv(path, index=False)
            print(f"Processed row {index+1}")

if __name__ == "__main__":
    # df = pd.read_csv('./all_predicted.csv')
    # record_scores(df, './all_predicted.csv')

    # df = pd.read_csv('./clean_data/direct_comments_html.csv')
    # # df = pd.read_csv('./direct_predicted.csv')
    # record_scores(df, './direct_predicted.csv')

    # df = pd.read_csv('./results/all_predicted_2.csv')
    # record_scores(df, './results/all_predicted_2.csv')

    # df = pd.read_csv('./clean_data/direct_comments_html.csv')
    # record_scores(df, './results/direct_predicted_2.csv')

    # df = pd.read_csv('./results/feature_1.csv')
    # record_scores_and_feature(df, './results/feature_1.csv', './results/features.txt')

    df = pd.read_csv('./clean_data/all_comments_html.csv')
    record_scores_and_feature_binary(df, './results/feature_2.csv')



    