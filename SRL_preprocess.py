import json
import os

from allennlp.predictors import Predictor


def get_predictor():
    return Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


print("---------")
print(os.getcwd())
print("---------")

raw_data_path = "/preprocessed/openbookqa/openbookqa-test-processed-questions.jsonl"

preprocessed_data_path = "/preprocessed/openbookqa/openbookqa-test-srl-processed-questions.jsonl"

target_dir = os.path.dirname(preprocessed_data_path)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

predictor = get_predictor()

with open(preprocessed_data_path, "w") as write_file:
    # 1.load raw data
    with open(raw_data_path, 'r') as entailment_file:

        # print("enter 3")
        for line in entailment_file:

            if line.strip():
                instances_json = json.loads(line.strip())
                premises = instances_json["premises"]
                raw_question = instances_json["raw_question"]
                question_id = instances_json["question_id"]
                hypotheses = instances_json["hypotheses"]
                entailments = instances_json.get("entailments", None)

            # 2.process each hypothesis
            count = 0
            hypothesesReplace = []

            for hypothesis in hypotheses:
                count += 1
                hypothesisReplace = ""

                print()
                print("------------------------------")
                print('Number of processed hypothesis: ', count)
                print('hypothesis:', hypothesis)
                print()
                print('prediction:\n', predictor.predict(hypothesis))
                try:
                    hypothesisReplace = predictor.predict(hypothesis)['verbs'][0]['description']
                except:
                    hypothesisReplace = hypothesis
                    print("Same as original.Thrown.")
                else:
                    print('result:\n', hypothesisReplace)
                    print("------------------------------")
                    hypothesesReplace.append(hypothesisReplace)

            instance = {}
            instance["premises"] = premises
            instance["raw_question"] = raw_question
            instance["question_id"] = question_id
            instance["hypotheses"] = hypothesesReplace
            instance["entailments"] = entailments

            write_file.write(json.dumps(instance) + "\n")
