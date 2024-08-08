from datasets import load_dataset, Dataset, Features, ClassLabel, Value
from llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer
import pandas as pd
import csv
import numpy as np
import pyarrow as pa


# data = []
# with open('Lease Model Troubleshooting Guide - Query.csv', mode ='r')as file:
#   df = csv.reader(file, delimiter=',', quotechar='"', quoting=1, doublequote=True)
#   for lines in df:
#       lines = [line.replace("", "None") if len(line)==0 else line for line in lines]
#       data.append(lines)

# data_numpy = np.array(data, dtype=str)
# df = pd.DataFrame(data_numpy, dtype=data_numpy.dtype)
# df.to_csv("QA_Dataset.csv", header=False, index=False)
# df = pd.read_csv("F:\AugustRoboticsLLM\dataset\QA_Dataset.csv", sep=',', na_values=['No solution'], quoting=1, quotechar=r'"', dtype=str, on_bad_lines='skip', engine='python', doublequote=True)
# df = df.fillna('No solution')


class_names = ["Symptom", "Solution1", "Solution2", "Solution3", "Solution4", "Solution5", "Solution6", "Solution7"]
features = Features({name: Value('string') for name in class_names})
# QandA_dataset = Dataset.from_pandas(df, features=emotion_features)
QandA_dataset = load_dataset("csv", data_dir="F:\AugustRoboticsLLM\dataset", sep=',', quoting=1, quotechar=r'"', doublequote=True, split="train", features=features)
QandA_dataset = QandA_dataset.train_test_split(test_size=0.1)
QandA_dataset_sample = QandA_dataset["train"].shuffle(seed=42).select(range(5))



def prepare_dataset(dataset, tokenizer):
    # prompting eng: https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
    formatter = ChatFormat(tokenizer=tokenizer)
    solutions = []
    for col in ['Solution1', 'Solution2', 'Solution3', 'Solution4', 'Solution5', 'Solution6', 'Solution7']:
        if dataset[col] != None:
            print(dataset[col])
            solutions.append(dataset[col])

    solutions = "\n".join(solutions)
    dialogs = [
        [{"role": "system", "content": "Your role is an on-site robotics operation engineer who gives technical and practical advice to client who use Lionel robot to draw lines on the ground."
                                       "You belong to August Robotics Ltd. Please do not answer any question not relate to our business, you can simply refuse it by saying no."
                                       "Let's think through this carefully, step by step."}],
        [{"role": "user", "content": f"{dataset['Symptom']}"}],
        [{"role": "assistant", "content": "Here are the possible solutions provided by me:"
                                          f" {solutions}"}],
    ]
    prompt_tokens = [
        formatter.encode_dialog_prompt(dialog) for dialog in dialogs
    ]
    return prompt_tokens