
from markovify.text import Text
from multiprocessing import Pool, cpu_count
import math



#############

import argparse
import json
import markovify
import os
from tqdm import tqdm
import re
    
class NoWordException(Exception): pass

class TextFromComboList(markovify.Text):

    def sentence_split(self, text):
        # return List[List], list of chars in list of words, words[chars]
        output = []
        for line in text.split("\n"):
            name, word = self.process_line(line)
            if name and word:
                if filter_username(name, pattern_list=FILTER_PATTERN):
                    output.append(word)
        if len(output) > 0:
            return output
        else:
            raise NoWordException("No word found in list using pattern :", FILTER_PATTERN)
        
    def process_line(self, line):

        splitted = line.split(":")[:2]
        if len(splitted) > 1:
            name, word = splitted
            name = "".join(x for x in name if x.isascii())
            if len(name) > 0 and len(word) > 0:
                return name, word
            else:
                return None, None
        else:
            return None, None

def get_filelist(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            matches.append(os.path.join(root, filename))
    return matches

def save_model(output_path, model):
    with open(output_path, "w") as f:
        json.dump(model.to_dict(), f)

def filter_username(text, pattern_list=None):

    if pattern_list:
        pattern_list = pattern_list.split(",")
        matches = []
        for pattern in pattern_list:
            match = re.findall(pattern, text)
            if len(match) > 0:
                matches.append(match)
        if len(matches) > 0:
            return text
    else:
        return text

def format_batch(filelist, batch_size=8):
    """Read files from filelist and returns batch of str + updated filelist"""

    if len(filelist) >= batch_size:
        batch_filelist = [filelist.pop(0) for i in range(batch_size)]
    else:
        batch_filelist = [filelist.pop(0) for i in range(len(filelist))]
    
    texts = []
    for path in batch_filelist:
        with open(path) as f:
            text = f.read()
            texts.append(text)

    return filelist, texts, batch_filelist

def train_model(text):
    try:
        model = TextFromComboList(text, retain_original=False)
    except NoWordException as e:
        return None
    return model

def main_loop(args):
    """"Run training over all files in parallel"""

    if not args.n_cpu:
        n_cpu = cpu_count()
    else:
        n_cpu = args.n_cpu
    
    # If multiple cpu, keep 1 free
    if args.keep_free_cpu:
        if n_cpu > 1:
            n_cpu = n_cpu - 1

    filelist = get_filelist(args.input_dir)

    n_files = len(filelist)
    print(f"Found {n_files} files.")
    print("Batch size :", args.batch_size)

    if args.restore_path:
        with open(args.restore_path) as f:
            model_json = f.read()
        combined_model = TextFromComboList.from_json(model_json)
        last_step = int(os.path.basename(args.restore_path).split("_")[-1].split(".")[0])
        print("Restoring from :", last_step)
    else:
        combined_model = None
        last_step = 0

    #model = None
    last_step = 0
    
    n_batches = math.floor(len(filelist) / args.batch_size)
    n_batches = (n_batches + 1) if len(filelist) % args.batch_size > 0 else n_batches

    with Pool(n_cpu - 1) as pool:
        with tqdm(total=n_batches) as pbar:
            while len(filelist) > 0:
                filelist, texts, batch = format_batch(
                    filelist,
                    args.batch_size
                    )
                intermediate_models = pool.map(train_model, texts)
                
                # Clean models and display corresponding files
                for i, model in enumerate(intermediate_models):
                    if not model:
                        intermediate_models.pop(i)
                        print("ERROR:Error training model on :", batch[i])

                # Combine model with previous batch models
                if combined_model:
                    combined_model = markovify.combine(
                    models=[combined_model] + intermediate_models
                    )
                else:
                    combined_model = markovify.combine(
                        models=intermediate_models
                        )
                # last_step += args.batch_size
                last_step += 1 # last_step is not step but epoch !

                output_name = args.output.split("/")[-1].split(".")[0]
                output_name = f"{output_name}_{last_step}.json"
                if last_step % args.save_step == 0 and last_step > 0:
                    if combined_model is not None:
                        save_model(output_name, combined_model)

                pbar.update(1)

    # Save final model
    save_model(output_name, combined_model)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Root path containing files with combolists")
    parser.add_argument("-o", "--output", default="model.json")
    parser.add_argument("-f", "--filter")
    parser.add_argument("-r", "--restore_path")
    parser.add_argument("--save_step", default=100, type=int)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--n_cpu", default=8)
    parser.add_argument("--keep_free_cpu", default=True)
    args = parser.parse_args()

    FILTER_PATTERN = args.filter

    main_loop(args)
