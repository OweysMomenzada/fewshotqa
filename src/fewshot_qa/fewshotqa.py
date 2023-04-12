from trainer import validate
from text_dataset import TextDataset
from transformers import BartTokenizer, BartForConditionalGeneration

from torch.utils.data import DataLoader
import utils
from utils import get_data
import pandas as pd
import json

REL_MODEL_PATH = "data_seed_results/128_42_qa_only/model_files"
MAX_GEN_LEN = 50


TEST_SAMPLE = {"context": "40\u00b048\u203247\u2033N 73\u00b057\u203227\u2033W\ufeff / \ufeff40.813\u00b0N 73.9575\u00b0W\ufeff / 40.813; -73.9575 La Salle Street is a street in West Harlem that runs just two blocks between Amsterdam Avenue and Claremont Avenue. West of Convent Avenue, 125th Street was re-routed onto the old Manhattan Avenue. The original 125th Street west of Convent Avenue was swallowed up to make the super-blocks where the low income housing projects now exist. La Salle Street is the only vestige of the original routing.", 
               "qas": [{
                        "question": "In which neighborhood does La Salle Street run?", "answers": ["<mask>"]
                }]}


class FewShotQA:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained(REL_MODEL_PATH)
        self.model = BartForConditionalGeneration.from_pretrained(REL_MODEL_PATH)


    def help_function(self, input):
        header = [{"header": {"dataset": "SQuAD", "split": "dev"}}]
        with open("tmp.jsonl", 'w') as f:
            f.write(json.dumps(header) + "\n")
            f.write(json.dumps(input) + "\n")

        train_srcs, train_trgs = get_data('tmp.jsonl', multi_answer=False)
        _, train_multi_trgs = get_data('tmp.jsonl', multi_answer=True)
        train_samples = list(zip(train_srcs, train_trgs, train_multi_trgs))
        train_df = pd.DataFrame(train_samples, columns=['source_text', 'target_text', 'multi_target'])

        val_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        }

        training_set = TextDataset(
            train_df,
            self.tokenizer,
            800, # MAX CONTEXT LEN
            min(1024, 1), # MAX TARGET LEN
            "source_text",
            "target_text",
        )

        test_loader = DataLoader(training_set, **val_params)
        
        pred, _ = validate( 0, 
                            self.tokenizer, 
                            self.model, 
                            "cpu", 
                            test_loader, 
                            max_gen_len=MAX_GEN_LEN)
        
        return utils.postprocess_preds(pred)

#qamodel = FewShotQA()

#print(qamodel.help_function(TEST_SAMPLE))