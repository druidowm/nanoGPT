import os
import torch
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Set the base directory for all data
BASE_DIR = '/scratch/owen/nanoGPT/openwebtext_char_to_token'
os.makedirs(BASE_DIR, exist_ok=True)

# Set the cache directory for datasets
os.environ['HF_DATASETS_CACHE'] = os.path.join('/scratch/owen/nanoGPT', 'hf_cache')

num_proc = 8
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    train_portion = 0.1

    full_train = dataset["train"]
    train_size = int(len(full_train) * train_portion)
    
    split_dataset = full_train.train_test_split(
        train_size=train_size,
        test_size=0.0005,
        seed=2357,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        example = "|START|" + example["text"]
        char_tokens = [ord(char) for char in example]
        ids = enc.encode_ordinary(example)
        token_starts = enc.decode_with_offsets(ids)[1]

        token_starts = torch.tensor(token_starts[1:], dtype=torch.long)
        tokens = torch.tensor(ids[1:], dtype=torch.long)

        output_tokens = torch.zeros(len(char_tokens), dtype=torch.long)-1
        output_tokens[token_starts] = tokens

        assert len(char_tokens) == len(output_tokens)
        out = {'chars': char_tokens, 'output_tokens': output_tokens, 'len': len(char_tokens)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits, char_to_token ",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(BASE_DIR, f'{split}.bin')
        dtype = np.uint16
        in_arr = np.memmap(filename, dtype=np.int32, mode='w+', shape=(arr_len,))
        out_arr = np.memmap(filename.replace('.bin', '_token_out.bin'), dtype=np.int32, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            in_arr_batch = np.concatenate(batch['chars'])
            out_arr_batch = np.concatenate(batch['output_tokens'])
            in_arr[idx : idx + len(in_arr_batch)] = in_arr_batch
            out_arr[idx : idx + len(out_arr_batch)] = out_arr_batch
            idx += len(in_arr_batch)
        in_arr.flush()
        out_arr.flush()

    print(f"Data processing complete. All files are stored in {BASE_DIR}")